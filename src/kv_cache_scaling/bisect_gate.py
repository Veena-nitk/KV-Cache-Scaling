"""Receipt and per-cohort budget gates for the separately preregistered bisect."""

from dataclasses import asdict
import json
from pathlib import Path
from uuid import UUID

from .coordinator_store import atomic_json, file_lock
from .gate import GateError, claim_launch, verify_receipt

LOWER = 114688
UPPER = 131072
RESOLUTION = 512
MAX_PER_ALLOCATOR = 6
ALLOCATORS = ("default", "expandable")


def load_verified_bisection(root, receipt_path):
    """Verify the real pushed receipt and the additional bisection commitment."""
    verified = verify_receipt(root, receipt_path)
    receipt = json.loads((Path(root) / receipt_path).read_text())
    prediction = json.loads((Path(root) / receipt["prediction_path"]).read_text())
    policy = prediction.get("acceptance_policy", {})
    expected = {
        "anchor_length": LOWER,
        "anchor_repeats": 3,
        "confirmation_upper_length": UPPER,
        "resolution_tokens": RESOLUTION,
        "max_oom_count_per_allocator": MAX_PER_ALLOCATOR,
        "max_oom_count_total": 2 * MAX_PER_ALLOCATOR,
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise GateError(f"Bisection receipt has wrong {key}")
    if prediction.get("gpu_measurements_authorized") is not False:
        raise GateError("Bisection prediction must remain a preregistration, not execution authority")
    original_id = None
    for path in (Path(root) / "prereg/final-2026-10-19/receipt.json",
                 Path(root) / "prereg/formal-2026-10-18-t4/receipt.json"):
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("experiment_id") == verified.experiment_id:
                original_id = verified.experiment_id
                break
    if original_id:
        raise GateError("Bisection must use a new experiment_id; the original OOM allowance is already consumed")
    required_inputs = {
        "scripts/bisect_boundary.py",
        "src/kv_cache_scaling/bisect_worker.py",
        "src/kv_cache_scaling/bisect_gate.py",
        "notebooks/BOUNDARY_COLAB.ipynb",
        "notebooks/BOUNDARY_COLAB.md",
    }
    if not required_inputs.issubset(prediction.get("input_sha256", {})):
        raise GateError("Bisection scripts are not frozen in the amendment prediction")
    return verified


def _budget_path(verified):
    return verified.state_root / ("paired_boundary_oom_budget-" + str(UUID(verified.experiment_id)) + ".json")


def initialize_budget(verified):
    path = _budget_path(verified)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            value = json.loads(path.read_text())
            if value.get("experiment_id") != verified.experiment_id or value.get("max_per_allocator") != MAX_PER_ALLOCATOR:
                raise GateError("Bisection OOM budget identity mismatch")
            return value
        value = {
            "schema_version": "paired-boundary-oom-budget-v1",
            "experiment_id": verified.experiment_id,
            "receipt_sha256": verified.receipt_sha256,
            "max_per_allocator": MAX_PER_ALLOCATOR,
            "max_total": 2 * MAX_PER_ALLOCATOR,
            "attempts": {name: 0 for name in ALLOCATORS},
            "oom_outcomes": {name: 0 for name in ALLOCATORS},
            "run_ids": {name: None for name in ALLOCATORS},
            "pending": None,
            "halted": {name: False for name in ALLOCATORS},
        }
        atomic_json(path, value)
        return value


def bind_allocator_run(verified, allocator, run_id):
    """Prevent resetting an allocator's attempt history with a new run UUID."""
    if allocator not in ALLOCATORS:
        raise GateError("Unknown allocator arm")
    run_id = str(UUID(run_id))
    path = _budget_path(verified)
    with file_lock(path.with_suffix(".lock")):
        if not path.exists():
            raise GateError("Bisection OOM budget was not initialized")
        value = json.loads(path.read_text())
        previous = value["run_ids"].get(allocator)
        if previous is None:
            value["run_ids"][allocator] = run_id
            atomic_json(path, value)
        elif previous != run_id:
            raise GateError("This allocator's run ID is already bound; new IDs cannot reset its budget")
        return value


def claim_deliberate_attempt(verified, request, run_id, point_key):
    """Spend one potential-OOM slot durably before any deliberate worker launch."""
    path = _budget_path(verified)
    with file_lock(path.with_suffix(".lock")):
        if not path.exists():
            raise GateError("Bisection OOM budget was not initialized")
        value = json.loads(path.read_text())
        allocator = request.allocator
        if value.get("pending") is not None:
            raise GateError("A previous deliberate dispatch is unresolved; do not retry")
        if value["halted"].get(allocator):
            raise GateError("This allocator arm has been stopped")
        if value["attempts"][allocator] >= MAX_PER_ALLOCATOR:
            raise GateError("Per-allocator deliberate OOM attempt cap reached")
        if sum(value["attempts"].values()) >= 2 * MAX_PER_ALLOCATOR:
            raise GateError("Total deliberate OOM attempt cap reached")
        value["attempts"][allocator] += 1
        value["pending"] = {"allocator": allocator, "run_id": str(UUID(run_id)),
                            "probe_id": request.probe_id, "point_key": point_key,
                            "length": request.length, "attempt_number": value["attempts"][allocator]}
        atomic_json(path, value)
        return value


def finish_attempt(verified, request, run_id, status):
    """Record the result; any unexpected failure halts the arm without retry."""
    path = _budget_path(verified)
    with file_lock(path.with_suffix(".lock")):
        value = json.loads(path.read_text())
        allocator = request.allocator
        pending = value.get("pending")
        if request.attempt_kind == "deliberate":
            if not pending or pending.get("probe_id") != request.probe_id or pending.get("run_id") != str(UUID(run_id)):
                raise GateError("Result does not match the consumed deliberate budget slot")
            value["pending"] = None
        if status == "cuda_oom":
            value["oom_outcomes"][allocator] += 1
            if request.attempt_kind == "safe":
                value["halted"][allocator] = True
        elif status != "success":
            value["halted"][allocator] = True
        if value["oom_outcomes"][allocator] > MAX_PER_ALLOCATOR:
            raise GateError("Recorded OOM outcomes exceeded the approved per-allocator cap")
        atomic_json(path, value)
        return value


def verify_bisection_launch(verified, request, run_id):
    """Worker-side authorization for one journaled bisection request."""
    run_id = str(UUID(run_id))
    if request.allocator not in ALLOCATORS or request.measurement != "primary":
        raise GateError("Request is outside the paired boundary protocol")
    if (request.length < LOWER or request.length > UPPER
            or request.length % RESOLUTION != 0
            or request.attempt_kind not in ("safe", "deliberate")):
        raise GateError("Request length or intent is outside the approved bisection domain")
    verified.check_request(request)
    run_dir = verified.state_root / "runs" / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (manifest.get("receipt_sha256") != verified.receipt_sha256
            or manifest.get("code_commit") != verified.code_commit
            or manifest.get("experiment_id") != verified.experiment_id
            or manifest.get("gpu_uuid") != verified.gpu_uuid
            or manifest.get("allocator") != request.allocator
            or manifest.get("execution_scope") != "paired_allocator_bisection"):
        raise GateError("Bisection manifest differs from verified receipt or request")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    starts = [row for row in events if row.get("event") == "probe_started" and row.get("probe_id") == request.probe_id]
    if len(starts) != 1 or starts[0].get("request") != asdict(request):
        raise GateError("No unique matching durable probe_started event")
    if any(row.get("probe_id") == request.probe_id and row.get("event") in ("probe_result", "probe_interrupted") for row in events):
        raise GateError("Probe has already completed or was interrupted")
    budget = initialize_budget(verified)
    allocator = request.allocator
    if budget.get("run_ids", {}).get(allocator) != run_id:
        raise GateError("Worker run ID is not the one bound to this allocator budget")
    if budget["halted"][allocator]:
        raise GateError("Allocator arm is stopped")
    if request.attempt_kind == "deliberate":
        pending = budget.get("pending") or {}
        if (pending.get("probe_id") != request.probe_id or pending.get("run_id") != run_id
                or pending.get("allocator") != allocator or pending.get("length") != request.length
                or pending.get("point_key") != starts[0].get("point_key")):
            raise GateError("Deliberate probe has no matching consumed bisection budget slot")
    elif request.length != LOWER:
        raise GateError("Safe probes are permitted only for the registered lower anchor")
    expected = next_probe_spec(events, allocator,
                               oom_count=budget["oom_outcomes"][allocator],
                               attempts=budget["attempts"][allocator])
    if expected.get("complete"):
        raise GateError("Boundary interval is already complete; no further worker is authorized")
    if any(getattr(request, key) != expected[key] for key in
           ("length", "attempt_kind", "repeat", "allocator")):
        raise GateError("Worker request is not the next preregistered bisection point")
    claim_launch(verified, request, run_id)


def next_probe_spec(events, allocator, *, oom_count=None, attempts=None):
    """Pure deterministic protocol state machine shared by parent and worker."""
    if allocator not in ALLOCATORS:
        raise GateError("Unknown allocator arm")
    if any(e.get("event") in ("probe_interrupted", "run_stopped", "run_deviation") for e in events):
        raise GateError("Run has a stop/interruption record")
    results = [e for e in events if e.get("event") == "probe_result"
               and e.get("request", {}).get("allocator") == allocator]
    idle = {e.get("probe_id") for e in events if e.get("event") == "idle_verified"}
    if any(e.get("probe_id") not in idle for e in results):
        raise GateError("Completed result lacks idle verification")
    if any(e.get("status") not in ("success", "cuda_oom") for e in results):
        raise GateError("A non-success/non-OOM result stops this allocator arm")
    observed_ooms = sum(e.get("status") == "cuda_oom" for e in results)
    observed_attempts = sum(e.get("request", {}).get("attempt_kind") == "deliberate" for e in results)
    if oom_count is not None and oom_count < observed_ooms:
        raise GateError("OOM budget is behind the event journal")
    if attempts is not None and attempts < observed_attempts:
        raise GateError("Attempt budget is behind the event journal")
    oom_count = observed_ooms if oom_count is None else oom_count
    attempts = observed_attempts if attempts is None else attempts

    anchors = [e for e in results if e["request"].get("length") == LOWER
               and e["request"].get("attempt_kind") == "safe"]
    if any(e.get("status") != "success" for e in anchors):
        raise GateError("Lower anchor failed; stop the arm without retry")
    repeats = [e["request"].get("repeat") for e in anchors]
    if len(repeats) != len(set(repeats)) or any(r not in (1, 2, 3) for r in repeats):
        raise GateError("Anchor repeats are duplicated or outside the registered set")
    if len(repeats) < 3:
        repeat = next(r for r in (1, 2, 3) if r not in repeats)
        return {"length": LOWER, "attempt_kind": "safe", "repeat": repeat,
                "phase": "anchor", "allocator": allocator}
    if len(repeats) != 3:
        raise GateError("There must be exactly three lower anchor repeats")

    confirmations = [e for e in results if e["request"].get("length") == UPPER]
    if len(confirmations) > 1:
        raise GateError("Duplicate upper-bound confirmations")
    if not confirmations:
        return {"length": UPPER, "attempt_kind": "deliberate", "repeat": 1,
                "phase": "confirmation", "allocator": allocator}
    if (confirmations[0]["request"].get("attempt_kind") != "deliberate"
            or confirmations[0].get("status") != "cuda_oom"):
        raise GateError("Upper endpoint did not produce the required typed CUDA OOM")

    lower, upper = LOWER, UPPER
    bisection = [e for e in results if e["request"].get("length") not in (LOWER, UPPER)]
    for e in bisection:
        req, length = e["request"], e["request"].get("length")
        if req.get("attempt_kind") != "deliberate" or not isinstance(length, int) or length % RESOLUTION:
            raise GateError("Bisection point has wrong intent or token resolution")
        midpoint = ((lower + upper) // (2 * RESOLUTION)) * RESOLUTION
        if length != midpoint:
            raise GateError("Journal does not follow the registered midpoint sequence")
        if e["status"] == "success":
            lower = length
        else:
            upper = length
    gap = upper - lower
    if gap <= RESOLUTION:
        return {"complete": True, "allocator": allocator, "last_success": lower,
                "first_cuda_oom": upper, "gap": gap, "oom_count": oom_count,
                "deliberate_attempts": attempts}
    if oom_count >= MAX_PER_ALLOCATOR or attempts >= MAX_PER_ALLOCATOR:
        raise GateError("OOM or deliberate attempt cap reached with unresolved bracket")
    midpoint = ((lower + upper) // (2 * RESOLUTION)) * RESOLUTION
    if not lower < midpoint < upper:
        raise GateError("No valid midpoint remains before the registered resolution")
    return {"length": midpoint, "attempt_kind": "deliberate",
            "repeat": len(bisection) + 1, "phase": "bisection", "allocator": allocator,
            "bounds": (lower, upper), "oom_count": oom_count,
            "deliberate_attempts": attempts}


__all__ = ["load_verified_bisection", "initialize_budget", "claim_deliberate_attempt",
           "finish_attempt", "verify_bisection_launch", "bind_allocator_run", "next_probe_spec",
           "LOWER", "UPPER", "RESOLUTION", "MAX_PER_ALLOCATOR"]

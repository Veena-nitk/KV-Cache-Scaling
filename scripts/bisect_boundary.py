"""Run exactly one receipt-gated step of the paired allocator boundary bisect."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kv_cache_scaling.bisect_gate import (
    LOWER, UPPER, RESOLUTION, MAX_PER_ALLOCATOR, claim_deliberate_attempt,
    bind_allocator_run, finish_attempt, initialize_budget, load_verified_bisection,
    next_probe_spec,
)
from kv_cache_scaling.coordinator_store import RunStore, atomic_json, file_lock, now
from kv_cache_scaling.gate import digest, json_bytes
from kv_cache_scaling.probe import ProbeRequest
from kv_cache_scaling.processes import worker_env
from kv_cache_scaling.sweep import IDLE_TOLERANCE, DriverObserver, collect_process, point_key


def get_next_probe_spec(events, allocator, *, oom_count=None, attempts=None):
    """Return the next expected probe or a completed interval; fail closed on drift."""
    try:
        return next_probe_spec(events, allocator, oom_count=oom_count, attempts=attempts)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def bind_session_gpu(verified, observer):
    """Bind both allocator arms to the receipt's UUID, driver, and physical memory."""
    observed = observer.read()
    if observed["uuid"].lower() != verified.gpu_uuid.lower():
        raise RuntimeError("Current Colab GPU UUID differs from the preregistered inventory")
    path = verified.state_root / ("paired_boundary_gpu-" + verified.experiment_id + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"experiment_id": verified.experiment_id, "gpu_uuid": observed["uuid"],
                "driver_version": observed["driver_version"], "total_bytes": observed["total_bytes"],
                "receipt_sha256": verified.receipt_sha256}
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            saved = json.loads(path.read_text())
            if saved != identity:
                raise RuntimeError("Paired allocator GPU identity changed; stop the cohort")
        else:
            atomic_json(path, identity)
    return observed


def ensure_expandable_follows_default(verified, allocator, default_run_id):
    if allocator == "default":
        if default_run_id:
            raise ValueError("Do not supply --default-run-id for the default arm")
        return
    if not default_run_id:
        raise ValueError("Expandable arm requires the completed default --default-run-id")
    default_run_id = str(UUID(default_run_id))
    root = verified.state_root / "runs" / default_run_id
    rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    completion = [r for r in rows if r.get("event") == "allocator_bisection_complete"
                  and r.get("allocator") == "default"]
    if len(completion) != 1:
        raise RuntimeError("Default arm must have one completed bracket before expandable starts")


def execute_single_probe(store, observer, spec, verified, run_id, local_root):
    before = observer.read()
    if before["uuid"].lower() != verified.gpu_uuid.lower():
        raise RuntimeError("GPU identity changed before worker launch")
    request = ProbeRequest(
        probe_id=str(uuid4()), length=spec["length"], gpu_uuid=verified.gpu_uuid,
        protocol_sha256=verified.protocol_sha256, config_sha256=verified.config_sha256,
        allocator=spec["allocator"], repeat=spec["repeat"],
        attempt_kind=spec["attempt_kind"], measurement="primary")
    key = point_key(verified, request.length, request.allocator, request.repeat,
                    request.attempt_kind, request.measurement)
    store.artifact(f"requests/{request.probe_id}.json", json_bytes(asdict(request)) + b"\n")
    if request.attempt_kind == "deliberate":
        claim_deliberate_attempt(verified, request, run_id, key)
    store.append("probe_started", probe_id=request.probe_id, point_key=key,
                 request=asdict(request), phase=spec["phase"], idle_before=before)

    def command(output):
        return [sys.executable, "-m", "kv_cache_scaling.bisect_worker",
                "--root", str(verified.root), "--receipt", verified.receipt_path,
                "--run-id", run_id,
                "--request", str(store.local / "requests" / (request.probe_id + ".json")),
                "--output", str(output)]

    protocol = json.loads((verified.root / "configs/experiment.json").read_text())
    try:
        record = collect_process(store, request, command,
                                 worker_env(request.allocator, verified.root),
                                 verified.root, protocol["per_probe_timeout_seconds"])
    except BaseException:
        store.append("probe_interrupted", probe_id=request.probe_id, point_key=key,
                     reason="No automatic retry; deliberate budget remains consumed")
        store.append("run_stopped", reason="Probe interrupted; inspect capsule before any new cohort")
        raise

    finish_attempt(verified, request, run_id, record.get("status"))
    row = store.append("probe_result", probe_id=request.probe_id, point_key=key,
                       request=asdict(request), phase=spec["phase"], **record)
    after = observer.read()
    if (after["uuid"] != before["uuid"] or after["driver_version"] != before["driver_version"]
            or after["total_bytes"] != before["total_bytes"]
            or abs(after["used_bytes"] - before["used_bytes"]) > IDLE_TOLERANCE):
        store.append("run_deviation", probe_id=request.probe_id, idle_after=after)
        store.append("run_stopped", reason="GPU identity or idle memory drift")
        raise RuntimeError("GPU failed the post-worker idle/identity check")
    store.append("idle_verified", probe_id=request.probe_id, idle_after=after)
    if record.get("status") not in ("success", "cuda_oom"):
        store.append("run_stopped", reason="Worker returned a non-success/non-OOM status")
        raise RuntimeError("Worker failed; arm stopped")
    if record.get("status") == "cuda_oom" and request.attempt_kind != "deliberate":
        store.append("run_stopped", reason="Unexpected OOM during safe anchor; arm stopped")
        raise RuntimeError("Unexpected anchor OOM; arm stopped")
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="Completed pushed amendment receipt")
    parser.add_argument("--run-id", required=True, help="Persistent run UUID for this allocator arm")
    parser.add_argument("--local-root", required=True, help="Local checkpoint staging directory")
    parser.add_argument("--allocator", choices=("default", "expandable"), required=True)
    parser.add_argument("--default-run-id", help="Completed default-arm run UUID for the expandable arm")
    args = parser.parse_args(argv)

    verified = load_verified_bisection(ROOT, args.receipt)
    run_id = str(UUID(args.run_id))
    ensure_expandable_follows_default(verified, args.allocator, args.default_run_id)
    observer = DriverObserver(verified)
    bind_session_gpu(verified, observer)
    initialize_budget(verified)
    bind_allocator_run(verified, args.allocator, run_id)
    persistent_root = verified.state_root
    identity = {
        "schema_version": "paired-allocator-bisection-v1",
        "receipt_sha256": verified.receipt_sha256,
        "prediction_commit": verified.prediction_commit,
        "code_commit": verified.code_commit,
        "environment_fingerprint": verified.environment_fingerprint,
        "gpu_uuid": verified.gpu_uuid,
        "protocol_sha256": verified.protocol_sha256,
        "config_sha256": verified.config_sha256,
        "experiment_id": verified.experiment_id,
        "execution_scope": "paired_allocator_bisection",
        "allocator": args.allocator,
        "checkpoint_each_probe": True,
        "created_at_utc": now(),
    }

    with RunStore(args.local_root, persistent_root, run_id) as store:
        manifest = store.local / "manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if {k: v for k, v in previous.items() if k != "created_at_utc"} != {
                    k: v for k, v in identity.items() if k != "created_at_utc"}:
                raise RuntimeError("Run manifest differs from the approved receipt or current arm")
        else:
            store.manifest(identity)
        finished = [e for e in store.events if e.get("event") == "allocator_bisection_complete"
                    and e.get("allocator") == args.allocator]
        if finished:
            result = finished[-1]
            print(json.dumps({"status": "COMPLETE", **{k: result[k] for k in
                             ("allocator", "last_success", "first_cuda_oom", "gap", "oom_count")}}))
            return 0
        orphans = store.orphaned()
        if orphans:
            store.append("run_stopped", reason="Orphaned start requires manual review; never redispatch")
            raise RuntimeError("An incomplete worker start exists; no automatic retry")
        if any(e.get("event") == "run_stopped" for e in store.events):
            raise RuntimeError("This allocator run is stopped; no automatic retry")

        current = initialize_budget(verified)
        spec = get_next_probe_spec(store.events, args.allocator,
                                   oom_count=current["oom_outcomes"][args.allocator],
                                   attempts=current["attempts"][args.allocator])
        if spec.get("complete"):
            store.append("allocator_bisection_complete", allocator=args.allocator,
                         last_success=spec["last_success"], first_cuda_oom=spec["first_cuda_oom"],
                         gap=spec["gap"], oom_count=spec["oom_count"],
                         deliberate_attempts=spec["deliberate_attempts"])
            print(json.dumps({"status": "COMPLETE", **spec}))
            return 0

        row = execute_single_probe(store, observer, spec, verified, run_id, args.local_root)
        print(json.dumps({"status": "STEP_COMPLETE", "allocator": args.allocator,
                          "length": row["request"]["length"], "probe_status": row["status"],
                          "phase": spec["phase"], "probe_id": row["probe_id"]}))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

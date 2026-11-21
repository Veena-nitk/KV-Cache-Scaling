import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "bisect_boundary", Path(__file__).resolve().parents[1] / "scripts/bisect_boundary.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def add_result(events, length, status, *, repeat=1, attempt="safe", allocator="default"):
    probe_id = f"p{len(events)}"
    events.extend([
        {"event": "probe_result", "probe_id": probe_id,
         "request": {"allocator": allocator, "length": length,
                     "attempt_kind": attempt, "repeat": repeat}, "status": status},
        {"event": "idle_verified", "probe_id": probe_id},
    ])


class BisectBoundaryTests(unittest.TestCase):
    def test_anchor_repeats(self):
        events = []
        for repeat in (1, 2, 3):
            result = m.get_next_probe_spec(events, "default")
            self.assertEqual((result["length"], result["phase"], result["repeat"]),
                             (114688, "anchor", repeat))
            add_result(events, 114688, "success", repeat=repeat)
        result = m.get_next_probe_spec(events, "default")
        self.assertEqual(result["phase"], "confirmation")

    def test_anchor_oom_stops_without_retry(self):
        events = []
        add_result(events, 114688, "cuda_oom", repeat=1)
        with self.assertRaisesRegex(RuntimeError, "anchor"):
            m.get_next_probe_spec(events, "default")

    def test_upper_bound_must_be_typed_cuda_oom(self):
        events = []
        for repeat in (1, 2, 3):
            add_result(events, 114688, "success", repeat=repeat)
        add_result(events, 131072, "error", attempt="deliberate")
        with self.assertRaisesRegex(RuntimeError, "non-success/non-OOM"):
            m.get_next_probe_spec(events, "default")

    def test_bisection_completes_at_512(self):
        events = []
        for repeat in (1, 2, 3):
            add_result(events, 114688, "success", repeat=repeat)
        add_result(events, 131072, "cuda_oom", attempt="deliberate")
        sequence = [(122880, "success"), (126976, "cuda_oom"),
                    (124928, "success"), (125952, "cuda_oom"),
                    (125440, "success")]
        for expected, status in sequence:
            result = m.get_next_probe_spec(events, "default")
            self.assertEqual(result["length"], expected)
            add_result(events, expected, status, attempt="deliberate")
        result = m.get_next_probe_spec(events, "default")
        self.assertTrue(result["complete"])
        self.assertEqual((result["last_success"], result["first_cuda_oom"], result["gap"]),
                         (125440, 125952, 512))

    def test_budget_cap_stops_unresolved_interval(self):
        events = []
        for repeat in (1, 2, 3):
            add_result(events, 114688, "success", repeat=repeat)
        add_result(events, 131072, "cuda_oom", attempt="deliberate")
        with self.assertRaisesRegex(RuntimeError, "cap"):
            m.get_next_probe_spec(events, "default", oom_count=6, attempts=6)

    def test_missing_idle_check_stops(self):
        events = [{"event": "probe_result", "probe_id": "x",
                   "request": {"allocator": "default", "length": 114688,
                               "attempt_kind": "safe", "repeat": 1},
                   "status": "success"}]
        with self.assertRaisesRegex(RuntimeError, "idle"):
            m.get_next_probe_spec(events, "default")


class PairedIntegrationTests(unittest.TestCase):
    """Drive both real entry points, substituting only receipt, GPU and transport."""

    def test_notebook_runs_both_arms_and_checkpoints_without_skipping_attempt_six(self):
        from contextlib import redirect_stdout
        from datetime import datetime, timezone
        import hashlib
        import io
        import json
        import subprocess
        from tempfile import TemporaryDirectory
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        from uuid import uuid4
        from zipfile import ZipFile

        from kv_cache_scaling import bisect_gate, bisect_worker, gate
        from kv_cache_scaling.probe import ProbeRequest
        from test_coordinator import GitFixture, SyntheticRunner

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = GitFixture(tmp)
            repo = fixture.root
            policy = dict(anchor_length=114688, anchor_repeats=3,
                          confirmation_upper_length=131072, resolution_tokens=512,
                          max_oom_count_per_allocator=6, max_oom_count_total=12)
            approval = "SYNTHETIC FIXTURE ONLY"
            prediction = fixture.prediction
            prediction.update(acceptance_policy=policy, gpu_measurements_authorized=False,
                              experiment_id=fixture.receipt["experiment_id"], ta_approval_reference=approval)
            for name in ("scripts/bisect_boundary.py", "src/kv_cache_scaling/bisect_worker.py",
                         "src/kv_cache_scaling/bisect_gate.py", "notebooks/BOUNDARY_COLAB.ipynb",
                         "notebooks/BOUNDARY_COLAB.md", "requirements-gpu.txt"):
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((m.ROOT / name).read_bytes())
                prediction["input_sha256"][name] = gate.digest(path.read_bytes())
            prediction_name = "prereg/paired-prediction.json"
            fixture.write(prediction_name, prediction)
            fixture.commit("synthetic paired prediction")
            prediction_sha = fixture.git("rev-parse", "HEAD").decode().strip()
            receipt = dict(fixture.receipt, prediction_path=prediction_name,
                           prediction_commit_sha=prediction_sha, ta_approval_reference=approval,
                           prediction_sha256=gate.digest((repo / prediction_name).read_bytes()),
                           remote_commit_url=fixture.receipt["remote_commit_url"].rsplit("/", 1)[0] + "/" + prediction_sha,
                           remote_receipt_observed_at_utc=datetime.now(timezone.utc).isoformat())
            receipt_name = "prereg/paired-receipt.json"
            fixture.write(receipt_name, receipt)
            fixture.commit("synthetic subsequent paired receipt")
            real_git = gate.git
            head = fixture.git("rev-parse", "HEAD").decode().strip()
            def remote(local_root, *args):
                if args[0] == "ls-remote":
                    return (head + "\trefs/heads/main\n").encode()
                return real_git(local_root, *args)

            calls = []
            downloads = []
            acknowledgements = []
            targets = {"default": 120000, "expandable": 126000}

            def process(command, **kwargs):
                argv = command[3:]
                request_path = Path(argv[argv.index("--request") + 1])
                output_path = Path(argv[argv.index("--output") + 1])
                request = ProbeRequest.from_dict(json.loads(request_path.read_bytes()))
                # The next dispatch follows a recorded acknowledgement of the previous capsule.
                if calls:
                    self.assertGreaterEqual(len(acknowledgements), len(calls))
                def engine(req, *, root, verified):
                    self.assertEqual(req, request)
                    self.assertEqual(verified.code_commit, head)
                    status = "success" if req.length <= targets[req.allocator] else "cuda_oom"
                    calls.append(req)
                    synthetic_output = output_path.with_suffix(".fixture.json")
                    SyntheticRunner(status)([*command[:-1], str(synthetic_output)], **kwargs)
                    record = json.loads(synthetic_output.read_bytes())
                    synthetic_output.unlink()
                    record.update(stage="forward" if status == "cuda_oom" else "complete",
                                  oom_classification="deliberate_request", warnings=[])
                    return record
                with patch.object(bisect_worker, "_execute_probe", side_effect=engine):
                    code = bisect_worker.main(argv)
                return {"process_status": "exited", "returncode": code, "stdout": b"synthetic",
                        "stderr": b"", "logs": {"stdout": {}, "stderr": {}}}

            def driver(verified):
                return SimpleNamespace(read=lambda: {"uuid": verified.gpu_uuid, "driver_version": verified.driver_version,
                    "total_bytes": verified.total_bytes, "used_bytes": 0})

            def run_source(*args, **kwargs):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = m.main(args[1:])
                return SimpleNamespace(stdout=stdout.getvalue(), stderr="", returncode=code)

            def download(path):
                capsule = Path(path)
                with ZipFile(capsule) as z:
                    manifest = json.loads(z.read("MANIFEST.json"))
                    self.assertEqual(set(z.namelist()), set(manifest) | {"MANIFEST.json"})
                    for name, digest in manifest.items():
                        self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), digest)
                downloads.append(capsule)

            def acknowledge(_prompt):
                sha = hashlib.sha256(downloads[-1].read_bytes()).hexdigest()
                acknowledgements.append(sha)
                return sha

            google = ModuleType("google")
            colab = ModuleType("google.colab")
            colab.files = SimpleNamespace(download=download)
            notebook = json.loads((repo / "notebooks/BOUNDARY_COLAB.ipynb").read_bytes())
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            self.assertEqual(len(cells), 5)
            for code in cells:
                compile(code, "<colab>", "exec")
            gpu = fixture.inventory["gpu_identity_csv"].split(",")[0].strip()
            env = dict(REPO=repo, RECEIPT=receipt_name, receipt=receipt, receipt_path=repo / receipt_name,
                       SESSION_GPU_UUID=gpu, run_source=run_source, input=acknowledge, json=json,
                       subprocess=SimpleNamespace(check_output=lambda *_a, **_kw: gpu),
                       hashlib=hashlib, uuid=__import__("uuid"), os=__import__("os"),
                       pathlib=__import__("pathlib"), zipfile=__import__("zipfile"))
            real_collect = m.collect_process
            with (patch.object(gate, "git", side_effect=remote),
                  patch.object(m, "ROOT", repo),
                  patch.object(m, "DriverObserver", side_effect=driver),
                  patch.object(m, "collect_process", side_effect=lambda *args: real_collect(*args, runner=process)),
                  patch.dict("sys.modules", {"google": google, "google.colab": colab}),
                  redirect_stdout(io.StringIO())):
                verified = bisect_gate.load_verified_bisection(repo, receipt_name)
                for code in cells[1:]:
                    exec(code.replace("/content/kvcache-paired-", str(root / "colab-")), env)
                self.assertEqual(18, len(calls))
                self.assertEqual(21, len(acknowledgements))
                self.assertEqual({"default": 6, "expandable": 6}, bisect_gate.initialize_budget(verified)["attempts"])
                self.assertEqual(6144, env["summary"]["expandable_delta_tokens"])
                self.assertEqual(512, env["summary"]["default"]["gap"])
                self.assertEqual(512, env["summary"]["expandable"]["gap"])
                self.assertFalse((fixture.state / "global_deliberate_attempt.json").exists())
                with self.assertRaises(FileExistsError):
                    exec(cells[2], env)
                self.assertEqual(18, len(calls))
                with ZipFile(downloads[-1]) as final:
                    self.assertIn("audit/bisection_summary.json", final.namelist())
                    self.assertEqual(20, len(final.read("audit/checkpoint-acknowledgements.jsonl").splitlines()))


if __name__ == "__main__":
    unittest.main()

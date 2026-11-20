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


if __name__ == "__main__":
    unittest.main()

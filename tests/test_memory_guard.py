import unittest

from experiment_memory_guard import choose_memory_budget, current_process_memory_kb


class MemoryGuardTests(unittest.TestCase):
    def test_oom_backoff_reduces_workers_and_batch(self):
        decision = choose_memory_budget(
            requested_batch_size=32,
            requested_max_workers=4,
            status={"state": "error", "last_rc": "137"},
            meminfo={"MemTotal": 32768000, "MemAvailable": 12000000},
        )
        self.assertEqual(decision.batch_size, 12)
        self.assertEqual(decision.max_workers, 1)
        self.assertEqual(decision.memory_pressure_level, "critical")
        self.assertIn("oom_backoff", decision.reason)

    def test_confirmation_cycles_are_small(self):
        decision = choose_memory_budget(
            requested_batch_size=24,
            requested_max_workers=2,
            status={"state": "success", "last_rc": "0"},
            meminfo={"MemTotal": 32768000, "MemAvailable": 20000000},
            cycle_mode="confirmation",
            confirmation_required=True,
        )
        self.assertEqual(decision.batch_size, 8)
        self.assertEqual(decision.max_workers, 1)
        self.assertIn("confirmation_or_holdout_small_batch", decision.reason)

    def test_large_search_caps_workers(self):
        decision = choose_memory_budget(
            requested_batch_size=96,
            requested_max_workers=6,
            status={"state": "success", "last_rc": "0"},
            meminfo={"MemTotal": 32768000, "MemAvailable": 28000000},
            large_search_mode=True,
        )
        self.assertEqual(decision.batch_size, 24)
        self.assertEqual(decision.max_workers, 4)
        self.assertIn("large_search_worker_cap", decision.reason)

    def test_conservative_memory_pressure_allows_multiple_workers(self):
        # mem_ratio ~0.55 (< 0.60): batch_size capped at 24, workers capped at 3
        decision = choose_memory_budget(
            requested_batch_size=24,
            requested_max_workers=6,
            status={"state": "success", "last_rc": "0"},
            meminfo={"MemTotal": 32768000, "MemAvailable": 18000000},
        )
        self.assertEqual(decision.batch_size, 24)
        self.assertEqual(decision.max_workers, 3)
        self.assertIn("conservative_memory_pressure", decision.reason)

    def test_current_process_memory_snapshot_is_available(self):
        snapshot = current_process_memory_kb()
        self.assertIn("rss_kb", snapshot)
        self.assertIn("peak_rss_kb", snapshot)
        self.assertIn("tree_rss_kb", snapshot)
        self.assertIsNotNone(snapshot["rss_kb"])


if __name__ == "__main__":
    unittest.main()

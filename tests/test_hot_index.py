from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_best_results import latest_non_empty_batch
from experiment_dashboard import build_best_results_dashboard
from experiment_hot_index import (
    archive_hot_batch_summaries,
    archive_hot_experiment_summaries,
    compact_hot_index_storage,
    load_hot_batch_summary,
    load_hot_experiment_index,
    load_hot_summary_item,
    refresh_hot_index_reports,
    upsert_batch_summary,
    upsert_experiment_summary,
)


class HotIndexTests(unittest.TestCase):
    def _write_archive_index(self, base_dir: Path, rows: list[dict[str, object]]) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(base_dir / "index.csv", index=False)

    def test_hot_index_bootstraps_bounded_rows_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            rows = []
            for idx in range(5):
                rows.append(
                    {
                        "experiment_id": f"momentum_{idx}",
                        "timestamp_utc": f"2026-04-14T00:00:0{idx}+00:00",
                        "strategy_family": "momentum",
                        "config_hash": f"m{idx}",
                        "status": "success",
                        "objective_score": 1.0 + idx,
                        "viable": True,
                        "result_dir": f"runs/momentum_{idx}",
                    }
                )
            self._write_archive_index(experiments_dir, rows)

            frame = load_hot_experiment_index(str(experiments_dir), families=["momentum"], recent_limit_per_family=2)
            self.assertTrue((experiments_dir / "hot_index.sqlite3").exists())
            # The 2 most-recent rows are always included via the recency window.
            ids = frame["experiment_id"].tolist()
            self.assertIn("momentum_3", ids)
            self.assertIn("momentum_4", ids)
            # Top-scorers from index.csv are augmented in; with 5 rows and pin_top=20
            # all 5 distinct config_hashes end up represented.
            self.assertEqual(len(frame["config_hash"].dropna().unique()), 5)

    def test_hot_batch_summary_is_used_by_latest_non_empty_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            upsert_batch_summary(
                str(experiments_dir),
                {
                    "batch_id": "batch_hot",
                    "timestamp_utc": "2026-04-14T00:00:00+00:00",
                    "total_executed": 3,
                    "total_sampled": 3,
                    "total_skipped": 0,
                    "total_failed": 0,
                    "status_counts": {"success": 3},
                    "family_summary": {},
                    "proposal_metadata": {},
                    "throughput_diagnostics": {},
                },
            )
            latest = latest_non_empty_batch(str(experiments_dir))
            self.assertIsNotNone(latest)
            self.assertEqual(latest["batch_id"], "batch_hot")
            self.assertEqual(latest["executed_count"], 3)

    def test_refresh_hot_index_writes_dashboard_and_family_scorecards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            upsert_experiment_summary(
                str(experiments_dir),
                {
                    "experiment_id": "momentum_a",
                    "timestamp_utc": "2026-04-14T00:00:00+00:00",
                    "strategy_family": "momentum",
                    "config_hash": "m_a",
                    "batch_id": "batch_a",
                    "status": "success",
                    "objective_score": 1.2,
                    "viable": True,
                    "source_type": "classical",
                    "template_id": "tmpl",
                    "strategy_type": "classical",
                    "result_dir": "runs/momentum_a",
                },
            )
            upsert_batch_summary(
                str(experiments_dir),
                {
                    "batch_id": "batch_a",
                    "timestamp_utc": "2026-04-14T00:00:00+00:00",
                    "total_executed": 1,
                    "total_sampled": 1,
                    "total_skipped": 0,
                    "total_failed": 0,
                    "status_counts": {"success": 1},
                    "family_summary": {},
                    "proposal_metadata": {},
                    "throughput_diagnostics": {},
                },
            )
            result = refresh_hot_index_reports(base_dir=str(experiments_dir), families=["momentum"], recent_limit_per_family=8)
            self.assertIn("dashboard", result)
            self.assertIn("family_scorecards", result)
            dashboard = load_hot_summary_item(str(experiments_dir), "best_results_dashboard", "global")
            scorecard = load_hot_summary_item(str(experiments_dir), "family_scorecard", "momentum")
            self.assertIsInstance(dashboard, dict)
            self.assertIsInstance(scorecard, dict)
            self.assertEqual(scorecard["family"], "momentum")
            self.assertGreaterEqual(int(scorecard["total_experiments"]), 1)
            built = build_best_results_dashboard(base_dir=str(experiments_dir), families=["momentum"], include_idea_yield=False)
            self.assertGreaterEqual(built.counts["official_result_rows"], 1)
            self.assertEqual(built.latest_non_empty_batch["batch_id"], "batch_a")

    def test_archive_compaction_reduces_hot_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            for idx in range(6):
                upsert_experiment_summary(
                    str(experiments_dir),
                    {
                        "experiment_id": f"momentum_{idx}",
                        "timestamp_utc": f"2026-04-14T00:00:0{idx}+00:00",
                        "strategy_family": "momentum",
                        "config_hash": f"m{idx}",
                        "batch_id": f"batch_{idx}",
                        "status": "success",
                        "objective_score": 1.0 + idx,
                        "viable": True,
                        "result_dir": f"runs/momentum_{idx}",
                    },
                    keep_recent_per_family=6,
                )
            # pin_top_per_family=1 so only the single best (momentum_5) is pinned;
            # recency window of 2 keeps momentum_4 and momentum_5; the remaining
            # 4 (momentum_0..3) fall outside both and should be archived.
            counts = archive_hot_experiment_summaries(str(experiments_dir), keep_recent_per_family=2, pin_top_per_family=1)
            self.assertGreaterEqual(counts["archived_count"], 4)
            frame = load_hot_experiment_index(str(experiments_dir), families=["momentum"], recent_limit_per_family=10)
            # After archive (2 recency + 1 pinned best already in recency = 2 in SQLite),
            # augmentation from index.csv injects missing top-scorers: with pin_top=20
            # (default for load) up to 20 top rows from index.csv are recovered.
            ids = set(frame["experiment_id"].tolist())
            self.assertIn("momentum_4", ids)
            self.assertIn("momentum_5", ids)

    def test_batch_compaction_reduces_payloads_and_retains_recent_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            for idx in range(4):
                upsert_batch_summary(
                    str(experiments_dir),
                    {
                        "batch_id": f"batch_{idx}",
                        "timestamp_utc": f"2026-04-14T00:00:0{idx}+00:00",
                        "total_executed": idx,
                        "total_sampled": idx + 1,
                        "total_skipped": 1,
                        "total_failed": 0,
                        "status_counts": {"success": idx},
                        "family_summary": {},
                        "proposal_metadata": {
                            "runtime_decision": {
                                "decision_id": f"runtime_{idx}",
                                "cycle_mode": "confirmation",
                                "large_nested_payload": "x" * 100_000,
                            }
                        },
                        "throughput_diagnostics": {},
                        "large_nested_payload": "x" * 100_000,
                    },
                )
            with sqlite3.connect(experiments_dir / "hot_index.sqlite3") as conn:
                conn.execute(
                    "UPDATE batch_summaries SET summary_json = ?, proposal_metadata_json = ? WHERE batch_id = ?",
                    (
                        '{"batch_id":"batch_3","large_nested_payload":"' + ("x" * 100_000) + '"}',
                        '{"runtime_decision":{"decision_id":"runtime_3","large_nested_payload":"' + ("x" * 100_000) + '"}}',
                        "batch_3",
                    ),
                )
                conn.commit()

            result = compact_hot_index_storage(str(experiments_dir), keep_recent_batches=2, max_json_bytes=1024, vacuum=False)
            self.assertGreaterEqual(result["compacted_batch_rows"], 1)
            self.assertEqual(result["retained_batch_rows"], 2)
            self.assertEqual(result["archived_batch_rows"], 2)

            latest = load_hot_batch_summary(str(experiments_dir))
            self.assertEqual(latest["batch_id"], "batch_3")
            self.assertNotIn("large_nested_payload", latest)

            with sqlite3.connect(experiments_dir / "hot_index.sqlite3") as conn:
                row_count = conn.execute("SELECT COUNT(*) FROM batch_summaries").fetchone()[0]
                max_summary_size = conn.execute("SELECT MAX(length(summary_json)) FROM batch_summaries").fetchone()[0]
            self.assertEqual(row_count, 2)
            self.assertLess(max_summary_size, 10_000)

    def test_archive_hot_batch_summaries_keeps_recent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            for idx in range(3):
                upsert_batch_summary(
                    str(experiments_dir),
                    {
                        "batch_id": f"batch_{idx}",
                        "timestamp_utc": f"2026-04-14T00:00:0{idx}+00:00",
                        "total_executed": 1,
                        "total_sampled": 1,
                        "total_skipped": 0,
                        "total_failed": 0,
                    },
                )
            counts = archive_hot_batch_summaries(str(experiments_dir), keep_recent=1)
            self.assertEqual(counts["retained_count"], 1)
            self.assertEqual(counts["archived_count"], 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

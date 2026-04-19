from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_hot_index import (
    backfill_batch_metadata_from_artifacts,
    cleanup_validated_batch_artifacts,
    load_hot_batch_index,
    load_hot_batch_summary_by_id,
    validate_hot_batch_metadata,
)


def _write_batch(base_dir: Path, batch_id: str, *, timestamp: str, payload_extra: dict | None = None) -> Path:
    batch_dir = base_dir / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "experiment_id": f"momentum_a_{batch_id}",
            "strategy_family": "momentum",
            "config_hash": "hash_a",
            "status": "success",
            "objective_score": 1.25,
            "viable": True,
            "beats_baseline_objective": True,
            "source_type": "idea_seed",
            "template_id": "momentum_balanced_default",
            "strategy_type": "classical",
        },
        {
            "experiment_id": f"momentum_b_{batch_id}",
            "strategy_family": "momentum",
            "config_hash": "hash_b",
            "status": "error",
            "objective_score": None,
            "viable": False,
            "beats_baseline_objective": False,
            "source_type": "idea_seed",
            "template_id": "momentum_crash_robust",
            "strategy_type": "classical",
        },
    ]
    pd.DataFrame(rows).to_csv(batch_dir / "raw_results.csv", index=False)
    pd.DataFrame(rows).to_csv(batch_dir / "leaderboard.csv", index=False)
    payload = {
        "batch_id": batch_id,
        "timestamp_utc": timestamp,
        "strategy_families": ["momentum"],
        "sampler_type": "random",
        "execution_mode": "parallel",
        "max_workers": 2,
        "source_proposal_id": batch_id.removesuffix("_batch"),
        "total_sampled": 2,
        "total_executed": 1,
        "total_skipped": 0,
        "total_failed": 1,
        "status_counts": {"success": 1, "error": 1},
        "family_summary": {
            "momentum": {
                "results": 2,
                "executed": 1,
                "skipped": 0,
                "failed": 1,
                "best_objective_score": 1.25,
                "best_experiment_id": f"momentum_a_{batch_id}",
                "source_type_counts": {"idea_seed": 2},
                "strategy_type_counts": {"classical": 2},
                "template_counts": {"momentum_balanced_default": 1, "momentum_crash_robust": 1},
            }
        },
        "proposal_quality": {"status": "ok", "candidate_count": 2},
        "proposal_metadata": {
            "history": [{"blob": "x" * 1000} for _ in range(120)],
            "branch_budgets": {"momentum": [{"blob": "y" * 1000} for _ in range(120)]},
        },
        "leaderboard_path": str(batch_dir / "leaderboard.csv"),
        "raw_results_path": str(batch_dir / "raw_results.csv"),
        "summary_path": str(batch_dir / "summary.json"),
    }
    if payload_extra:
        payload.update(payload_extra)
    (batch_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return batch_dir


class BatchHotMetadataCleanupTests(unittest.TestCase):
    def test_backfill_batch_metadata_from_artifacts_writes_queryable_sqlite_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            _write_batch(base_dir, "proposal_old_batch", timestamp="2026-04-14T00:00:00+00:00")

            result = backfill_batch_metadata_from_artifacts(str(base_dir))
            self.assertEqual(result["written_count"], 1)

            row = load_hot_batch_summary_by_id(str(base_dir), "proposal_old_batch")
            self.assertIsNotNone(row)
            self.assertEqual(row["batch_id"], "proposal_old_batch")
            self.assertEqual(row["total_sampled"], 2)
            self.assertEqual(row["total_executed"], 1)
            self.assertEqual(row["status_counts"]["success"], 1)
            self.assertEqual(row["family_summary"]["momentum"]["best_experiment_id"], "momentum_a_proposal_old_batch")
            self.assertEqual(row["top_results"][0]["objective_score"], 1.25)

            validation = validate_hot_batch_metadata(str(base_dir), "proposal_old_batch")
            self.assertEqual(validation["valid_count"], 1)
            self.assertEqual(validation["missing_db_count"], 0)
            self.assertEqual(validation["mismatch_count"], 0)

            frame = load_hot_batch_index(str(base_dir))
            self.assertEqual(frame["batch_id"].tolist(), ["proposal_old_batch"])

    def test_cleanup_validated_batch_artifacts_dry_run_then_thin_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            old_dir = _write_batch(base_dir, "proposal_old_batch", timestamp="2026-04-14T00:00:00+00:00")
            _write_batch(base_dir, "proposal_new_batch", timestamp="2026-04-15T00:00:00+00:00")
            before = (old_dir / "summary.json").stat().st_size
            backfill_batch_metadata_from_artifacts(str(base_dir))

            dry_run = cleanup_validated_batch_artifacts(
                str(base_dir),
                keep_recent=1,
                min_summary_bytes=1,
                dry_run=True,
                mode="thin",
            )
            self.assertEqual(dry_run["eligible_count"], 1)
            self.assertEqual(dry_run["changed_count"], 0)
            self.assertGreater(dry_run["estimated_recoverable_bytes"], 0)
            self.assertEqual((old_dir / "summary.json").stat().st_size, before)

            cleanup = cleanup_validated_batch_artifacts(
                str(base_dir),
                keep_recent=1,
                min_summary_bytes=1,
                dry_run=False,
                mode="thin",
            )
            self.assertEqual(cleanup["changed_count"], 1)
            after = (old_dir / "summary.json").stat().st_size
            self.assertLess(after, before)
            thinned = json.loads((old_dir / "summary.json").read_text())
            self.assertEqual(thinned["batch_id"], "proposal_old_batch")
            self.assertEqual(thinned["total_sampled"], 2)
            self.assertTrue(thinned["artifact_compaction"]["sqlite_validated"])

            validation = validate_hot_batch_metadata(str(base_dir), "proposal_old_batch")
            self.assertEqual(validation["valid_count"], 1)

    def test_unbackfilled_batch_is_not_cleanup_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            _write_batch(base_dir, "proposal_unbackfilled_batch", timestamp="2026-04-14T00:00:00+00:00")

            dry_run = cleanup_validated_batch_artifacts(
                str(base_dir),
                keep_recent=0,
                min_summary_bytes=1,
                dry_run=True,
                mode="thin",
            )
            self.assertEqual(dry_run["eligible_count"], 0)
            self.assertEqual(dry_run["skipped_invalid_count"], 1)

            with sqlite3.connect(base_dir / "hot_index.sqlite3") as conn:
                try:
                    row_count = conn.execute("SELECT COUNT(*) FROM batch_summaries").fetchone()[0]
                except sqlite3.OperationalError:
                    row_count = 0
                self.assertEqual(row_count, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiment_hot_index import (
    backfill_proposal_metadata_from_json,
    cleanup_validated_proposal_json_exports,
    load_hot_proposal,
    load_hot_proposal_candidates,
    load_hot_proposals,
    proposal_storage_stats,
    validate_hot_proposal_write,
)
from experiment_store import save_proposal_result


def _proposal_payload(proposal_id: str, timestamp_utc: str = "2026-04-14T00:00:00+00:00") -> dict:
    return {
        "request": {
            "proposal_id": proposal_id,
            "timestamp_utc": timestamp_utc,
            "source_batch_ids": ["batch_a"],
            "strategy_families": ["momentum", "superstock"],
            "objective_name": "wf_v1_score",
            "baseline_name": "momentum_champion_s10005",
            "seed": 42,
            "exploration_fraction": 0.65,
            "exploitation_fraction": 0.35,
            "max_experiments": 2,
            "source_idea_ids": ["idea_a"],
        },
        "status": "generated",
        "candidate_configs": {
            "momentum": [{"LOOKBACK_WEEKS": 26, "TOP_N": 10}],
            "superstock": [{"max_positions": 5, "trend_filter": True}],
        },
        "candidate_metadata": {
            "momentum": [
                {
                    "source_type": "idea_seed",
                    "template_id": "momentum_balanced_default",
                    "strategy_type": "classical",
                    "proposal_role": "explore",
                    "exploration_mode": "idea_seed",
                    "novelty_score": 0.72,
                    "selection_score": 1.4,
                    "is_new_idea": True,
                    "source_idea_ids": ["idea_a"],
                }
            ],
            "superstock": [
                {
                    "source_type": "template_expansion",
                    "template_id": "superstock_high_liquidity",
                    "strategy_type": "classical",
                    "proposal_role": "explore",
                    "exploration_mode": "structural_exploration",
                    "novelty_score": 0.81,
                    "selection_score": 1.2,
                    "is_uncommon_idea": True,
                    "source_idea_ids": [],
                }
            ],
        },
        "reasoning_summary": {
            "proposal_id": proposal_id,
            "proposal_quality": {
                "status": "ok",
                "candidate_count": 2,
                "requested": 2,
                "quality_score": 1.0,
            },
            "cycle_mode": "normal_exploration",
        },
    }


class ProposalHotMetadataTests(unittest.TestCase):
    def test_save_proposal_dual_writes_json_and_queryable_sqlite_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            paths = save_proposal_result(_proposal_payload("proposal_test"), base_dir=str(base_dir))

            self.assertTrue(Path(paths["proposal_path"]).exists())
            self.assertTrue(Path(paths["summary_path"]).exists())
            self.assertTrue(Path(paths["candidate_configs_path"]).exists())

            proposal = load_hot_proposal(str(base_dir), "proposal_test")
            self.assertIsNotNone(proposal)
            self.assertEqual(proposal["proposal_id"], "proposal_test")
            self.assertEqual(proposal["candidate_count"], 2)
            self.assertEqual(proposal["strategy_families"], ["momentum", "superstock"])
            self.assertEqual(proposal["proposal_quality"]["status"], "ok")

            candidates = load_hot_proposal_candidates(str(base_dir), "proposal_test")
            self.assertEqual(len(candidates), 2)
            self.assertEqual(set(candidates["family"].tolist()), {"momentum", "superstock"})
            momentum = candidates[candidates["family"] == "momentum"].iloc[0].to_dict()
            self.assertEqual(momentum["config"]["LOOKBACK_WEEKS"], 26)
            self.assertEqual(momentum["metadata"]["template_id"], "momentum_balanced_default")
            self.assertTrue(momentum["is_new_idea"])

            validation = validate_hot_proposal_write(str(base_dir), "proposal_test")
            self.assertTrue(validation["valid"])
            self.assertEqual(validation["candidate_rows"], 2)

            recent = load_hot_proposals(str(base_dir), families=["superstock"])
            self.assertEqual(recent["proposal_id"].tolist(), ["proposal_test"])

            with sqlite3.connect(base_dir / "hot_index.sqlite3") as conn:
                proposal_rows = conn.execute("SELECT COUNT(*) FROM proposal_summaries").fetchone()[0]
                candidate_rows = conn.execute("SELECT COUNT(*) FROM proposal_candidates").fetchone()[0]
            self.assertEqual(proposal_rows, 1)
            self.assertEqual(candidate_rows, 2)

    def test_storage_stats_show_filesystem_entries_replaceable_by_sqlite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            save_proposal_result(_proposal_payload("proposal_test"), base_dir=str(base_dir))

            stats = proposal_storage_stats(str(base_dir))
            self.assertEqual(stats["proposal_dir_count"], 1)
            self.assertEqual(stats["export_file_count"], 3)
            self.assertEqual(stats["sqlite_proposal_rows"], 1)
            self.assertEqual(stats["sqlite_candidate_rows"], 2)
            self.assertEqual(stats["filesystem_entries_replaceable_after_validation"], 4)

    def test_backfill_imports_existing_json_exports_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            legacy_dir = base_dir / "proposals" / "proposal_legacy"
            legacy_dir.mkdir(parents=True)
            payload = _proposal_payload("proposal_legacy")
            (legacy_dir / "proposal.json").write_text(json.dumps(payload))
            (legacy_dir / "summary.json").write_text(json.dumps(payload["reasoning_summary"]))
            (legacy_dir / "candidate_configs.json").write_text(json.dumps(payload["candidate_configs"]))

            result = backfill_proposal_metadata_from_json(str(base_dir))
            self.assertEqual(result["written_count"], 1)
            self.assertTrue(validate_hot_proposal_write(str(base_dir), "proposal_legacy")["valid"])
            candidates = load_hot_proposal_candidates(str(base_dir), "proposal_legacy")
            self.assertEqual(len(candidates), 2)

    def test_cleanup_policy_only_prunes_validated_old_json_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp) / "experiments"
            save_proposal_result(_proposal_payload("proposal_old_1", "2026-04-14T00:00:00+00:00"), base_dir=str(base_dir))
            save_proposal_result(_proposal_payload("proposal_old_2", "2026-04-14T00:01:00+00:00"), base_dir=str(base_dir))
            save_proposal_result(_proposal_payload("proposal_new", "2026-04-14T00:02:00+00:00"), base_dir=str(base_dir))

            dry_run = cleanup_validated_proposal_json_exports(str(base_dir), keep_recent=1, dry_run=True)
            self.assertEqual(dry_run["eligible_count"], 2)
            self.assertEqual(dry_run["deleted_count"], 0)
            self.assertTrue((base_dir / "proposals" / "proposal_old_1").exists())

            deleted = cleanup_validated_proposal_json_exports(str(base_dir), keep_recent=1, dry_run=False)
            self.assertEqual(deleted["deleted_count"], 2)
            self.assertFalse((base_dir / "proposals" / "proposal_old_1").exists())
            self.assertFalse((base_dir / "proposals" / "proposal_old_2").exists())
            self.assertTrue((base_dir / "proposals" / "proposal_new").exists())

            self.assertTrue(validate_hot_proposal_write(str(base_dir), "proposal_old_1")["valid"])
            old_record = load_hot_proposal(str(base_dir), "proposal_old_1")
            self.assertEqual(old_record["json_export_status"], "deleted")


if __name__ == "__main__":
    unittest.main()

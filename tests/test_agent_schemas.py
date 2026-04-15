import tempfile
import unittest
from pathlib import Path

from agents.schemas import (
    AnalysisReport,
    IdeaRecord,
    ProposalRecord,
    ensure_helper_dirs,
    load_json_records,
    load_latest_pending_proposal_record,
    save_analysis_report,
    save_idea_record,
    save_proposal_record,
    update_proposal_record_status,
)


class AgentSchemaTests(unittest.TestCase):
    def test_helper_dirs_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = ensure_helper_dirs(tmp)
            self.assertTrue(paths["ideas_dir"].exists())
            self.assertTrue(paths["proposals_dir"].exists())
            self.assertTrue(paths["reports_dir"].exists())

    def test_records_round_trip_to_expected_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            idea = IdeaRecord(
                idea_id="idea1",
                family="momentum",
                strategy_type="classical",
                hypothesis="Test idea",
                source="history_mining",
                priority=0.8,
                estimated_cost="medium_cpu",
                timestamp_utc="2026-04-04T00:00:00+00:00",
            )
            proposal = ProposalRecord(
                proposal_id="proposal1",
                strategy_families=["momentum"],
                source_idea_ids=["idea1"],
                candidate_specs=[{"family": "momentum"}],
                exploration_fraction=0.6,
                exploitation_fraction=0.4,
                family_budget={"momentum": 4},
                timestamp_utc="2026-04-04T00:00:00+00:00",
            )
            report = AnalysisReport(
                report_id="report1",
                batch_ids=["batch1"],
                summary={"ok": True},
                next_focus=[{"family": "momentum"}],
                timestamp_utc="2026-04-04T00:00:00+00:00",
            )

            idea_path = save_idea_record(idea, workspace_root=tmp)
            proposal_path = save_proposal_record(proposal, workspace_root=tmp)
            report_path = save_analysis_report(report, workspace_root=tmp)

            self.assertTrue(idea_path.exists())
            self.assertTrue(proposal_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(len(load_json_records(Path(tmp) / "queues" / "ideas")), 1)
            self.assertEqual(len(load_json_records(Path(tmp) / "queues" / "proposals")), 1)
            self.assertEqual(len(load_json_records(Path(tmp) / "reports")), 1)
            pending = load_latest_pending_proposal_record(tmp)
            self.assertEqual(pending["proposal_id"], "proposal1")
            update_proposal_record_status("proposal1", status="consumed", workspace_root=tmp)
            self.assertIsNone(load_latest_pending_proposal_record(tmp))


if __name__ == "__main__":
    unittest.main()

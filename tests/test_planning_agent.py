import inspect
import json
import tempfile
import unittest
from pathlib import Path

from agents import planning_agent
from agents.schemas import AnalysisReport, IdeaRecord, save_analysis_report, save_idea_record
from experiment_store import init_store, save_experiment_result


def _save_result(tmp: str, experiment_id: str, family: str, objective_score: float, viable: bool = True) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"LOOKBACK_WEEKS": 26},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "single",
                "config_hash": f"{experiment_id}_hash",
                "experiment_id": experiment_id,
                "timestamp_utc": "2026-04-04T00:00:00+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2014-01-01",
                "data_end": "2026-04-03",
                "split": "walk-forward",
            },
            "status": "success",
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score / 2.0,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
            },
            "robustness": {"negative_windows": 0, "viable": viable},
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=tmp,
    )


class PlanningAgentTests(unittest.TestCase):
    def test_planning_agent_writes_only_proposal_queue_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)
            save_idea_record(
                IdeaRecord(
                    idea_id="idea1",
                    family="momentum",
                    strategy_type="classical",
                    hypothesis="Use helper idea to bias momentum search.",
                    source="history_mining",
                    priority=0.9,
                    estimated_cost="medium_cpu",
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                ),
                workspace_root=tmp,
            )
            save_analysis_report(
                AnalysisReport(
                    report_id="report1",
                    batch_ids=["batch1"],
                    summary={"ok": True},
                    next_focus=[{"family": "momentum", "focus": "refine", "reason": "leader is viable"}],
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                ),
                workspace_root=tmp,
            )
            before = (experiments_dir / "index.csv").read_text()

            rc = planning_agent.main(
                [
                    "--workspace-root",
                    tmp,
                    "--experiments-dir",
                    str(experiments_dir),
                    "--family",
                    "momentum",
                    "--n",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((experiments_dir / "index.csv").read_text(), before)
            proposal_files = list((Path(tmp) / "queues" / "proposals").glob("*.json"))
            self.assertEqual(len(proposal_files), 1)
            payload = json.loads(proposal_files[0].read_text())
            self.assertIn("source_idea_ids", payload)
            self.assertIn("candidate_specs", payload)
            self.assertIn("planning_rationale", payload)
            self.assertIn("quality_report", payload)
            self.assertIn("proposal_quality", payload["planning_rationale"]["reasoning_summary"])
            self.assertEqual(len(list((Path(tmp) / "reports").glob("*.json"))), 1)

    def test_planning_agent_does_not_reference_execution_authority(self):
        source = inspect.getsource(planning_agent)
        self.assertNotIn("run_batch_experiments", source)
        self.assertNotIn("run_single_experiment", source)
        self.assertNotIn("save_experiment_result_atomic", source)


if __name__ == "__main__":
    unittest.main()

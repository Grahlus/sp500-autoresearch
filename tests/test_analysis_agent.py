import inspect
import json
import tempfile
import unittest
from pathlib import Path

from agents import analysis_agent
from experiment_store import init_store, save_experiment_result


def _save_result(
    tmp: str,
    experiment_id: str,
    family: str,
    objective_score: float,
    viable: bool = True,
    *,
    status: str = "success",
    timestamp_utc: str = "2026-04-04T00:00:00+00:00",
    beats_baseline_objective: bool | None = None,
    dead_zone_risk: float | None = None,
    duplicate_risk: str | None = None,
    source_proposal_id: str | None = None,
    source_idea_ids: list[str] | None = None,
) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"X": objective_score},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "batch1",
                "config_hash": f"{experiment_id}_hash",
                "experiment_id": experiment_id,
                "timestamp_utc": timestamp_utc,
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2014-01-01",
                "data_end": "2026-04-03",
                "split": "walk-forward",
                "duplicate_risk": duplicate_risk,
                "dead_zone_risk": dead_zone_risk,
                "source_proposal_id": source_proposal_id,
                "source_idea_ids": source_idea_ids,
            },
            "status": status,
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score / 2.0,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
            },
            "robustness": {"negative_windows": 0, "viable": viable},
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "beats_baseline_objective": beats_baseline_objective,
                "beats_baseline_guardrails": beats_baseline_objective,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=tmp,
    )


class AnalysisAgentTests(unittest.TestCase):
    def test_analysis_agent_writes_report_without_mutating_official_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)
            batch_dir = experiments_dir / "batches" / "batch1"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "summary.json").write_text(json.dumps({"total_executed": 1}))
            (batch_dir / "leaderboard.csv").write_text("experiment_id,status\nm1,success\n")
            before = (experiments_dir / "index.csv").read_text()

            rc = analysis_agent.main(
                [
                    "--workspace-root",
                    tmp,
                    "--experiments-dir",
                    str(experiments_dir),
                    "--overall",
                    "5",
                    "--per-family",
                    "3",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((experiments_dir / "index.csv").read_text(), before)
            report_files = list((Path(tmp) / "reports").glob("analysis_*.json"))
            self.assertEqual(len(report_files), 1)
            payload = json.loads(report_files[0].read_text())
            self.assertIn("summary", payload)
            self.assertIn("next_focus", payload)
            latest_score_summary = Path(tmp) / "reports" / "score_summaries" / "latest.json"
            self.assertTrue(latest_score_summary.exists())
            score_payload = json.loads(latest_score_summary.read_text())
            self.assertEqual(score_payload["report_id"], payload["report_id"])
            self.assertIn("score_summary", score_payload)

    def test_analysis_report_includes_long_term_memory_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                "m1",
                "momentum",
                1.0,
                viable=True,
                timestamp_utc="2026-04-04T00:00:00+00:00",
                beats_baseline_objective=False,
            )
            _save_result(
                str(experiments_dir),
                "m2",
                "momentum",
                1.5,
                viable=True,
                timestamp_utc="2026-04-05T00:00:00+00:00",
                beats_baseline_objective=True,
                source_proposal_id="proposal1",
                source_idea_ids=["idea1"],
            )
            _save_result(
                str(experiments_dir),
                "s1",
                "superstock",
                -0.5,
                viable=False,
                timestamp_utc="2026-04-05T00:00:00+00:00",
                dead_zone_risk=0.9,
            )
            before = (experiments_dir / "index.csv").read_text()

            report = analysis_agent.build_analysis_report(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                overall_limit=5,
                per_family_limit=3,
            )

            self.assertEqual((experiments_dir / "index.csv").read_text(), before)
            self.assertEqual(report.summary["best_viable_result"]["experiment_id"], "m2")
            self.assertEqual(report.summary["best_baseline_beating_result"]["experiment_id"], "m2")
            self.assertIn("momentum", report.summary["top_per_family"])
            self.assertIn("superstock", report.summary["family_trends"])
            self.assertEqual(report.summary["lineage"]["source_proposal_ids"], ["proposal1"])
            self.assertEqual(report.summary["lineage"]["source_idea_ids"], ["idea1"])
            self.assertIn("momentum", report.summary["family_scorecards"])
            self.assertIn("superstock", report.summary["score_summary"])
            self.assertIn("duplicate_saturation", report.metrics_summary)
            self.assertIn("dead_zone_density", report.metrics_summary)
            focus_by_family = {item["family"]: item for item in report.next_focus}
            self.assertIn("momentum", focus_by_family)
            self.assertIn("recommended_exploration_fraction", focus_by_family["momentum"])
            self.assertIn("stagnation_experiments", focus_by_family["momentum"])
            self.assertIn("reason", focus_by_family["momentum"])

    def test_score_summary_payload_is_planner_friendly(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.2, viable=True, beats_baseline_objective=True)

            report = analysis_agent.build_analysis_report(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                overall_limit=5,
                per_family_limit=3,
            )
            paths = analysis_agent.save_analysis_score_summary(report, workspace_root=tmp)

            latest = Path(paths["latest_score_summary_path"])
            timestamped = Path(paths["score_summary_path"])
            self.assertTrue(latest.exists())
            self.assertTrue(timestamped.exists())
            payload = json.loads(latest.read_text())
            self.assertEqual(payload["report_id"], report.report_id)
            self.assertIn("momentum", payload["score_summary"])
            self.assertIn("next_focus", payload)
            self.assertIn("best_viable_result", payload)
            self.assertEqual(payload["best_viable_result"]["experiment_id"], "m1")

    def test_analysis_agent_does_not_reference_execution_authority(self):
        source = inspect.getsource(analysis_agent)
        self.assertNotIn("run_batch_experiments", source)
        self.assertNotIn("run_single_experiment", source)
        self.assertNotIn("save_experiment_result_atomic", source)


if __name__ == "__main__":
    unittest.main()

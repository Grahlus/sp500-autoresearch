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
            proposals_before = (experiments_dir / "proposals" / "index.csv").read_text()

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
            self.assertEqual((experiments_dir / "proposals" / "index.csv").read_text(), proposals_before)
            self.assertFalse((experiments_dir / "memory.json").exists())
            self.assertFalse((experiments_dir / "scorecards").exists())
            self.assertEqual([path.name for path in (experiments_dir / "proposals").iterdir()], ["index.csv"])
            proposal_files = list((Path(tmp) / "queues" / "proposals").glob("*.json"))
            self.assertEqual(len(proposal_files), 1)
            payload = json.loads(proposal_files[0].read_text())
            self.assertIn("source_idea_ids", payload)
            self.assertIn("idea1", payload["source_idea_ids"])
            self.assertIn("candidate_specs", payload)
            self.assertIn("planning_rationale", payload)
            self.assertEqual(payload["analysis_provenance"]["report_id"], "report1")
            self.assertIn("quality_report", payload)
            self.assertIn("proposal_quality", payload["planning_rationale"]["reasoning_summary"])
            self.assertIn("family_budget_rationale", payload["planning_rationale"])
            self.assertIn("cycle_mode", payload["planning_rationale"])
            self.assertEqual(len(list((Path(tmp) / "reports").glob("*.json"))), 1)

    def test_planning_agent_uses_scorecards_to_weight_family_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            reports_dir = Path(tmp) / "reports" / "score_summaries"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "report_id": "analysis_score",
                        "batch_ids": ["batch1"],
                        "score_summary": {
                            "momentum": {
                                "total_experiments": 50,
                                "viable_rate": 0.35,
                                "search_priority": 0.85,
                                "confidence": 0.8,
                                "dead_zone_density": 0.05,
                                "duplicate_saturation": 0.10,
                                "stagnation_experiments": 2,
                            },
                            "superstock": {
                                "total_experiments": 50,
                                "viable_rate": 0.0,
                                "search_priority": 0.10,
                                "confidence": 0.8,
                                "dead_zone_density": 0.80,
                                "duplicate_saturation": 0.20,
                                "stagnation_experiments": 10,
                            },
                            "rl_bandit": {
                                "total_experiments": 20,
                                "viable_rate": 0.0,
                                "search_priority": 0.05,
                                "confidence": 0.5,
                                "dead_zone_density": 0.90,
                                "duplicate_saturation": 0.20,
                                "stagnation_experiments": 10,
                            },
                        },
                        "next_focus": [
                            {"family": "momentum", "focus": "refine", "reason": "strong viable evidence"},
                            {"family": "superstock", "focus": "deprioritize", "reason": "dead-zone heavy"},
                            {"family": "rl_bandit", "focus": "deprioritize", "reason": "weak evidence"},
                        ],
                        "best_viable_result": {"experiment_id": "m_best", "strategy_family": "momentum"},
                        "best_baseline_beating_result": {"experiment_id": "m_base", "strategy_family": "momentum"},
                    }
                )
            )

            record = planning_agent.build_planning_proposal(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum", "superstock", "rl_bandit"],
                max_experiments=12,
                seed=7,
            )

            self.assertGreater(record.family_budget.get("momentum", 0), record.family_budget.get("superstock", 0))
            self.assertGreater(record.family_budget.get("momentum", 0), record.family_budget.get("rl_bandit", 0))
            self.assertGreaterEqual(record.family_budget.get("superstock", 0), 1)
            rationale = record.planning_rationale
            self.assertEqual(rationale["analysis_report_ids_used"]["score_summary_report_id"], "analysis_score")
            self.assertEqual(rationale["best_viable_result"]["experiment_id"], "m_best")
            self.assertEqual(rationale["best_baseline_beating_result"]["experiment_id"], "m_base")
            self.assertEqual(rationale["family_budget_rationale"]["mode"], "analysis_scorecard_weighted")
            self.assertIn("momentum", rationale["family_budget_rationale"]["families"])

    def test_stagnation_signal_widens_exploration(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            reports_dir = Path(tmp) / "reports" / "score_summaries"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "report_id": "analysis_stagnation",
                        "score_summary": {
                            "momentum": {
                                "total_experiments": 100,
                                "viable_rate": 0.20,
                                "search_priority": 0.70,
                                "confidence": 1.0,
                                "dead_zone_density": 0.10,
                                "duplicate_saturation": 0.20,
                                "stagnation_experiments": 75,
                            }
                        },
                        "next_focus": [{"family": "momentum", "focus": "recover", "reason": "stagnated"}],
                    }
                )
            )

            record = planning_agent.build_planning_proposal(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum"],
                max_experiments=8,
                seed=7,
                exploration_fraction=0.55,
                exploitation_fraction=0.45,
            )

            self.assertEqual(record.planning_rationale["cycle_mode"], "stagnation_escape")
            self.assertGreaterEqual(record.exploration_fraction, 0.80)
            self.assertEqual(record.exploitation_fraction, 0.20)
            self.assertEqual(record.planning_rationale["stagnation_signals"]["momentum"], 75)

    def test_planning_agent_falls_back_when_analysis_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))

            record = planning_agent.build_planning_proposal(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum", "superstock"],
                max_experiments=4,
                seed=7,
                use_analysis_guidance=True,
            )

            self.assertEqual(record.planning_rationale["family_budget_rationale"]["mode"], "fallback_internal")
            self.assertFalse(record.planning_rationale["analysis_guidance_used"])
            self.assertTrue(record.candidate_specs)

    def test_planning_agent_does_not_reference_execution_authority(self):
        source = inspect.getsource(planning_agent)
        self.assertNotIn("run_batch_experiments", source)
        self.assertNotIn("run_single_experiment", source)
        self.assertNotIn("save_experiment_result_atomic", source)


if __name__ == "__main__":
    unittest.main()

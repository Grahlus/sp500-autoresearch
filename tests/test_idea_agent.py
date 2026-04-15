import inspect
import json
import tempfile
import unittest
from pathlib import Path

from agents import idea_agent
from experiment_store import init_store, save_experiment_result


def _save_result(tmp: str, experiment_id: str, family: str, objective_score: float, viable: bool = True) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"X": objective_score},
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


class IdeaAgentTests(unittest.TestCase):
    def test_idea_agent_writes_only_queue_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)
            before = (experiments_dir / "index.csv").read_text()

            rc = idea_agent.main(
                [
                    "--workspace-root",
                    tmp,
                    "--experiments-dir",
                    str(experiments_dir),
                    "--family",
                    "momentum",
                    "--limit",
                    "1",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((experiments_dir / "index.csv").read_text(), before)
            idea_files = list((Path(tmp) / "queues" / "ideas").glob("*.json"))
            self.assertEqual(len(idea_files), 1)
            self.assertEqual(len(list((Path(tmp) / "reports").glob("*.json"))), 0)
            payload = json.loads(idea_files[0].read_text())
            self.assertIn("novelty_score", payload)
            self.assertIn("estimated_runtime_cost", payload)
            self.assertIn("rationale", payload)
            self.assertEqual(payload["strategy_type"], "classical")

    def test_idea_agent_uses_analysis_memory_and_generates_diverse_ideas(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)
            _save_result(str(experiments_dir), "ml1", "ml_ranker", -0.5, viable=False)
            score_summary_dir = Path(tmp) / "reports" / "score_summaries"
            score_summary_dir.mkdir(parents=True, exist_ok=True)
            (score_summary_dir / "latest.json").write_text(
                json.dumps(
                    {
                        "report_id": "analysis1",
                        "score_summary": {
                            "momentum": {
                                "search_priority": 0.8,
                                "viable_rate": 1.0,
                                "dead_zone_density": 0.0,
                                "duplicate_saturation": 0.1,
                            },
                            "ml_ranker": {
                                "search_priority": 0.25,
                                "viable_rate": 0.0,
                                "dead_zone_density": 0.4,
                                "duplicate_saturation": 0.2,
                            },
                            "rl_bandit": {
                                "search_priority": 0.15,
                                "viable_rate": 0.0,
                                "dead_zone_density": 0.7,
                                "duplicate_saturation": 0.2,
                            },
                        },
                        "next_focus": [
                            {"family": "momentum", "focus": "refine"},
                            {"family": "ml_ranker", "focus": "explore"},
                            {"family": "rl_bandit", "focus": "deprioritize"},
                        ],
                        "family_trends": {},
                    }
                )
            )
            before = (experiments_dir / "index.csv").read_text()

            records = idea_agent.generate_idea_records(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum", "ml_ranker", "rl_bandit"],
                limit=20,
            )

            self.assertEqual((experiments_dir / "index.csv").read_text(), before)
            strategy_types = {record.strategy_type for record in records}
            self.assertIn("classical", strategy_types)
            self.assertIn("ml", strategy_types)
            self.assertIn("rl", strategy_types)
            self.assertTrue(all(record.estimated_runtime_cost for record in records))
            self.assertTrue(all(record.rationale for record in records))
            self.assertTrue(all(record.novelty_score is not None for record in records))
            template_records = [record for record in records if record.suggested_template_id]
            self.assertTrue(template_records)
            self.assertTrue(all("template_config" not in (record.suggested_config or {}) for record in template_records))
            analysis_backed = [record for record in records if (record.metadata or {}).get("score_summary_report_id") == "analysis1"]
            self.assertTrue(analysis_backed)
            rl_records = [record for record in records if record.strategy_type == "rl"]
            self.assertTrue(rl_records)
            self.assertLessEqual(max(record.priority for record in rl_records), max(record.priority for record in records))
            self.assertTrue(any(record.is_new_idea for record in records))
            self.assertTrue(any(record.idea_source for record in records))
            self.assertTrue(any(record.idea_kind for record in records))
            self.assertTrue(any(record.novelty_reason for record in records))

    def test_idea_agent_reads_persisted_family_scorecards(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            scorecard_dir = experiments_dir / "scorecards"
            scorecard_dir.mkdir(parents=True, exist_ok=True)
            (scorecard_dir / "family_scorecards.json").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-12T00:00:00+00:00",
                        "families": {
                            "momentum": {
                                "search_priority": 0.9,
                                "viable_rate": 0.4,
                                "dead_zone_density": 0.1,
                                "duplicate_saturation": 0.2,
                            }
                        },
                    }
                )
            )

            records = idea_agent.generate_idea_records(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum"],
                limit=3,
            )

            self.assertTrue(records)
            self.assertTrue(any((record.metadata or {}).get("scorecard", {}).get("search_priority") == 0.9 for record in records))
            self.assertTrue(any((record.metadata or {}).get("consumable_by_proposal_flow") for record in records))

    def test_idea_agent_uses_idea_yield_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            scorecard_dir = experiments_dir / "scorecards"
            scorecard_dir.mkdir(parents=True, exist_ok=True)
            (scorecard_dir / "idea_yield.json").write_text(
                json.dumps(
                    {
                        "timestamp_utc": "2026-04-12T00:00:00+00:00",
                        "families": {
                            "momentum": {
                                "idea_state": "promising",
                                "idea_quality_score": 0.92,
                                "idea_decay_score": 0.12,
                                "idea_fresh_share": 0.20,
                                "idea_promising_share": 0.80,
                                "idea_retired_share": 0.0,
                                "search_priority": 0.90,
                                "top": [
                                    {"idea_key": "momentum::idea_kind::template_variation", "idea_kind": "template_variation"}
                                ],
                            }
                        },
                    }
                )
            )
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            records = idea_agent.generate_idea_records(
                workspace_root=tmp,
                experiments_dir=str(experiments_dir),
                families=["momentum"],
                limit=4,
            )

            self.assertTrue(records)
            self.assertTrue(any((record.metadata or {}).get("idea_yield_state") == "promising" for record in records))
            self.assertTrue(any((record.metadata or {}).get("idea_yield_report_id") == "2026-04-12T00:00:00+00:00" for record in records))
            self.assertGreater(max(record.priority for record in records), 0.6)

    def test_idea_agent_does_not_reference_execution_authority(self):
        source = inspect.getsource(idea_agent)
        self.assertNotIn("run_batch_experiments", source)
        self.assertNotIn("run_single_experiment", source)
        self.assertNotIn("save_experiment_result_atomic", source)


if __name__ == "__main__":
    unittest.main()

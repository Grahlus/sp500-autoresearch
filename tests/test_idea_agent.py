import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import idea_agent
from agents.schemas import save_idea_record
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
    def test_web_research_session_limit_records_backoff_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            with patch.object(idea_agent, "_search_web_research_results", side_effect=RuntimeError("search failed")):
                ideas = idea_agent._web_research_ideas(tmp, families=["momentum"], max_ideas=2)

            self.assertEqual(ideas, [])
            state_path = Path(tmp) / "queues" / "web_research" / "web_research_state.json"
            status_path = Path(tmp) / "queues" / "web_research" / "web_research_status.json"
            self.assertTrue(state_path.exists())
            self.assertTrue(status_path.exists())
            state = json.loads(state_path.read_text())
            status = json.loads(status_path.read_text())
            self.assertFalse(state["session_limit_hit"])
            self.assertEqual(state["backoff_state"], "error_backoff")
            self.assertIn("next_retry_at", state)
            self.assertFalse(status["session_limit_hit"])
            self.assertEqual(status["backoff_state"], "error_backoff")
            self.assertEqual(status["backoff_reason"], "error_backoff")

    def test_web_research_retries_after_backoff_expires(self):
        from datetime import UTC, datetime, timedelta

        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            state_dir = Path(tmp) / "queues" / "web_research"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_dir.joinpath("web_research_state.json").write_text(
                json.dumps(
                    {
                        "last_attempt_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                        "last_failure_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                        "last_run_ts": (datetime.now(UTC) - timedelta(hours=2)).timestamp(),
                        "next_retry_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
                        "backoff_state": "session_limit",
                        "backoff_reason": "session_limit",
                        "session_limit_hit": True,
                        "used_topic_slugs": [],
                    }
                )
            )

            papers = [
                {
                    "title": "Recent momentum paper",
                    "source": "SSRN",
                    "hypothesis": "Rank recent winners with a volatility gate.",
                    "rationale": "Recent practitioner note.",
                    "techniques": ["volatility gate"],
                    "positive_result": True,
                    "lookback_days": 126,
                    "top_n_pct": 0.15,
                }
            ]
            with patch.object(idea_agent, "_search_web_research_results", return_value=[{"title": "Recent momentum paper", "source": "SSRN", "snippet": "summary"}]), \
                 patch.object(idea_agent, "_call_minimax_web_synthesis", return_value=papers) as call:
                ideas = idea_agent._web_research_ideas(tmp, families=["momentum"], max_ideas=1)

            self.assertEqual(call.call_count, 1)
            self.assertEqual(len(ideas), 1)
            status = json.loads((state_dir / "web_research_status.json").read_text())
            self.assertEqual(status["backoff_state"], "cooldown")
            self.assertFalse(status["session_limit_hit"])
            self.assertIsNotNone(status["next_retry_at"])
            self.assertTrue((Path(tmp) / "queues" / "ideas").exists())
            self.assertEqual(len(list((Path(tmp) / "queues" / "ideas").glob("*.json"))), 1)

    def test_web_research_success_becomes_queued_idea_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            papers = [
                {
                    "title": "Cross-sectional momentum with risk control",
                    "source": "Working Paper",
                    "hypothesis": "Use a volatility-adjusted ranking score.",
                    "rationale": "Strong recent evidence for downside-aware ranking.",
                    "techniques": ["volatility gate", "risk control"],
                    "positive_result": True,
                    "lookback_days": 252,
                    "skip_days": 5,
                    "hold_days": 20,
                    "top_n_pct": 0.1,
                }
            ]
            with patch.object(idea_agent, "_search_web_research_results", return_value=[{"title": "Cross-sectional momentum with risk control", "source": "Working Paper", "snippet": "summary"}]), \
                 patch.object(idea_agent, "_call_minimax_web_synthesis", return_value=papers):
                ideas = idea_agent._web_research_ideas(tmp, families=["momentum"], max_ideas=1)

            self.assertEqual(len(ideas), 1)
            idea_dir = Path(tmp) / "queues" / "ideas"
            files = list(idea_dir.glob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text())
            self.assertTrue(payload["web_search_used"])
            self.assertEqual(payload["idea_source"], "web_search")
            self.assertEqual(payload["paper_title"], "Cross-sectional momentum with risk control")
            self.assertEqual(payload["source_idea_ids"], [])
            self.assertEqual((payload.get("metadata") or {}).get("paper_title"), "Cross-sectional momentum with risk control")
            self.assertEqual((payload.get("metadata") or {}).get("web_search_used"), True)
            self.assertEqual(payload["idea_provider"], "minimax")
            self.assertEqual(payload["idea_model"], "MiniMax-M2.7")
            self.assertTrue(payload["is_structurally_novel"])
            self.assertTrue(payload["is_out_of_box"])
            self.assertTrue(payload["is_uncommon_idea"])

    def test_web_research_unavailable_does_not_stop_idea_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
                records = idea_agent.generate_idea_records(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    families=["momentum"],
                    limit=4,
                )

            self.assertTrue(records)

    def test_idea_agent_writes_only_queue_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)
            before = (experiments_dir / "index.csv").read_text()

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
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

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
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
            self.assertTrue(any(record.is_structurally_novel for record in records))
            self.assertTrue(any(record.is_out_of_box for record in records))
            self.assertLessEqual(sum(1 for record in records if record.source == "history_mining"), 1)
            self.assertLessEqual(sum(1 for record in records if record.source in {"template_expansion", "cross_family_hybrid"}), 2)

    def test_idea_agent_rotates_recent_queued_ideas(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
                first_records = idea_agent.generate_idea_records(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    families=["momentum"],
                    limit=4,
                )
                for record in first_records:
                    save_idea_record(record, workspace_root=tmp)
                second_records = idea_agent.generate_idea_records(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    families=["momentum"],
                    limit=4,
                )

            first_signatures = [idea_agent._candidate_signature(record) for record in first_records]
            second_signatures = [idea_agent._candidate_signature(record) for record in second_records]
            self.assertNotEqual(first_signatures, second_signatures)
            self.assertNotEqual(
                [record.template_id for record in first_records],
                [record.template_id for record in second_records],
            )

    def test_fresh_web_idea_ranks_above_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), "m1", "momentum", 1.0, viable=True)

            web_record = idea_agent.IdeaRecord(
                idea_id="idea_web_1",
                family="momentum",
                strategy_type="classical",
                hypothesis="Fresh MiniMax idea",
                source="structural_extension",
                priority=0.90,
                estimated_cost="medium_cpu",
                timestamp_utc="2026-04-19T00:00:00+00:00",
                idea_source="web_search",
                paper_title="Fresh MiniMax paper",
                web_search_used=True,
                idea_provider="minimax",
                idea_model="MiniMax-M2.7",
                idea_kind="new_portfolio_exposure_control",
                structural_distance=0.90,
                template_similarity_class="portfolio_overlay",
                uncommon_idea_reason="Fresh structural hypothesis from web synthesis.",
                is_structurally_novel=True,
                is_out_of_box=True,
                is_uncommon_idea=True,
            )

            with patch.object(idea_agent, "_web_research_ideas", return_value=[web_record]):
                records = idea_agent.generate_idea_records(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    families=["momentum"],
                    limit=4,
                )

            self.assertTrue(records)
            self.assertEqual(records[0].idea_source, "web_search")
            self.assertTrue(records[0].web_search_used)
            self.assertEqual(records[0].idea_model, "MiniMax-M2.7")
            self.assertTrue(records[0].is_out_of_box)
            self.assertTrue(records[0].is_structurally_novel)

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

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
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

            with patch.object(idea_agent, "_web_research_ideas", return_value=[]):
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

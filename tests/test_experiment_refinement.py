import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from agents.schemas import AnalysisReport, IdeaRecord, save_analysis_report, save_idea_record
from experiment_batch import proposal_to_batch_request, run_batch_experiments
from experiment_refinement import (
    analyze_experiment_history,
    build_local_neighbors,
    build_proposal_request,
    generate_next_round_proposal,
    minimum_viable_candidate_count,
    negotiate_family_budgets,
    sample_exploration_configs,
    score_proposal_quality,
)
from experiment_store import compute_config_hash, save_experiment_result
from experiment_types import ExperimentResult, ExperimentSpec


def _save_fake_result(
    base_dir: str,
    *,
    family: str,
    config: dict,
    experiment_id: str,
    objective_score: float,
    status: str = "success",
    viable: bool = True,
    baseline_status: str | None = None,
    beats_baseline_objective: bool | None = None,
):
    config_hash = compute_config_hash(family, config)
    result = ExperimentResult(
        spec=ExperimentSpec(
            family=family,
            params=config,
            search_method="single",
            objective_name="wf_v1_score",
            batch_id="batch_hist",
            config_hash=config_hash,
            experiment_id=experiment_id,
            timestamp_utc="2026-04-03T00:00:00+00:00",
            benchmark_source="spy_symbol",
            dataset_id="d1",
            data_start="2020-01-01",
            data_end="2020-12-31",
            split="walk-forward",
            git_commit="deadbeef",
            family_version="test",
        ),
        status=status,
        objective_score=objective_score,
        metrics={
            "sharpe": objective_score,
            "calmar": objective_score,
            "total_return": objective_score * 10.0,
            "trades_per_year": 10.0,
            "max_drawdown": -5.0,
            "sharpe_min": objective_score - 0.2,
        },
        robustness={"viable": viable, "negative_windows": 0 if viable else 5},
        artifacts={},
        baseline_comparison={
            "baseline_name": "momentum_champion_s10005" if family == "momentum" else None,
            "comparison_status": baseline_status,
            "beats_baseline_objective": beats_baseline_objective,
        }
        if family == "momentum"
        else None,
        runtime_seconds=0.1,
    )
    payload = asdict(result)
    payload["spec"] = asdict(result.spec)
    save_experiment_result(payload, base_dir=base_dir)
    return config_hash


class ExperimentRefinementTests(unittest.TestCase):
    def test_proposal_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.0,
                baseline_status="exact_verified_current_engine",
                beats_baseline_objective=True,
            )
            request = build_proposal_request(strategy_families=["momentum"], seed=7, max_experiments=4)
            a = generate_next_round_proposal(request, base_dir=tmp)
            b = generate_next_round_proposal(request, base_dir=tmp)
        self.assertEqual(a.candidate_configs, b.candidate_configs)

    def test_proposal_never_emits_invalid_configs_and_avoids_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.0,
            )
            request = build_proposal_request(strategy_families=["momentum"], seed=7, max_experiments=4)
            proposal = generate_next_round_proposal(request, base_dir=tmp)
            hashes = {compute_config_hash("momentum", config) for config in proposal.candidate_configs["momentum"]}
        self.assertNotIn(seen, hashes)

    def test_local_neighbors_stay_near_seed_config(self):
        seed_config = {
            "LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None,
        }
        neighbors = build_local_neighbors("momentum", seed_config, limit=3, seed=7)
        self.assertTrue(neighbors)
        for neighbor in neighbors:
            diff = sum(1 for key in seed_config if seed_config[key] != neighbor[key])
            self.assertLessEqual(diff, 2)

    def test_exploration_sampler_covers_unseen_region(self):
        explored = {
            compute_config_hash("superstock", {
                "max_positions": 5, "price_min": 5.0, "price_max": 15.0, "min_dollar_volume": 1_000_000.0,
                "above_52w_low_mult": 1.25, "near_52w_high_mult": 0.75, "rs_rank_26w_min": 0.7, "rs_rank_52w_min": 0.7,
                "base_depth_max": 0.6, "weekly_range_median_max": 0.18, "weekly_volatility_max": 0.12,
                "volume_dryup_max": 0.8, "vix_hard_cap": 35.0, "vix_ma_multiplier": 1.25,
                "breakout_extension_max": 0.1, "daily_volume_expansion_mult": 1.5,
                "daily_dollar_volume_expansion_mult": 1.5, "parabolic_from_pivot_min": 0.25,
                "parabolic_above_10w_min": 0.2, "late_stage_range_mult": 1.75, "late_stage_volume_mult": 2.0,
            })
        }
        configs = sample_exploration_configs("superstock", limit=2, seed=7, explored_hashes=explored)
        self.assertEqual(len(configs), 2)
        for config in configs:
            self.assertNotIn(compute_config_hash("superstock", config), explored)

    def test_dead_zones_do_not_hard_block_neighbor_or_exploration_sampling(self):
        seed_config = {
            "LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None,
        }
        dead_zones = {
            "LOOKBACK_WEEKS": {20, 26, 39},
            "SKIP_WEEKS": {2, 3, 4},
        }
        neighbors = build_local_neighbors("momentum", seed_config, limit=2, seed=7, dead_zones=dead_zones)
        exploration = sample_exploration_configs(
            "momentum",
            limit=2,
            seed=7,
            explored_hashes=set(),
            dead_zones=dead_zones,
        )
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(len(exploration), 2)

    def test_proposal_can_be_passed_directly_to_batch_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.0,
            )
            request = build_proposal_request(strategy_families=["momentum"], seed=7, max_experiments=2)
            proposal = generate_next_round_proposal(request, base_dir=tmp)
            batch_request = proposal_to_batch_request(proposal)
            fake_result = ExperimentResult(
                spec=ExperimentSpec(family="momentum", params={}, config_hash="a", experiment_id="a"),
                status="success",
                objective_score=1.0,
                metrics={"sharpe": 1.0},
                robustness={"viable": True},
                artifacts={},
            )
            with patch("experiment_batch.load_data", return_value={}), patch(
                "experiment_batch.run_single_experiment", return_value=fake_result
            ):
                batch_result = run_batch_experiments(batch_request, base_dir=tmp)
        self.assertGreaterEqual(batch_result.total_executed, 1)

    def test_analysis_outputs_dead_zones_and_baseline_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 20, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m_bad1",
                objective_score=-0.5,
                status="no_trades",
                viable=False,
                baseline_status="exact_verified_current_engine",
                beats_baseline_objective=False,
            )
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 20, "SKIP_WEEKS": 2, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m_bad2",
                objective_score=-0.4,
                status="invalid",
                viable=False,
                baseline_status="exact_verified_current_engine",
                beats_baseline_objective=False,
            )
            analysis = analyze_experiment_history(families=["momentum"], base_dir=tmp)
        self.assertIn("momentum", analysis["families"])
        self.assertIn("LOOKBACK_WEEKS", analysis["families"]["momentum"]["dead_zones"])

    def test_proposal_includes_metadata_and_broader_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.0,
                baseline_status="exact_verified_current_engine",
                beats_baseline_objective=True,
            )
            _save_fake_result(
                tmp,
                family="superstock",
                config={"max_positions": 5, "price_min": 5.0, "price_max": 15.0, "min_dollar_volume": 1_000_000.0,
                        "above_52w_low_mult": 1.25, "near_52w_high_mult": 0.75, "rs_rank_26w_min": 0.7, "rs_rank_52w_min": 0.7,
                        "base_depth_max": 0.6, "weekly_range_median_max": 0.18, "weekly_volatility_max": 0.12,
                        "volume_dryup_max": 0.8, "vix_hard_cap": 35.0, "vix_ma_multiplier": 1.25,
                        "breakout_extension_max": 0.1, "daily_volume_expansion_mult": 1.5,
                        "daily_dollar_volume_expansion_mult": 1.5, "parabolic_from_pivot_min": 0.25,
                        "parabolic_above_10w_min": 0.2, "late_stage_range_mult": 1.75, "late_stage_volume_mult": 2.0},
                experiment_id="s1",
                objective_score=0.4,
            )
            request = build_proposal_request(strategy_families=["momentum", "superstock"], seed=11, max_experiments=6)
            proposal = generate_next_round_proposal(request, base_dir=tmp)

        self.assertIn("momentum", proposal.candidate_metadata)
        self.assertIn("superstock", proposal.candidate_metadata)
        momentum_sources = {meta["source_type"] for meta in proposal.candidate_metadata["momentum"]}
        superstock_sources = {meta["source_type"] for meta in proposal.candidate_metadata["superstock"]}
        self.assertTrue({"template_expansion", "cross_family_hybrid"} & momentum_sources)
        self.assertTrue({"template_expansion", "cross_family_hybrid"} & superstock_sources)
        self.assertGreater(len(proposal.candidate_configs["momentum"]), len(proposal.candidate_configs["superstock"]))
        for family_metadata in proposal.candidate_metadata.values():
            for metadata in family_metadata:
                self.assertIn("strategy_type", metadata)
                self.assertIn("proposal_role", metadata)
                self.assertIn("region_label", metadata)
                self.assertIn("duplicate_risk", metadata)
                self.assertIn("dead_zone_risk", metadata)
        for family in proposal.candidate_configs:
            hashes = {compute_config_hash(family, config) for config in proposal.candidate_configs[family]}
            self.assertEqual(len(hashes), len(proposal.candidate_configs[family]))

    def test_queued_idea_records_influence_proposal_and_persist_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_config = {
                "LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None,
            }
            parent_hash = _save_fake_result(
                tmp,
                family="momentum",
                config=base_config,
                experiment_id="m1",
                objective_score=1.0,
                baseline_status="exact_verified_current_engine",
                beats_baseline_objective=True,
            )
            save_idea_record(
                IdeaRecord(
                    idea_id="idea_momentum_seed",
                    family="momentum",
                    strategy_type="classical",
                    hypothesis="Seed momentum proposal from helper idea",
                    source="history_mining",
                    priority=1.0,
                    estimated_cost="medium_cpu",
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                    rationale="helper idea should influence proposal generation",
                    suggested_config={"parent_config_hash": parent_hash},
                ),
                workspace_root=tmp,
            )
            request = build_proposal_request(strategy_families=["momentum"], seed=7, max_experiments=4)
            proposal = generate_next_round_proposal(request, base_dir=tmp)

        self.assertIn("idea_momentum_seed", proposal.request.source_idea_ids)
        source_idea_ids = [
            tuple(metadata.get("source_idea_ids") or [])
            for metadata in proposal.candidate_metadata["momentum"]
        ]
        self.assertIn(("idea_momentum_seed",), source_idea_ids)
        batch_request = proposal_to_batch_request(proposal)
        propagated = [
            tuple(spec.source_idea_ids or [])
            for spec in batch_request.precomputed_specs["momentum"]
        ]
        self.assertIn(("idea_momentum_seed",), propagated)

    def test_analysis_guidance_can_bias_family_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.0,
                viable=True,
            )
            _save_fake_result(
                tmp,
                family="superstock",
                config={"max_positions": 5, "price_min": 5.0, "price_max": 15.0, "min_dollar_volume": 1_000_000.0,
                        "above_52w_low_mult": 1.25, "near_52w_high_mult": 0.75, "rs_rank_26w_min": 0.7, "rs_rank_52w_min": 0.7,
                        "base_depth_max": 0.6, "weekly_range_median_max": 0.18, "weekly_volatility_max": 0.12,
                        "volume_dryup_max": 0.8, "vix_hard_cap": 35.0, "vix_ma_multiplier": 1.25,
                        "breakout_extension_max": 0.1, "daily_volume_expansion_mult": 1.5,
                        "daily_dollar_volume_expansion_mult": 1.5, "parabolic_from_pivot_min": 0.25,
                        "parabolic_above_10w_min": 0.2, "late_stage_range_mult": 1.75, "late_stage_volume_mult": 2.0},
                experiment_id="s1",
                objective_score=0.2,
                viable=False,
            )
            save_analysis_report(
                AnalysisReport(
                    report_id="analysis1",
                    batch_ids=["batch1"],
                    summary={},
                    next_focus=[
                        {"family": "momentum", "focus": "refine"},
                        {"family": "superstock", "focus": "explore"},
                    ],
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                ),
                workspace_root=tmp,
            )
            request = build_proposal_request(strategy_families=["momentum", "superstock"], seed=7, max_experiments=8)
            proposal = generate_next_round_proposal(request, base_dir=tmp)

        family_budget = proposal.reasoning_summary["family_budget_decision"]
        self.assertGreaterEqual(family_budget["momentum"], family_budget["superstock"])
        self.assertTrue(proposal.reasoning_summary["analysis_guidance"])

    def test_family_budget_negotiation_is_recorded_and_controls_weak_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_fake_result(
                tmp,
                family="momentum",
                config={"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                        "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None},
                experiment_id="m1",
                objective_score=1.5,
                viable=True,
            )
            for idx in range(5):
                _save_fake_result(
                    tmp,
                    family="rl_bandit",
                    config={
                        "policy_type": "ucb" if idx % 2 else "epsilon_greedy",
                        "lookback_days": 126 if idx % 2 else 252,
                        "rebalance_days": 5 if idx < 3 else 10,
                        "epsilon": [0.05, 0.10, 0.20, 0.05, 0.10][idx],
                        "ucb_bonus": [0.5, 1.0, 2.0, 0.5, 1.0][idx],
                        "max_positions": [3, 5, 8, 3, 5][idx],
                        "momentum_top_pct": 0.025, "superstock_top_pct": 0.025,
                        "use_vix_gate": True, "use_fear_greed_gate": True,
                    },
                    experiment_id=f"rl{idx}",
                    objective_score=-1.0,
                    viable=False,
                )
            request = build_proposal_request(
                strategy_families=["momentum", "superstock", "ml_ranker", "rl_bandit"],
                seed=7,
                max_experiments=24,
            )
            analysis = analyze_experiment_history(families=request.strategy_families, base_dir=tmp)
            budgets, report = negotiate_family_budgets(request, analysis)

        self.assertEqual(sum(budgets.values()), 24)
        self.assertGreater(budgets["momentum"], budgets["rl_bandit"])
        self.assertEqual(report["mode"], "evidence_weighted")

    def test_proposal_quality_detects_underfilled_large_search(self):
        request = build_proposal_request(
            strategy_families=["momentum", "superstock"],
            seed=7,
            max_experiments=96,
            min_viable_fill_rate=0.50,
            min_large_search_candidates=48,
        )
        analysis = {
            "families": {
                "momentum": {"explored_hashes": set(), "dead_zones": {}, "poor_region_signatures": set()},
                "superstock": {"explored_hashes": set(), "dead_zones": {}, "poor_region_signatures": set()},
            }
        }
        quality = score_proposal_quality(
            request,
            budgets={"momentum": 72, "superstock": 24},
            candidate_configs={"momentum": [], "superstock": []},
            candidate_metadata={"momentum": [], "superstock": []},
            analysis=analysis,
        )

        self.assertEqual(minimum_viable_candidate_count(request), 48)
        self.assertEqual(quality["status"], "fail")
        self.assertFalse(quality["execution_allowed"])
        self.assertIn("zero_selected_for_family", quality["shortfall_reasons"])

    def test_proposal_quality_passes_meaningful_large_search(self):
        request = build_proposal_request(strategy_families=["momentum"], seed=7, max_experiments=96)
        configs = [{"LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4, "TOP_PCT": 0.025, "MA_WEEKS": 20,
                    "STOP_TYPE": "adaptive", "STOP_LOSS_PCT": 0.2, "STOP_PARABOLIC": 0.3, "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97, "RANK_EXIT_CONFIRM": None}
                   for _ in range(48)]
        metadata = [{"source_type": "broad_exploration", "strategy_type": "classical", "proposal_role": "explore"} for _ in configs]
        analysis = {"families": {"momentum": {"explored_hashes": set(), "dead_zones": {}, "poor_region_signatures": set()}}}
        quality = score_proposal_quality(
            request,
            budgets={"momentum": 96},
            candidate_configs={"momentum": configs},
            candidate_metadata={"momentum": metadata},
            analysis=analysis,
        )
        self.assertIn(quality["status"], {"pass", "warn"})
        self.assertTrue(quality["execution_allowed"])


if __name__ == "__main__":
    unittest.main()

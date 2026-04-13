import tempfile
import unittest
from pathlib import Path

from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_runtime_decision import build_runtime_decision
from experiment_store import init_store, save_experiment_result
from experiment_types import RuntimeDecisionInput


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    config_hash: str,
    objective_score: float,
    viable: bool,
    parent_config_hash: str | None = None,
    status: str = "success",
) -> None:
    config = {
        "LOOKBACK_WEEKS": 26,
        "SKIP_WEEKS": 3,
        "REBAL_WEEKS": 4,
        "TOP_PCT": 0.025,
        "MA_WEEKS": 20,
        "STOP_TYPE": "adaptive",
        "STOP_LOSS_PCT": 0.2,
        "STOP_PARABOLIC": 0.3,
        "INV_VOL_DAYS": 15,
        "MIN_HOLD_DAYS": 5,
        "FG_MIN": 10.0,
        "EXIT_PCT_RANK": 0.97,
        "RANK_EXIT_CONFIRM": None,
    }
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": config,
                "search_method": "proposal",
                "objective_name": "wf_v1_score",
                "batch_id": "branch_budget_test",
                "config_hash": config_hash,
                "experiment_id": experiment_id,
                "timestamp_utc": "2026-04-13T00:00:00+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-13",
                "split": "walk-forward",
                "parent_config_hash": parent_config_hash,
            },
            "status": status,
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score / 2.0,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
                "exposure": 0.5 if viable else 0.05,
            },
            "robustness": {"negative_windows": 0 if viable else 3, "viable": viable},
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "baseline_verified": True,
                "baseline_metric_source": "verified_current_engine",
                "comparison_kind": "partial",
                "delta_sharpe": objective_score - 1.0,
                "delta_calmar": objective_score - 1.0,
                "delta_return": objective_score * 10.0,
                "beats_baseline_objective": objective_score >= 1.0,
                "beats_baseline_guardrails": objective_score >= 1.0,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class BranchBudgetTests(unittest.TestCase):
    def test_runtime_decision_allocates_more_to_strengthening_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="good_root",
                family="momentum",
                config_hash="good_root",
                objective_score=1.0,
                viable=True,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="good_child",
                family="momentum",
                config_hash="good_child",
                objective_score=1.5,
                viable=True,
                parent_config_hash="good_root",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_root",
                family="momentum",
                config_hash="weak_root",
                objective_score=0.2,
                viable=False,
                status="invalid",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_child",
                family="momentum",
                config_hash="weak_child",
                objective_score=-0.4,
                viable=False,
                status="invalid",
                parent_config_hash="weak_root",
            )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=12,
                )
            )

        self.assertIn("momentum", decision.branch_budgets)
        branch_budgets = decision.branch_budgets["momentum"]
        self.assertGreaterEqual(len(branch_budgets), 2)
        good_branch = next(item for item in branch_budgets if item["branch_root_config_hash"] == "good_root")
        weak_branch = next(item for item in branch_budgets if item["branch_root_config_hash"] == "weak_root")
        self.assertGreater(good_branch["branch_budget"], weak_branch["branch_budget"])
        self.assertIn("branch_budget_rationale", decision.rationale)
        self.assertIn("branch_budget_rationale", decision.used_signals)

    def test_proposal_generation_threads_branch_budget_into_candidate_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="good_root",
                family="momentum",
                config_hash="good_root",
                objective_score=1.0,
                viable=True,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="good_child",
                family="momentum",
                config_hash="good_child",
                objective_score=1.4,
                viable=True,
                parent_config_hash="good_root",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_root",
                family="momentum",
                config_hash="weak_root",
                objective_score=0.2,
                viable=False,
                status="invalid",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_child",
                family="momentum",
                config_hash="weak_child",
                objective_score=-0.4,
                viable=False,
                status="invalid",
                parent_config_hash="weak_root",
            )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=12,
                )
            )
            proposal = generate_next_round_proposal(
                build_proposal_request(
                    strategy_families=["momentum"],
                    max_experiments=12,
                    per_family_budgets=decision.family_budgets,
                    branch_budgets=decision.branch_budgets,
                    branch_budget_rationale=decision.branch_budget_rationale,
                    confirmation_state="unconfirmed",
                    confirmation_required=False,
                    holdout_check_required=False,
                ),
                base_dir=str(experiments_dir),
            )

        self.assertIn("branch_budgets", proposal.reasoning_summary)
        self.assertIn("branch_budget_rationale", proposal.reasoning_summary)
        self.assertTrue(proposal.reasoning_summary["families"]["momentum"]["branch_budgets"])
        self.assertIn("branch_budget_rationale", proposal.reasoning_summary["families"]["momentum"])
        self.assertEqual(proposal.request.branch_budgets, decision.branch_budgets)
        self.assertTrue(any(item.get("branch_budget_stance") for item in proposal.candidate_metadata["momentum"]))
        self.assertTrue(any(item.get("lineage_root_config_hash") for item in proposal.candidate_metadata["momentum"]))


if __name__ == "__main__":
    unittest.main()

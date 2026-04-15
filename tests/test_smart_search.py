"""
tests/test_smart_search.py — Smart search system tests.

Covers:
 1. Family-specific step sets applied correctly (momentum, superstock, ml_ranker, rl_bandit)
 2. Family-specific conditional parameters respected
 3. No family falls back to brute-force enumeration
 4. ML/RL search is architecture-aware
 5. Superstock search uses meaningful screening/breakout-style steps
 6. Exact-duplicate protection works
 7. Explored-hash exclusion works
 8. select_search_method returns correct methods per cycle mode and family
 9. smart_sample_candidates metadata is complete and correct
 10. run.py / evaluate.py entry points remain importable (no regression)
"""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from experiment_search import (
    _CONDITIONAL_PARAMS,
    _SEARCH_PROFILES,
    _STEP_SETS,
    coarse_grid_candidates,
    get_search_profile,
    get_step_sets,
    latin_hypercube_candidates,
    local_refinement_candidates,
    select_search_method,
    smart_sample_candidates,
)
from experiment_spaces import normalize_experiment_config
from experiment_store import compute_config_hash


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_momentum() -> dict:
    return normalize_experiment_config("momentum", {
        "LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4,
        "TOP_PCT": 0.025, "MA_WEEKS": 20, "STOP_TYPE": "adaptive",
        "STOP_LOSS_PCT": 0.20, "STOP_PARABOLIC": 0.30, "INV_VOL_DAYS": 15,
        "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97,
        "RANK_EXIT_CONFIRM": None, "SIGNAL_TYPE": "momentum",
        "REVERSAL_FILTER_DAYS": 0, "DVOL_QUANTILE": 0.70, "CRASH_PROTECTION": 0,
    })


def _default_superstock() -> dict:
    return normalize_experiment_config("superstock", {
        "max_positions": 5, "price_min": 5.0, "price_max": 15.0,
        "min_dollar_volume": 1_000_000.0, "above_52w_low_mult": 1.25,
        "near_52w_high_mult": 0.75, "rs_rank_26w_min": 0.70,
        "rs_rank_52w_min": 0.70, "base_depth_max": 0.60,
        "weekly_range_median_max": 0.18, "weekly_volatility_max": 0.12,
        "volume_dryup_max": 0.80, "vix_hard_cap": 35.0, "vix_ma_multiplier": 1.25,
        "breakout_extension_max": 0.10, "daily_volume_expansion_mult": 1.50,
        "daily_dollar_volume_expansion_mult": 1.50,
        "parabolic_from_pivot_min": 0.25, "parabolic_above_10w_min": 0.20,
        "late_stage_range_mult": 1.75, "late_stage_volume_mult": 2.00,
    })


def _default_ml_ranker() -> dict:
    return normalize_experiment_config("ml_ranker", {
        "model_type": "ridge", "lookback_days": 252, "horizon_days": 10,
        "rebalance_days": 5, "top_pct": 0.03, "max_positions": 10,
        "feature_set": "trend_volume", "allow_short": False,
        "use_vix_gate": True, "use_fear_greed_gate": False,
    })


def _default_rl_bandit() -> dict:
    return normalize_experiment_config("rl_bandit", {
        "policy_type": "ucb", "lookback_days": 252, "rebalance_days": 5,
        "epsilon": 0.10, "ucb_bonus": 1.0, "max_positions": 5,
        "momentum_top_pct": 0.03, "superstock_top_pct": 0.03,
        "use_vix_gate": True, "use_fear_greed_gate": True,
    })


# ── Step-set tests ────────────────────────────────────────────────────────────

class TestStepSets(unittest.TestCase):

    def _check_family(self, family: str) -> None:
        sets = get_step_sets(family)
        self.assertIn("coarse", sets, f"{family}: missing 'coarse'")
        self.assertIn("fine", sets, f"{family}: missing 'fine'")
        # Every coarse value must also appear in fine
        for param, coarse_vals in sets["coarse"].items():
            fine_vals = sets["fine"].get(param, [])
            for v in coarse_vals:
                self.assertIn(v, fine_vals,
                    f"{family}/{param}: coarse value {v!r} not in fine {fine_vals}")

    def test_momentum_step_sets(self):
        self._check_family("momentum")

    def test_superstock_step_sets(self):
        self._check_family("superstock")

    def test_ml_ranker_step_sets(self):
        self._check_family("ml_ranker")

    def test_rl_bandit_step_sets(self):
        self._check_family("rl_bandit")

    def test_coarse_not_larger_than_fine(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            sets = get_step_sets(family)
            for param in sets["coarse"]:
                if param in sets["fine"]:
                    self.assertLessEqual(
                        len(sets["coarse"][param]),
                        len(sets["fine"][param]),
                        f"{family}/{param}: coarse has more choices than fine",
                    )

    def test_unknown_family_fallback(self):
        sets = get_step_sets("nonexistent_family")
        self.assertIn("coarse", sets)
        self.assertIn("fine", sets)

    def test_all_registered_families_have_step_sets(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            self.assertIn(family, _STEP_SETS, f"{family} missing from _STEP_SETS")


# ── Search-profile tests ──────────────────────────────────────────────────────

class TestSearchProfiles(unittest.TestCase):

    def test_all_families_have_profiles(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            p = get_search_profile(family)
            self.assertIn("architecture_params", p)
            self.assertIn("early_history_threshold", p)

    def test_superstock_prefers_lhc(self):
        p = get_search_profile("superstock")
        self.assertTrue(p["prefer_lhc"])

    def test_momentum_does_not_prefer_lhc(self):
        p = get_search_profile("momentum")
        self.assertFalse(p["prefer_lhc"])

    def test_ml_ranker_has_architecture_params(self):
        p = get_search_profile("ml_ranker")
        self.assertIn("model_type", p["architecture_params"])
        self.assertIn("feature_set", p["architecture_params"])

    def test_rl_bandit_has_policy_type_as_arch_param(self):
        p = get_search_profile("rl_bandit")
        self.assertIn("policy_type", p["architecture_params"])

    def test_superstock_lower_early_threshold(self):
        # Superstock has a smaller space so it should explore richer methods sooner
        ss = get_search_profile("superstock")
        mom = get_search_profile("momentum")
        self.assertLess(ss["early_history_threshold"], mom["early_history_threshold"])


# ── Method selection tests ────────────────────────────────────────────────────

class TestSelectSearchMethod(unittest.TestCase):

    def test_confirmation_always_local_refinement(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            m, p = select_search_method("confirmation", 999, 0, family=family)
            self.assertEqual(m, "local_refinement", f"{family}: expected local_refinement")
            self.assertEqual(p, "fine")

    def test_holdout_always_local_refinement(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            m, _ = select_search_method("holdout", 999, 0, family=family)
            self.assertEqual(m, "local_refinement")

    def test_stagnation_always_lhc(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            profile = get_search_profile(family)
            m, p = select_search_method("normal_exploration", 999,
                                        profile["stagnation_threshold"], family=family)
            self.assertEqual(m, "latin_hypercube")
            self.assertEqual(p, "coarse")

    def test_large_search_always_lhc(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            m, _ = select_search_method("large-search", 999, 0, family=family)
            self.assertEqual(m, "latin_hypercube")

    def test_superstock_early_uses_lhc_not_coarse_grid(self):
        # superstock prefer_lhc=True → LHC even with low history
        m, _ = select_search_method("normal_exploration", 5, 0, family="superstock")
        self.assertEqual(m, "latin_hypercube")

    def test_momentum_early_uses_coarse_grid(self):
        m, _ = select_search_method("normal_exploration", 5, 0, family="momentum")
        self.assertEqual(m, "coarse_grid")

    def test_momentum_normal_exploration_hybrid(self):
        m, p = select_search_method("normal_exploration", 200, 0, family="momentum")
        self.assertEqual(m, "hybrid")
        self.assertEqual(p, "fine")

    def test_no_family_arg_falls_back_to_defaults(self):
        # Should not raise — uses _DEFAULT_PROFILE
        m, p = select_search_method("normal_exploration", 200, 0)
        self.assertIn(m, {"coarse_grid", "latin_hypercube", "hybrid", "local_refinement"})


# ── Conditional parameter tests ───────────────────────────────────────────────

class TestConditionalParams(unittest.TestCase):

    def test_momentum_stop_loss_only_when_stop_type_active(self):
        candidates = coarse_grid_candidates("momentum", n=40, seed=42, step_policy="coarse")
        for c in candidates:
            if c.get("STOP_TYPE") == "none":
                self.assertIsNone(c.get("STOP_LOSS_PCT"),
                    "STOP_LOSS_PCT should be None when STOP_TYPE=none")

    def test_momentum_parabolic_only_for_adaptive(self):
        candidates = coarse_grid_candidates("momentum", n=40, seed=42, step_policy="coarse")
        for c in candidates:
            if c.get("STOP_TYPE") in {"fixed", "none"}:
                self.assertIsNone(c.get("STOP_PARABOLIC"),
                    f"STOP_PARABOLIC should be None for STOP_TYPE={c['STOP_TYPE']!r}")

    def test_momentum_rank_exit_confirm_only_when_exit_rank_set(self):
        candidates = coarse_grid_candidates("momentum", n=40, seed=42, step_policy="fine")
        for c in candidates:
            if c.get("EXIT_PCT_RANK") is None:
                self.assertIsNone(c.get("RANK_EXIT_CONFIRM"),
                    "RANK_EXIT_CONFIRM should be None when EXIT_PCT_RANK is None")

    def test_rl_bandit_epsilon_only_for_epsilon_greedy(self):
        candidates = coarse_grid_candidates("rl_bandit", n=40, seed=42, step_policy="coarse")
        for c in candidates:
            if c.get("policy_type") == "ucb":
                # epsilon should be at its default when policy is ucb
                default = 0.10  # spec default
                actual = c.get("epsilon")
                # Allow None or default — the key point is it's not varied
                self.assertIn(actual, {None, default},
                    f"epsilon={actual!r} should not vary for ucb policy")

    def test_rl_bandit_ucb_bonus_only_for_ucb(self):
        candidates = coarse_grid_candidates("rl_bandit", n=40, seed=42, step_policy="coarse")
        for c in candidates:
            if c.get("policy_type") == "epsilon_greedy":
                default = 1.0  # spec default
                actual = c.get("ucb_bonus")
                self.assertIn(actual, {None, default},
                    f"ucb_bonus={actual!r} should not vary for epsilon_greedy policy")

    def test_superstock_has_no_conditional_rules(self):
        # Superstock avoids coupling by using LHC, not conditional rules
        self.assertNotIn("superstock", _CONDITIONAL_PARAMS)


# ── Sampling primitive tests ──────────────────────────────────────────────────

class TestCoarseGridCandidates(unittest.TestCase):

    def _check_no_duplicates(self, family: str) -> None:
        cands = coarse_grid_candidates(family, n=20, seed=42)
        hashes = [compute_config_hash(family, c) for c in cands]
        self.assertEqual(len(hashes), len(set(hashes)), f"{family}: duplicates found")

    def test_no_duplicates_momentum(self):   self._check_no_duplicates("momentum")
    def test_no_duplicates_superstock(self): self._check_no_duplicates("superstock")
    def test_no_duplicates_ml_ranker(self):  self._check_no_duplicates("ml_ranker")
    def test_no_duplicates_rl_bandit(self):  self._check_no_duplicates("rl_bandit")

    def test_respects_explored_hashes(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            first = coarse_grid_candidates(family, n=5, seed=42)
            fh = {compute_config_hash(family, c) for c in first}
            second = coarse_grid_candidates(family, n=5, seed=42, explored_hashes=fh)
            sh = {compute_config_hash(family, c) for c in second}
            self.assertEqual(len(fh & sh), 0, f"{family}: explored hashes not respected")

    def test_uses_only_coarse_choices(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            coarse_choices = _STEP_SETS[family]["coarse"]
            # Skip conditional params — they get set to spec defaults when inactive,
            # which may not coincide with the coarse choice list.
            cond_params = set(_CONDITIONAL_PARAMS.get(family, {}))
            cands = coarse_grid_candidates(family, n=30, seed=99, step_policy="coarse")
            for config in cands:
                for param, choices in coarse_choices.items():
                    if param in cond_params:
                        continue  # conditional params use spec defaults when inactive
                    val = config.get(param)
                    if val is not None:
                        self.assertIn(val, choices,
                            f"{family}/{param}={val!r} not in coarse {choices}")

    def test_does_not_enumerate_full_grid(self):
        start = time.time()
        cands = coarse_grid_candidates("momentum", n=10, seed=0)
        elapsed = time.time() - start
        self.assertEqual(len(cands), 10)
        self.assertLess(elapsed, 5.0)


class TestLatinHypercubeCandidates(unittest.TestCase):

    def test_no_duplicates_all_families(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            cands = latin_hypercube_candidates(family, n=20, seed=42)
            hashes = [compute_config_hash(family, c) for c in cands]
            self.assertEqual(len(hashes), len(set(hashes)), f"{family}: LHC duplicates")

    def test_numeric_coverage_momentum(self):
        cands = latin_hypercube_candidates("momentum", n=20, seed=0, step_policy="fine")
        lookbacks = {c["LOOKBACK_WEEKS"] for c in cands}
        self.assertGreater(len(lookbacks), 1)

    def test_numeric_coverage_superstock(self):
        cands = latin_hypercube_candidates("superstock", n=20, seed=0, step_policy="fine")
        rs_ranks = {c["rs_rank_26w_min"] for c in cands}
        self.assertGreater(len(rs_ranks), 1, "LHC should diversify rs_rank_26w_min")

    def test_respects_explored_hashes(self):
        for family in ["momentum", "superstock"]:
            first = latin_hypercube_candidates(family, n=5, seed=42)
            fh = {compute_config_hash(family, c) for c in first}
            second = latin_hypercube_candidates(family, n=5, seed=42, explored_hashes=fh)
            sh = {compute_config_hash(family, c) for c in second}
            self.assertEqual(len(fh & sh), 0)


class TestLocalRefinementCandidates(unittest.TestCase):

    def _check_refinement(self, family: str, center: dict) -> None:
        cands = local_refinement_candidates(family, center, n=8, seed=42)
        self.assertGreater(len(cands), 0, f"{family}: no refinement candidates")
        center_hash = compute_config_hash(family, center)
        for c in cands:
            self.assertNotEqual(compute_config_hash(family, c), center_hash,
                f"{family}: local_refinement returned the center unchanged")

    def test_local_refinement_momentum(self):
        self._check_refinement("momentum", _default_momentum())

    def test_local_refinement_superstock(self):
        self._check_refinement("superstock", _default_superstock())

    def test_local_refinement_ml_ranker(self):
        self._check_refinement("ml_ranker", _default_ml_ranker())

    def test_local_refinement_rl_bandit(self):
        self._check_refinement("rl_bandit", _default_rl_bandit())

    def test_uses_fine_choices_momentum(self):
        fine = _STEP_SETS["momentum"]["fine"]
        cands = local_refinement_candidates("momentum", _default_momentum(), n=20, seed=42)
        for c in cands:
            for param, choices in fine.items():
                val = c.get(param)
                if val is not None:
                    self.assertIn(val, choices, f"momentum/{param}={val!r} not in fine")

    def test_uses_fine_choices_superstock(self):
        fine = _STEP_SETS["superstock"]["fine"]
        cands = local_refinement_candidates("superstock", _default_superstock(), n=15, seed=42)
        for c in cands:
            for param, choices in fine.items():
                val = c.get(param)
                if val is not None:
                    self.assertIn(val, choices, f"superstock/{param}={val!r} not in fine")


# ── smart_sample_candidates tests ─────────────────────────────────────────────

class TestSmartSampleCandidates(unittest.TestCase):

    def _check_metadata_keys(self, meta: dict, family: str) -> None:
        for key in ["search_method", "step_policy", "coarse_or_fine", "quantized",
                    "capped", "cycle_mode", "family", "architecture_params",
                    "fallback_used", "candidates_generated"]:
            self.assertIn(key, meta, f"{family}: metadata missing '{key}'")
        self.assertEqual(meta["family"], family)

    def test_metadata_complete_momentum(self):
        _, meta = smart_sample_candidates("momentum", n=5, seed=42)
        self._check_metadata_keys(meta, "momentum")

    def test_metadata_complete_superstock(self):
        _, meta = smart_sample_candidates("superstock", n=5, seed=42)
        self._check_metadata_keys(meta, "superstock")

    def test_metadata_complete_ml_ranker(self):
        _, meta = smart_sample_candidates("ml_ranker", n=5, seed=42)
        self._check_metadata_keys(meta, "ml_ranker")

    def test_metadata_complete_rl_bandit(self):
        _, meta = smart_sample_candidates("rl_bandit", n=5, seed=42)
        self._check_metadata_keys(meta, "rl_bandit")

    def test_superstock_early_uses_lhc(self):
        _, meta = smart_sample_candidates("superstock", n=6, seed=42,
            cycle_mode="normal_exploration", history_count=5, stagnation_batches=0)
        self.assertEqual(meta["search_method"], "latin_hypercube")

    def test_rl_bandit_has_policy_type_in_arch_params(self):
        _, meta = smart_sample_candidates("rl_bandit", n=4, seed=42)
        self.assertIn("policy_type", meta["architecture_params"])

    def test_ml_ranker_has_model_type_in_arch_params(self):
        _, meta = smart_sample_candidates("ml_ranker", n=4, seed=42)
        self.assertIn("model_type", meta["architecture_params"])

    def test_no_duplicates_all_families(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            cands, _ = smart_sample_candidates(family, n=12, seed=42)
            hashes = [compute_config_hash(family, c) for c in cands]
            self.assertEqual(len(hashes), len(set(hashes)), f"{family}: duplicates")

    def test_no_cross_batch_duplicates(self):
        for family in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            first, _ = smart_sample_candidates(family, n=8, seed=1)
            fh = {compute_config_hash(family, c) for c in first}
            second, _ = smart_sample_candidates(family, n=8, seed=1, explored_hashes=fh)
            sh = {compute_config_hash(family, c) for c in second}
            self.assertEqual(len(fh & sh), 0, f"{family}: cross-batch duplicates")

    def test_structurally_novel_count_in_metadata(self):
        center = _default_momentum()
        _, meta = smart_sample_candidates("momentum", n=8, seed=42,
            cycle_mode="normal_exploration", history_count=100,
            top_configs=[center])
        self.assertIn("structurally_novel_count", meta)
        self.assertGreaterEqual(meta["structurally_novel_count"], 0)

    def test_confirmation_uses_local_refinement(self):
        _, meta = smart_sample_candidates("momentum", n=6, seed=42,
            cycle_mode="confirmation", history_count=200, top_configs=[_default_momentum()])
        self.assertEqual(meta["search_method"], "local_refinement")

    def test_space_not_exhausted_momentum(self):
        start = time.time()
        cands, _ = smart_sample_candidates("momentum", n=20, seed=42,
            cycle_mode="large-search", history_count=500)
        elapsed = time.time() - start
        self.assertGreater(len(cands), 0)
        self.assertLess(elapsed, 10.0, "Should not brute-force the full space")

    def test_space_not_exhausted_superstock(self):
        start = time.time()
        cands, _ = smart_sample_candidates("superstock", n=20, seed=42,
            cycle_mode="large-search", history_count=500)
        elapsed = time.time() - start
        self.assertGreater(len(cands), 0)
        self.assertLess(elapsed, 10.0)


# ── Architecture-awareness tests ──────────────────────────────────────────────

class TestArchitectureAwareness(unittest.TestCase):

    def test_ml_ranker_covers_model_types(self):
        """ML ranker sampling should produce both model types."""
        cands, _ = smart_sample_candidates("ml_ranker", n=20, seed=42,
            cycle_mode="large-search", history_count=200)
        model_types = {c.get("model_type") for c in cands}
        self.assertGreater(len(model_types), 1, "Should cover multiple model_type values")

    def test_ml_ranker_covers_feature_sets(self):
        cands, _ = smart_sample_candidates("ml_ranker", n=20, seed=42,
            cycle_mode="large-search", history_count=200)
        feature_sets = {c.get("feature_set") for c in cands}
        self.assertGreater(len(feature_sets), 1, "Should cover multiple feature_set values")

    def test_rl_bandit_covers_both_policies(self):
        cands, _ = smart_sample_candidates("rl_bandit", n=20, seed=42,
            cycle_mode="large-search", history_count=200)
        policies = {c.get("policy_type") for c in cands}
        self.assertGreater(len(policies), 1, "Should cover both policy_type values")

    def test_momentum_covers_signal_types(self):
        cands, _ = smart_sample_candidates("momentum", n=20, seed=42,
            cycle_mode="large-search", history_count=200)
        signals = {c.get("SIGNAL_TYPE") for c in cands}
        self.assertGreater(len(signals), 3, "Should cover multiple SIGNAL_TYPE values")

    def test_superstock_uses_lhc_to_avoid_all_extreme_combos(self):
        """LHC should prevent all RS thresholds being at max simultaneously."""
        cands, meta = smart_sample_candidates("superstock", n=20, seed=42,
            cycle_mode="large-search", history_count=200)
        self.assertEqual(meta["search_method"], "latin_hypercube")
        # Verify not all configs have both RS ranks at maximum
        all_extreme = sum(
            1 for c in cands
            if c.get("rs_rank_26w_min") == 0.80 and c.get("rs_rank_52w_min") == 0.80
        )
        self.assertLess(all_extreme, len(cands),
            "LHC should prevent all-extreme RS combinations dominating")


# ── Regression / compatibility tests ─────────────────────────────────────────

class TestRegressionCompatibility(unittest.TestCase):

    def test_experiment_search_importable(self):
        import experiment_search
        self.assertTrue(callable(experiment_search.smart_sample_candidates))

    def test_experiment_spaces_importable(self):
        import experiment_spaces
        self.assertTrue(callable(experiment_spaces.sample_random_candidates))
        self.assertTrue(callable(experiment_spaces.normalize_experiment_config))

    def test_experiment_refinement_importable(self):
        import experiment_refinement
        self.assertTrue(callable(experiment_refinement.sample_exploration_configs))
        self.assertTrue(callable(experiment_refinement.generate_next_round_proposal))

    def test_sample_exploration_configs_unchanged_signature_contract(self):
        from experiment_refinement import sample_exploration_configs
        # Must still return a list of configs (dicts)
        result = sample_exploration_configs(
            "momentum", limit=3, seed=42, explored_hashes=set()
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, dict)

    def test_sample_exploration_configs_superstock(self):
        from experiment_refinement import sample_exploration_configs
        result = sample_exploration_configs(
            "superstock", limit=3, seed=42, explored_hashes=set()
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_sample_exploration_configs_ml_ranker(self):
        from experiment_refinement import sample_exploration_configs
        result = sample_exploration_configs(
            "ml_ranker", limit=3, seed=42, explored_hashes=set()
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_sample_exploration_configs_rl_bandit(self):
        from experiment_refinement import sample_exploration_configs
        result = sample_exploration_configs(
            "rl_bandit", limit=3, seed=42, explored_hashes=set()
        )
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()

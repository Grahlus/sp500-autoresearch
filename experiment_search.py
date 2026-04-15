"""
experiment_search.py — Smart, non-exhaustive search strategies.

Replaces blind random sampling with principled coverage:
  - coarse_grid:        sample from reduced choice sets (broad sweep)
  - latin_hypercube:    even numeric coverage without full enumeration
  - local_refinement:   perturb 1–2 params near known-good configs
  - architecture_sweep: enumerate all architectural variants first
  - hybrid:             mix of coarse + local refinement

Entry point: smart_sample_candidates()
Each call returns (candidates, metadata) where metadata records:
  search_method, step_policy, coarse_or_fine, quantized, capped, cycle_mode,
  family, structurally_novel, architecture_params_used
"""
from __future__ import annotations

import math
import random
from typing import Any

from experiment_spaces import (
    get_family_search_space,
    normalize_experiment_config,
    sample_random_candidates,
)
from experiment_store import compute_config_hash


# ── Step-policy definitions ───────────────────────────────────────────────────
#
# "coarse": fewer choices per param → broad sweep, low enumeration cost
# "fine":   all choices per param   → local neighbourhood, finer resolution
#
# For families without a defined policy falls back to the full choice set.

_STEP_SETS: dict[str, dict[str, dict[str, list[Any]]]] = {

    # ── Momentum ──────────────────────────────────────────────────────────────
    "momentum": {
        "coarse": {
            # Time-window: span extremes only
            "LOOKBACK_WEEKS":       [20, 39],
            "SKIP_WEEKS":           [2, 4],
            "REBAL_WEEKS":          [2, 4],
            # Portfolio construction
            "TOP_PCT":              [0.015, 0.03],
            "MA_WEEKS":             [16, 24],
            "INV_VOL_DAYS":         [10, 30],
            "MIN_HOLD_DAYS":        [5, 15],
            # Stop / exit
            "STOP_TYPE":            ["adaptive", "fixed", "none"],
            "STOP_LOSS_PCT":        [0.15, 0.25],
            "STOP_PARABOLIC":       [0.25, 0.40],
            "EXIT_PCT_RANK":        [None, 0.97],
            "RANK_EXIT_CONFIRM":    [None, 2],
            # Market filters
            "FG_MIN":               [10.0, 30.0],
            "DVOL_QUANTILE":        [0.50, 0.60, 0.80],
            "CRASH_PROTECTION":     [0, 1],
            # Signal: enumerate all types — each is architecturally different
            "SIGNAL_TYPE":          [
                "momentum", "regression_slope", "sharpe_ratio", "52w_high_blend",
                "rsi_divergence", "volume_momentum", "bollinger_rank", "macd_signal",
                "adx_weighted", "cci_trend", "mfi_signal", "obv_momentum",
            ],
            "REVERSAL_FILTER_DAYS": [0, 5],
        },
        "fine": {
            "LOOKBACK_WEEKS":       [20, 26, 39, 52],
            "SKIP_WEEKS":           [2, 3, 4],
            "REBAL_WEEKS":          [2, 4],
            "TOP_PCT":              [0.015, 0.025, 0.03],
            "MA_WEEKS":             [16, 20, 24],
            "INV_VOL_DAYS":         [10, 15, 20, 30],
            "MIN_HOLD_DAYS":        [5, 10, 15],
            "STOP_TYPE":            ["adaptive", "fixed", "none"],
            "STOP_LOSS_PCT":        [0.15, 0.20, 0.25, 0.30],
            "STOP_PARABOLIC":       [0.25, 0.30, 0.35, 0.40],
            "EXIT_PCT_RANK":        [None, 0.94, 0.95, 0.97],
            "RANK_EXIT_CONFIRM":    [None, 1, 2],
            "FG_MIN":               [10.0, 22.0, 30.0],
            "DVOL_QUANTILE":        [0.50, 0.60, 0.70, 0.80],
            "CRASH_PROTECTION":     [0, 1],
            "SIGNAL_TYPE":          [
                "momentum", "regression_slope", "sharpe_ratio", "52w_high_blend",
                "rsi_divergence", "volume_momentum", "bollinger_rank", "macd_signal",
                "adx_weighted", "cci_trend", "mfi_signal", "obv_momentum",
            ],
            "REVERSAL_FILTER_DAYS": [0, 3, 5],
        },
    },

    # ── Superstock ────────────────────────────────────────────────────────────
    # Strategy: screen for tight-base breakouts. Key axes:
    #   RS-rank philosophy (strict vs relaxed), breakout extension tolerance,
    #   vol/tightness tolerance, and late-stage protection.
    # Avoid: brute-force tightening all thresholds together (empty universe).
    # Design: LHC preferred (naturally avoids all-extreme combinations).
    "superstock": {
        "coarse": {
            # Portfolio sizing
            "max_positions":                    [3, 8],
            # Liquidity / price universe
            "price_min":                        [3.0, 5.0],
            "price_max":                        [15.0, 20.0],
            "min_dollar_volume":                [500_000.0, 2_000_000.0],
            # Relative-strength philosophy — coarse spans strict vs relaxed
            "rs_rank_26w_min":                  [0.60, 0.80],
            "rs_rank_52w_min":                  [0.60, 0.80],
            # Base tightness — coarse spans loose vs tight
            "base_depth_max":                   [0.40, 0.80],
            # Volatility / tightness filters — coarse spans extremes
            "weekly_range_median_max":          [0.15, 0.22],
            "weekly_volatility_max":            [0.10, 0.15],
            # Volume filters — sample extremes to discover sensitivity
            "volume_dryup_max":                 [0.80, 1.00],
            "daily_volume_expansion_mult":      [1.25, 2.00],
            "daily_dollar_volume_expansion_mult": [1.25, 2.00],
            # Breakout tolerance
            "breakout_extension_max":           [0.08, 0.15],
            # Market-regime gate — coarse spans tight vs loose
            "vix_hard_cap":                     [30.0, 40.0],
            "vix_ma_multiplier":                [1.10, 1.50],
            # Late-stage / parabolic filters
            "parabolic_from_pivot_min":         [0.20, 0.30],
            "parabolic_above_10w_min":          [0.15, 0.25],
            "late_stage_range_mult":            [1.50, 2.00],
            "late_stage_volume_mult":           [1.50, 2.50],
            # Fixed-choice params (only one option): always use default
            "above_52w_low_mult":               [1.25],
            "near_52w_high_mult":               [0.75],
        },
        "fine": {
            "max_positions":                    [3, 5, 8],
            "price_min":                        [3.0, 5.0],
            "price_max":                        [15.0, 20.0],
            "min_dollar_volume":                [500_000.0, 1_000_000.0, 2_000_000.0],
            "rs_rank_26w_min":                  [0.60, 0.70, 0.80],
            "rs_rank_52w_min":                  [0.60, 0.70, 0.80],
            "base_depth_max":                   [0.40, 0.60, 0.80],
            "weekly_range_median_max":          [0.15, 0.18, 0.22],
            "weekly_volatility_max":            [0.10, 0.12, 0.15],
            "volume_dryup_max":                 [0.80, 0.90, 1.00],
            "daily_volume_expansion_mult":      [1.25, 1.50, 2.00],
            "daily_dollar_volume_expansion_mult": [1.25, 1.50, 2.00],
            "breakout_extension_max":           [0.08, 0.10, 0.15],
            "vix_hard_cap":                     [30.0, 35.0, 40.0],
            "vix_ma_multiplier":                [1.10, 1.25, 1.50],
            "parabolic_from_pivot_min":         [0.20, 0.25, 0.30],
            "parabolic_above_10w_min":          [0.15, 0.20, 0.25],
            "late_stage_range_mult":            [1.50, 1.75, 2.00],
            "late_stage_volume_mult":           [1.50, 2.00, 2.50],
            "above_52w_low_mult":               [1.25],
            "near_52w_high_mult":               [0.75],
        },
    },

    # ── ML Ranker ─────────────────────────────────────────────────────────────
    # Strategy: vary model architecture, feature space, and horizon concept.
    # Avoid: brute-force grid over regularization hyperparams at fixed architecture.
    # Architecture params: model_type, feature_set, allow_short
    # Coarse: span all architectures; use extremes for numeric params.
    "ml_ranker": {
        "coarse": {
            # Model architecture — always enumerate all (structurally different)
            "model_type":           ["ridge", "hgb"],
            # Horizon concept — short vs long (not the middle)
            "lookback_days":        [126, 504],
            "horizon_days":         [5, 20],
            "rebalance_days":       [5, 10],
            # Portfolio
            "top_pct":              [0.025, 0.05],
            "max_positions":        [5, 20],
            # Feature set — always enumerate all (different input spaces)
            "feature_set":          ["trend", "trend_volume", "full"],
            # Structural switches — always enumerate both
            "allow_short":          [False, True],
            "use_vix_gate":         [False, True],
            "use_fear_greed_gate":  [False, True],
        },
        "fine": {
            # All choices for local neighbourhood
            "model_type":           ["ridge", "hgb"],
            "lookback_days":        [126, 252, 504],
            "horizon_days":         [5, 10, 20],
            "rebalance_days":       [5, 10],
            "top_pct":              [0.025, 0.03, 0.05],
            "max_positions":        [5, 10, 20],
            "feature_set":          ["trend", "trend_volume", "full"],
            "allow_short":          [False, True],
            "use_vix_gate":         [False, True],
            "use_fear_greed_gate":  [False, True],
        },
    },

    # ── RL Bandit ─────────────────────────────────────────────────────────────
    # Strategy: vary policy concept and exploration style.
    # Critical conditional: epsilon is only relevant for epsilon_greedy;
    #                       ucb_bonus is only relevant for ucb policy.
    # Avoid: dense grid over both epsilon AND ucb_bonus regardless of policy.
    "rl_bandit": {
        "coarse": {
            # Policy architecture — always enumerate both (fundamentally different)
            "policy_type":          ["ucb", "epsilon_greedy"],
            # Time horizon concept
            "lookback_days":        [126, 252],
            "rebalance_days":       [5, 10],
            # Policy-specific exploration params: coarse uses extremes
            "epsilon":              [0.05, 0.20],
            "ucb_bonus":            [0.5, 2.0],
            # Portfolio
            "max_positions":        [3, 8],
            "momentum_top_pct":     [0.025, 0.05],
            "superstock_top_pct":   [0.025, 0.05],
            # Market gates
            "use_vix_gate":         [False, True],
            "use_fear_greed_gate":  [False, True],
        },
        "fine": {
            "policy_type":          ["ucb", "epsilon_greedy"],
            "lookback_days":        [126, 252],
            "rebalance_days":       [5, 10],
            "epsilon":              [0.05, 0.10, 0.20],
            "ucb_bonus":            [0.5, 1.0, 2.0],
            "max_positions":        [3, 5, 8],
            "momentum_top_pct":     [0.025, 0.03, 0.05],
            "superstock_top_pct":   [0.025, 0.03, 0.05],
            "use_vix_gate":         [False, True],
            "use_fear_greed_gate":  [False, True],
        },
    },
}


# ── Conditional parameter rules ───────────────────────────────────────────────
#
# When a condition is not met the param is set to its null/default value
# before normalization, preventing wasteful non-distinct combos.
#
# Rule formats:
#   "active_when":          {sibling_param: {valid_values,...}}
#   "active_when_not_null": "sibling_param"

_CONDITIONAL_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "momentum": {
        # STOP_LOSS_PCT is irrelevant when STOP_TYPE is "none"
        "STOP_LOSS_PCT":     {"active_when": {"STOP_TYPE": {"adaptive", "fixed"}}},
        # STOP_PARABOLIC is only used for adaptive stops
        "STOP_PARABOLIC":    {"active_when": {"STOP_TYPE": {"adaptive"}}},
        # RANK_EXIT_CONFIRM only applies when EXIT_PCT_RANK threshold is set
        "RANK_EXIT_CONFIRM": {"active_when_not_null": "EXIT_PCT_RANK"},
    },
    "rl_bandit": {
        # epsilon exploration rate only applies to epsilon-greedy policy
        "epsilon":    {"active_when": {"policy_type": {"epsilon_greedy"}}},
        # UCB confidence bonus only applies to UCB policy
        "ucb_bonus":  {"active_when": {"policy_type": {"ucb"}}},
    },
    # ml_ranker and superstock have no strong conditional dependencies
    # (superstock thresholds are independent axes of a screening template)
}


# ── Family search profiles ────────────────────────────────────────────────────
#
# Guides family-specific method selection.
# "early_history_threshold": switch from coarse_grid to richer methods
# "stagnation_threshold":    trigger LHC escape after N stagnant batches
# "architecture_params":     params that define structural novelty
# "prefer_lhc":              True → prefer LHC over coarse_grid even early
#                            (better for families with correlated screen params)
# "default_method":          fallback when no special condition applies
# "default_step_policy":     step resolution for default method

_SEARCH_PROFILES: dict[str, dict[str, Any]] = {
    "momentum": {
        "early_history_threshold": 50,
        "stagnation_threshold":    3,
        "architecture_params":     ["SIGNAL_TYPE"],
        "prefer_lhc":              False,
        "default_method":          "hybrid",
        "default_step_policy":     "fine",
    },
    "superstock": {
        # Superstock space has many coupled screening thresholds — LHC naturally
        # avoids all-tight / all-loose corner combinations.
        "early_history_threshold": 30,
        "stagnation_threshold":    2,
        "architecture_params":     ["max_positions", "rs_rank_26w_min"],
        "prefer_lhc":              True,
        "default_method":          "latin_hypercube",
        "default_step_policy":     "coarse",
    },
    "ml_ranker": {
        # Small space — emphasise architectural variation early.
        "early_history_threshold": 15,
        "stagnation_threshold":    2,
        "architecture_params":     ["model_type", "feature_set", "allow_short"],
        "prefer_lhc":              False,
        "default_method":          "hybrid",
        "default_step_policy":     "fine",
    },
    "rl_bandit": {
        "early_history_threshold": 15,
        "stagnation_threshold":    2,
        "architecture_params":     ["policy_type"],
        "prefer_lhc":              False,
        "default_method":          "hybrid",
        "default_step_policy":     "fine",
    },
}

# Default profile for families not in _SEARCH_PROFILES
_DEFAULT_PROFILE: dict[str, Any] = {
    "early_history_threshold": 50,
    "stagnation_threshold":    3,
    "architecture_params":     [],
    "prefer_lhc":              False,
    "default_method":          "hybrid",
    "default_step_policy":     "fine",
}


# ── Public helpers ────────────────────────────────────────────────────────────

def get_step_sets(family: str) -> dict[str, dict[str, list[Any]]]:
    """Return {coarse: {param: [values]}, fine: {param: [values]}} for a family."""
    family = family.strip().lower()
    if family in _STEP_SETS:
        return _STEP_SETS[family]
    # Fallback: use the full choice list for both coarse and fine
    try:
        space = get_family_search_space(family)
    except ValueError:
        return {"coarse": {}, "fine": {}}
    choices = {name: list(spec["choices"]) for name, spec in space.items()}
    return {"coarse": choices, "fine": choices}


def get_search_profile(family: str) -> dict[str, Any]:
    """Return the search profile for a family (with defaults for unknown families)."""
    return _SEARCH_PROFILES.get(family.strip().lower(), _DEFAULT_PROFILE)


def select_search_method(
    cycle_mode: str,
    history_count: int,
    stagnation_batches: int,
    family: str = "",
) -> tuple[str, str]:
    """
    Decide which search strategy and step resolution to use.

    Family-aware: uses _SEARCH_PROFILES thresholds when family is provided.

    Returns:
        (method, step_policy) where
        method      ∈ {coarse_grid, latin_hypercube, local_refinement, hybrid}
        step_policy ∈ {coarse, fine}
    """
    profile = get_search_profile(family) if family else _DEFAULT_PROFILE

    # Confirmation / holdout: stay close to known-good configs
    if cycle_mode in {"confirmation", "holdout"}:
        return "local_refinement", "fine"

    early_threshold = profile["early_history_threshold"]
    stagnation_threshold = profile["stagnation_threshold"]

    # Very early exploration — coarse sweep to map the landscape
    # (families that prefer LHC use it even at the start)
    if history_count < early_threshold:
        if profile.get("prefer_lhc"):
            return "latin_hypercube", "coarse"
        return "coarse_grid", "coarse"

    # Stagnation — force exploration of new regions via LHC
    if stagnation_batches >= stagnation_threshold:
        return "latin_hypercube", "coarse"

    # Large search pass — structured coverage of the full space
    if cycle_mode == "large-search":
        return "latin_hypercube", "coarse"

    # Saturation escape — also use LHC
    if cycle_mode == "saturation_escape":
        return "latin_hypercube", "coarse"

    # Family default
    return profile["default_method"], profile["default_step_policy"]


def _is_structurally_novel(
    family: str,
    candidate: dict[str, Any],
    reference_configs: list[dict[str, Any]],
) -> bool:
    """
    Return True if this candidate differs from ALL reference configs on at
    least one architecture param (i.e. not just a parameter tweak).
    """
    arch_params = get_search_profile(family).get("architecture_params", [])
    if not arch_params or not reference_configs:
        return True
    for ref in reference_configs:
        # Candidate is NOT novel w.r.t. this reference if all arch params match
        if all(candidate.get(p) == ref.get(p) for p in arch_params):
            return False
    return True


# ── Sampling primitives ───────────────────────────────────────────────────────

def coarse_grid_candidates(
    family: str,
    n: int,
    seed: int,
    explored_hashes: set[str] | None = None,
    step_policy: str = "coarse",
) -> list[dict[str, Any]]:
    """
    Random sample from the coarse (or fine) step set.
    Never enumerates the full Cartesian product — samples n configs.
    Respects family-specific conditional parameter rules.
    """
    explored_hashes = explored_hashes or set()
    choices_map = get_step_sets(family).get(step_policy, get_step_sets(family)["coarse"])
    space = get_family_search_space(family)
    cond_rules = _CONDITIONAL_PARAMS.get(family, {})
    rng = random.Random(seed)

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(300, n * 40)

    for _ in range(max_attempts):
        if len(candidates) >= n:
            break
        raw: dict[str, Any] = {}
        # First pass: sample non-conditional params
        for name, spec in space.items():
            if name in cond_rules:
                continue
            param_choices = choices_map.get(name, spec["choices"])
            raw[name] = rng.choice(param_choices) if param_choices else spec["default"]
        # Second pass: conditional params (depend on values already set)
        for name, rule in cond_rules.items():
            spec = space.get(name)
            if spec is None:
                continue
            raw[name] = _resolve_conditional(name, rule, raw, choices_map, spec, rng)
        try:
            config = normalize_experiment_config(family, raw)
        except ValueError:
            continue
        h = compute_config_hash(family, config)
        if h in explored_hashes or h in seen:
            continue
        seen.add(h)
        candidates.append(config)

    return candidates


def latin_hypercube_candidates(
    family: str,
    n: int,
    seed: int,
    explored_hashes: set[str] | None = None,
    step_policy: str = "coarse",
) -> list[dict[str, Any]]:
    """
    Latin Hypercube Sampling over numeric dimensions, random for categoricals.
    Ensures uniform marginal coverage without brute-force enumeration.
    Respects family-specific conditional parameter rules.
    """
    explored_hashes = explored_hashes or set()
    choices_map = get_step_sets(family).get(step_policy, get_step_sets(family)["coarse"])
    space = get_family_search_space(family)
    cond_rules = _CONDITIONAL_PARAMS.get(family, {})
    rng = random.Random(seed)

    # Numeric params that are not conditional get LHC strata
    numeric_names = [
        name for name, spec in space.items()
        if spec["type"] in {"int", "float"}
        and name not in cond_rules
        and choices_map.get(name)
    ]
    other_names = [name for name in space if name not in set(numeric_names)]

    # Build per-param strata: tile choices to n slots, scramble correlations
    param_strata: dict[str, list[Any]] = {}
    for name in numeric_names:
        param_choices = list(choices_map.get(name, space[name]["choices"]))
        k = len(param_choices)
        if k == 0:
            continue
        rng.shuffle(param_choices)
        tiled = (param_choices * math.ceil(n / k))[:n]
        rng.shuffle(tiled)
        param_strata[name] = tiled

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(n * 4, n + 100)

    for i in range(max_attempts):
        if len(candidates) >= n:
            break
        raw: dict[str, Any] = {}
        for name in numeric_names:
            strata = param_strata.get(name)
            raw[name] = strata[i % len(strata)] if strata else space[name]["default"]
        for name in other_names:
            if name in cond_rules:
                continue
            spec = space[name]
            param_choices = choices_map.get(name, spec["choices"])
            raw[name] = rng.choice(param_choices) if param_choices else spec["default"]
        # Conditional params
        for name, rule in cond_rules.items():
            spec = space.get(name)
            if spec is None:
                continue
            raw[name] = _resolve_conditional(name, rule, raw, choices_map, spec, rng)
        try:
            config = normalize_experiment_config(family, raw)
        except ValueError:
            continue
        h = compute_config_hash(family, config)
        if h in explored_hashes or h in seen:
            continue
        seen.add(h)
        candidates.append(config)

    return candidates


def local_refinement_candidates(
    family: str,
    center_config: dict[str, Any],
    n: int,
    seed: int,
    explored_hashes: set[str] | None = None,
    perturb_count: int = 2,
) -> list[dict[str, Any]]:
    """
    Perturb 1–2 params of a known-good config using the fine step set.
    Explores the local neighbourhood without re-running the center itself.
    """
    explored_hashes = explored_hashes or set()
    fine_choices = get_step_sets(family)["fine"]
    space = get_family_search_space(family)
    rng = random.Random(seed)

    mutable = [
        name for name in space
        if name in fine_choices and len(fine_choices[name]) > 1
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_attempts = max(300, n * 30)

    for _ in range(max_attempts):
        if len(candidates) >= n:
            break
        n_perturb = rng.randint(1, max(1, min(perturb_count, len(mutable))))
        params_to_change = rng.sample(mutable, n_perturb)
        raw = dict(center_config)
        for name in params_to_change:
            param_choices = fine_choices.get(name, space[name]["choices"])
            current = raw.get(name)
            alternatives = [v for v in param_choices if v != current]
            if alternatives:
                raw[name] = rng.choice(alternatives)
        try:
            config = normalize_experiment_config(family, raw)
        except ValueError:
            continue
        h = compute_config_hash(family, config)
        if h in explored_hashes or h in seen:
            continue
        seen.add(h)
        candidates.append(config)

    return candidates


# ── Private helpers ───────────────────────────────────────────────────────────

def _resolve_conditional(
    name: str,
    rule: dict[str, Any],
    raw: dict[str, Any],
    choices_map: dict[str, list[Any]],
    spec: dict[str, Any],
    rng: random.Random,
) -> Any:
    """Apply a conditional parameter rule and return the sampled value."""
    active_when = rule.get("active_when", {})
    active_when_not_null = rule.get("active_when_not_null")
    active = True
    for cond_param, cond_values in active_when.items():
        if raw.get(cond_param) not in cond_values:
            active = False
            break
    if active_when_not_null:
        if raw.get(active_when_not_null) is None:
            active = False
    if not active:
        return None if spec.get("nullable") else spec.get("default")
    param_choices = choices_map.get(name, spec["choices"])
    return rng.choice(param_choices) if param_choices else spec["default"]


# ── Main entry point ──────────────────────────────────────────────────────────

def smart_sample_candidates(
    family: str,
    n: int,
    seed: int,
    *,
    cycle_mode: str = "normal_exploration",
    history_count: int = 0,
    stagnation_batches: int = 0,
    top_configs: list[dict[str, Any]] | None = None,
    explored_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Unified smart-search entry point.

    Selects the search method and step resolution based on family profile and
    cycle context, then generates n non-duplicate candidate configs.

    Returns:
        (candidates, metadata) where metadata contains:
            search_method, step_policy, coarse_or_fine, quantized, capped,
            cycle_mode, family, fallback_used, structurally_novel_count,
            architecture_params, top_configs_used, candidates_generated
    """
    family_norm = family.strip().lower()
    explored_hashes = explored_hashes or set()
    top_configs = top_configs or []
    method, step_policy = select_search_method(
        cycle_mode, history_count, stagnation_batches, family=family_norm
    )
    arch_params = get_search_profile(family_norm).get("architecture_params", [])

    metadata: dict[str, Any] = {
        "search_method": method,
        "step_policy": step_policy,
        "coarse_or_fine": step_policy,
        "quantized": True,
        "capped": True,
        "cycle_mode": cycle_mode,
        "family": family_norm,
        "architecture_params": arch_params,
        "fallback_used": False,
        "top_configs_used": len(top_configs),
    }

    if method == "coarse_grid":
        candidates = coarse_grid_candidates(family_norm, n, seed, explored_hashes, step_policy)

    elif method == "latin_hypercube":
        candidates = latin_hypercube_candidates(family_norm, n, seed, explored_hashes, step_policy)

    elif method == "local_refinement":
        if top_configs:
            all_c: list[dict[str, Any]] = []
            seen_set: set[str] = set(explored_hashes)
            per_center = max(2, math.ceil(n / len(top_configs)))
            for i, center in enumerate(top_configs):
                new = local_refinement_candidates(
                    family_norm, center, per_center, seed + i, seen_set
                )
                for c in new:
                    h = compute_config_hash(family_norm, c)
                    if h not in seen_set:
                        seen_set.add(h)
                        all_c.append(c)
                    if len(all_c) >= n:
                        break
                if len(all_c) >= n:
                    break
            candidates = all_c[:n]
        else:
            # No top configs available — fine-resolution coarse grid
            candidates = coarse_grid_candidates(family_norm, n, seed, explored_hashes, "fine")

    else:  # hybrid
        n_coarse = max(1, int(round(n * 0.60)))
        n_local = max(0, n - n_coarse)
        coarse = coarse_grid_candidates(family_norm, n_coarse, seed, explored_hashes, step_policy)
        coarse_hashes = {compute_config_hash(family_norm, c) for c in coarse}
        merged = explored_hashes | coarse_hashes
        if top_configs and n_local > 0:
            local_all: list[dict[str, Any]] = []
            seen_set2: set[str] = set(merged)
            per_center = max(2, math.ceil(n_local / len(top_configs)))
            for i, center in enumerate(top_configs):
                new = local_refinement_candidates(
                    family_norm, center, per_center, seed + i + 100, seen_set2
                )
                for c in new:
                    h = compute_config_hash(family_norm, c)
                    if h not in seen_set2:
                        seen_set2.add(h)
                        local_all.append(c)
                    if len(local_all) >= n_local:
                        break
                if len(local_all) >= n_local:
                    break
            local = local_all[:n_local]
        else:
            local = coarse_grid_candidates(family_norm, n_local, seed + 100, merged, step_policy)
        candidates = coarse + local

    # Deduplicate final list
    final: list[dict[str, Any]] = []
    final_hashes: set[str] = set(explored_hashes)
    for c in candidates:
        h = compute_config_hash(family_norm, c)
        if h not in final_hashes:
            final_hashes.add(h)
            final.append(c)

    # Fallback: fill remaining slots via full random sampling
    if len(final) < n:
        fallback = sample_random_candidates(family_norm, n * 4, seed + 7777)
        for c in fallback:
            h = compute_config_hash(family_norm, c)
            if h not in final_hashes:
                final_hashes.add(h)
                final.append(c)
            if len(final) >= n:
                break
        metadata["fallback_used"] = True

    result = final[:n]

    # Structural novelty count vs top_configs
    if arch_params and top_configs:
        novel_count = sum(
            1 for c in result if _is_structurally_novel(family_norm, c, top_configs)
        )
        metadata["structurally_novel_count"] = novel_count
    else:
        metadata["structurally_novel_count"] = len(result)

    metadata["candidates_generated"] = len(result)
    return result, metadata

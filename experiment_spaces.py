from __future__ import annotations

import itertools
import random
from typing import Any


RETIRED_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "amihud_illiquidity_premium": {
        "lookback_days": {"name": "lookback_days", "type": "int", "default": 60, "choices": [40, 60, 90], "log_scale": False, "nullable": False},
        "rebalance_days": {"name": "rebalance_days", "type": "int", "default": 42, "choices": [21, 42], "log_scale": False, "nullable": False},
        "max_positions": {"name": "max_positions", "type": "int", "default": 10, "choices": [5, 10, 20], "log_scale": False, "nullable": False},
        "min_dollar_volume": {"name": "min_dollar_volume", "type": "float", "default": 5_000_000.0, "choices": [2_000_000.0, 5_000_000.0, 10_000_000.0], "log_scale": False, "nullable": False},
        "price_min": {"name": "price_min", "type": "float", "default": 10.0, "choices": [5.0, 10.0], "log_scale": False, "nullable": False},
        "max_dollar_volume_rank": {"name": "max_dollar_volume_rank", "type": "int", "default": 500, "choices": [250, 500, None], "log_scale": False, "nullable": True},
        "max_daily_volatility": {"name": "max_daily_volatility", "type": "float", "default": 0.035, "choices": [0.035, 0.05, None], "log_scale": False, "nullable": True},
        "max_abs_open_gap": {"name": "max_abs_open_gap", "type": "float", "default": 0.20, "choices": [0.15, 0.20, None], "log_scale": False, "nullable": True},
    },
    "fear_greed_contrarian": {
        "fear_entry": {"name": "fear_entry", "type": "float", "default": 25.0, "choices": [20.0, 25.0, 35.0], "log_scale": False, "nullable": False},
        "greed_exit": {"name": "greed_exit", "type": "float", "default": 55.0, "choices": [50.0, 55.0, 70.0], "log_scale": False, "nullable": False},
        "confirmation_days": {"name": "confirmation_days", "type": "int", "default": 1, "choices": [1, 2, 3], "log_scale": False, "nullable": False},
        "rebalance_days": {"name": "rebalance_days", "type": "int", "default": 21, "choices": [10, 21, 42], "log_scale": False, "nullable": False},
        "max_positions": {"name": "max_positions", "type": "int", "default": 25, "choices": [15, 25, 50], "log_scale": False, "nullable": False},
        "min_dollar_volume": {"name": "min_dollar_volume", "type": "float", "default": 2_000_000.0, "choices": [1_000_000.0, 2_000_000.0, 5_000_000.0], "log_scale": False, "nullable": False},
        "exposure": {"name": "exposure", "type": "float", "default": 0.75, "choices": [0.50, 0.75, 1.00], "log_scale": False, "nullable": False},
    },
}

FAMILY_SEARCH_STATUS: dict[str, dict[str, str]] = {
    "momentum": {
        "status": "active",
        "canonical_status": "active",
        "lane": "primary",
        "reason": "Core benchmark family; keep active search and runtime refinement fully enabled.",
    },
    "superstock": {
        "status": "active",
        "canonical_status": "active",
        "lane": "primary",
        "reason": "Core breakout family; keep active search and runtime refinement fully enabled.",
    },
    "ml_ranker": {
        "status": "active",
        "canonical_status": "active",
        "lane": "primary",
        "reason": "Core architecture family; keep active search enabled.",
    },
    "rl_bandit": {
        "status": "controlled_candidate",
        "canonical_status": "controlled_candidate",
        "lane": "bounded",
        "reason": "Useful but still exploratory; keep bounded and explicit so it does not dominate the search budget.",
    },
    "breadth_timing_overlay": {
        "status": "paused",
        "canonical_status": "paused",
        "lane": "paused",
        "reason": "Prototype overlay is not yet a live search target; keep it out of normal budget allocation.",
    },
    "amihud_illiquidity_premium": {
        "status": "retired_for_now",
        "canonical_status": "retired_for_now",
        "lane": "retired",
        "reason": (
            "Diagnostic audit found standalone Amihud selection behaved like fragility/distress exposure in this execution engine."
        ),
    },
    "fear_greed_contrarian": {
        "status": "retired_for_now",
        "canonical_status": "retired_for_now",
        "lane": "retired",
        "reason": (
            "Stock-basket v1 was superseded by fear_greed_contrarian_overlay after diagnostics showed the signal is cleaner as market exposure timing."
        ),
    },
    "fear_greed_contrarian_overlay": {
        "status": "redesign_candidate",
        "canonical_status": "redesign_candidate",
        "lane": "redesign",
        "reason": "Keep redesign candidate out of normal search until the overlay is redesigned around stronger evidence.",
    },
    "momentum_fear_greed_overlay": {
        "status": "redesign_candidate",
        "canonical_status": "redesign_candidate",
        "lane": "redesign",
        "reason": "Keep redesign candidate out of normal search until the sleeve overlay can be redesigned into a stronger candidate.",
    },
    "sector_breadth_overlay": {
        "status": "redesign_candidate",
        "canonical_status": "redesign_candidate",
        "lane": "redesign",
        "reason": "Keep redesign candidate out of normal search until the overlay is redesigned with a clearer edge.",
    },
    "volatility_compression_expansion": {
        "status": "retired_for_now",
        "canonical_status": "retired_for_now",
        "lane": "retired",
        "reason": "Direct overlay comparison found the v1 model too dormant for live budget allocation.",
    },
    "event_revision_drift": {
        "status": "paused",
        "canonical_status": "paused",
        "lane": "paused",
        "reason": "Deferred until the required fundamental/event data exists.",
    },
    "mean_reversion": {
        "status": "redesign_candidate",
        "canonical_status": "redesign_candidate",
        "lane": "redesign",
        "reason": "Current implementation is cost-dominated; keep as a redesign candidate before more search.",
    },
    "vix_regime_sector_tilt": {
        "status": "paused",
        "canonical_status": "paused",
        "lane": "paused",
        "reason": "Keep deferred behind the repaired mean-reversion foundation.",
    },
}

_CANONICAL_STATUSES = {
    "active",
    "controlled_candidate",
    "redesign_candidate",
    "retired_for_now",
    "paused",
}
_CANONICAL_SEARCHABLE_STATUSES = {"active", "controlled_candidate"}
_LEGACY_STATUS_ALIASES = {
    "retired": "retired_for_now",
    "retire_for_now": "retired_for_now",
    "deferred": "paused",
    "deferred_missing_data": "paused",
}


def canonicalize_family_search_status(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "active"
    return _LEGACY_STATUS_ALIASES.get(text, text)


def _status_lane(status: str) -> str:
    if status == "active":
        return "primary"
    if status == "controlled_candidate":
        return "bounded"
    if status == "redesign_candidate":
        return "redesign"
    if status == "retired_for_now":
        return "retired"
    return "paused"


def _normalized_status_entry(family: str, entry: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(entry or {})
    status = canonicalize_family_search_status(payload.get("status") or payload.get("canonical_status"))
    canonical_status = status
    searchable = bool(payload.get("searchable", canonical_status in _CANONICAL_SEARCHABLE_STATUSES))
    budget_scale = float(
        payload.get(
            "budget_scale",
            1.0 if canonical_status == "active" else 0.5 if canonical_status == "controlled_candidate" else 0.0,
        )
    )
    return {
        "family": family.strip().lower(),
        "status": status,
        "canonical_status": canonical_status,
        "searchable": searchable,
        "budget_scale": budget_scale,
        "lane": str(payload.get("lane") or _status_lane(canonical_status)),
        "reason": str(
            payload.get("reason")
            or (
                "active searchable family"
                if canonical_status == "active"
                else "controlled exploratory family"
                if canonical_status == "controlled_candidate"
                else "redesign candidate kept out of normal search"
                if canonical_status == "redesign_candidate"
                else "retired family excluded from normal search"
                if canonical_status == "retired_for_now"
                else "paused family excluded from normal search"
            )
        ),
    }


def build_family_search_status_registry(families: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if families is None:
        requested = sorted(
            {
                *SEARCH_SPACES.keys(),
                *RETIRED_SEARCH_SPACES.keys(),
                *FAMILY_SEARCH_STATUS.keys(),
            }
        )
    else:
        requested = [str(family).strip().lower() for family in families if str(family).strip()]
    return {
        family: get_family_search_status(family)
        for family in requested
    }


SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "breadth_timing_overlay": {
        "breadth_window": {"name": "breadth_window", "type": "int", "default": 200, "choices": [100, 150, 200], "log_scale": False, "nullable": False},
        "entry_threshold": {"name": "entry_threshold", "type": "float", "default": 0.55, "choices": [0.50, 0.55, 0.60], "log_scale": False, "nullable": False},
        "exit_threshold": {"name": "exit_threshold", "type": "float", "default": 0.45, "choices": [0.40, 0.45, 0.50], "log_scale": False, "nullable": False},
        "top_n": {"name": "top_n", "type": "int", "default": 100, "choices": [50, 100, 200], "log_scale": False, "nullable": False},
    },
    "mean_reversion": {
        "lookback_days": {"name": "lookback_days", "type": "int", "default": 10, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "max_positions": {"name": "max_positions", "type": "int", "default": 10, "choices": [8, 10, 12], "log_scale": False, "nullable": False},
        "max_sector_positions": {"name": "max_sector_positions", "type": "int", "default": 2, "choices": [1, 2, 3], "log_scale": False, "nullable": False},
        "min_hold_days": {"name": "min_hold_days", "type": "int", "default": 10, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "cooldown_days": {"name": "cooldown_days", "type": "int", "default": 10, "choices": [5, 10, 20], "log_scale": False, "nullable": False},
        "min_dollar_volume": {"name": "min_dollar_volume", "type": "float", "default": 2_000_000.0, "choices": [1_000_000.0, 2_000_000.0, 5_000_000.0], "log_scale": False, "nullable": False},
        "max_daily_volatility": {"name": "max_daily_volatility", "type": "float", "default": 0.05, "choices": [0.04, 0.05, 0.06], "log_scale": False, "nullable": False},
        "max_abs_open_gap": {"name": "max_abs_open_gap", "type": "float", "default": 0.10, "choices": [0.08, 0.10, 0.12], "log_scale": False, "nullable": False},
        "dislocation_z": {"name": "dislocation_z", "type": "float", "default": 1.0, "choices": [1.0, 1.25, 1.50], "log_scale": False, "nullable": False},
        "rebal_days": {"name": "rebal_days", "type": "int", "default": 10, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "vix_gate": {"name": "vix_gate", "type": "float", "default": 35.0, "choices": [None, 35.0, 40.0], "log_scale": False, "nullable": True},
    },
    "fear_greed_contrarian_overlay": {
        "entry_threshold": {"name": "entry_threshold", "type": "float", "default": 25.0, "choices": [20.0, 25.0, 35.0], "log_scale": False, "nullable": False},
        "exit_threshold": {"name": "exit_threshold", "type": "float", "default": 55.0, "choices": [50.0, 55.0, 70.0], "log_scale": False, "nullable": False},
        "exposure": {"name": "exposure", "type": "float", "default": 0.75, "choices": [0.50, 0.75], "log_scale": False, "nullable": False},
        "confirmation_days": {"name": "confirmation_days", "type": "int", "default": 1, "choices": [1, 2, 3], "log_scale": False, "nullable": False},
        "market_symbol": {"name": "market_symbol", "type": "categorical", "default": "SPY", "choices": ["SPY"], "log_scale": False, "nullable": False},
    },
    "ml_ranker": {
        "model_type": {"name": "model_type", "type": "categorical", "default": "ridge", "choices": ["ridge", "hgb"], "log_scale": False, "nullable": False},
        "lookback_days": {"name": "lookback_days", "type": "int", "default": 252, "choices": [126, 252, 504], "log_scale": False, "nullable": False},
        "horizon_days": {"name": "horizon_days", "type": "int", "default": 10, "choices": [5, 10, 20], "log_scale": False, "nullable": False},
        "rebalance_days": {"name": "rebalance_days", "type": "int", "default": 5, "choices": [5, 10], "log_scale": False, "nullable": False},
        "top_pct": {"name": "top_pct", "type": "float", "default": 0.03, "choices": [0.025, 0.03, 0.05], "log_scale": False, "nullable": False},
        "max_positions": {"name": "max_positions", "type": "int", "default": 10, "choices": [5, 10, 20], "log_scale": False, "nullable": False},
        "feature_set": {"name": "feature_set", "type": "categorical", "default": "trend_volume", "choices": ["trend", "trend_volume", "full"], "log_scale": False, "nullable": False},
        "allow_short": {"name": "allow_short", "type": "bool", "default": False, "choices": [False, True], "log_scale": False, "nullable": False},
        "use_vix_gate": {"name": "use_vix_gate", "type": "bool", "default": True, "choices": [False, True], "log_scale": False, "nullable": False},
        "use_fear_greed_gate": {"name": "use_fear_greed_gate", "type": "bool", "default": False, "choices": [False, True], "log_scale": False, "nullable": False},
    },
    "momentum": {
        "LOOKBACK_WEEKS": {"name": "LOOKBACK_WEEKS", "type": "int", "default": 26, "choices": [20, 26, 39], "log_scale": False, "nullable": False},
        "SKIP_WEEKS": {"name": "SKIP_WEEKS", "type": "int", "default": 3, "choices": [2, 3, 4], "log_scale": False, "nullable": False},
        "REBAL_WEEKS": {"name": "REBAL_WEEKS", "type": "int", "default": 4, "choices": [2, 4], "log_scale": False, "nullable": False},
        "TOP_PCT": {"name": "TOP_PCT", "type": "float", "default": 0.025, "choices": [0.015, 0.025, 0.03], "log_scale": False, "nullable": False},
        "MA_WEEKS": {"name": "MA_WEEKS", "type": "int", "default": 20, "choices": [16, 20, 24], "log_scale": False, "nullable": False},
        "STOP_TYPE": {"name": "STOP_TYPE", "type": "categorical", "default": "adaptive", "choices": ["adaptive", "fixed", "none"], "log_scale": False, "nullable": False},
        "STOP_LOSS_PCT": {"name": "STOP_LOSS_PCT", "type": "float", "default": 0.20, "choices": [0.15, 0.20, 0.25, 0.30], "log_scale": False, "nullable": True},
        "STOP_PARABOLIC": {"name": "STOP_PARABOLIC", "type": "float", "default": 0.30, "choices": [0.25, 0.30, 0.35, 0.40], "log_scale": False, "nullable": True},
        "INV_VOL_DAYS": {"name": "INV_VOL_DAYS", "type": "int", "default": 15, "choices": [15], "log_scale": False, "nullable": False},
        "MIN_HOLD_DAYS": {"name": "MIN_HOLD_DAYS", "type": "int", "default": 5, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "FG_MIN": {"name": "FG_MIN", "type": "float", "default": 10.0, "choices": [10.0, 22.0, 30.0], "log_scale": False, "nullable": False},
        "EXIT_PCT_RANK": {"name": "EXIT_PCT_RANK", "type": "float", "default": 0.97, "choices": [None, 0.94, 0.95, 0.97], "log_scale": False, "nullable": True},
        "RANK_EXIT_CONFIRM": {"name": "RANK_EXIT_CONFIRM", "type": "int", "default": None, "choices": [None, 1, 2], "log_scale": False, "nullable": True},
    },
    "momentum_fear_greed_overlay": {
        "entry_threshold": {"name": "entry_threshold", "type": "float", "default": 25.0, "choices": [20.0, 25.0, 35.0], "log_scale": False, "nullable": False},
        "exit_threshold": {"name": "exit_threshold", "type": "float", "default": 55.0, "choices": [50.0, 55.0, 70.0], "log_scale": False, "nullable": False},
        "greed_threshold": {"name": "greed_threshold", "type": "float", "default": 75.0, "choices": [65.0, 75.0, 85.0], "log_scale": False, "nullable": False},
        "fear_exposure": {"name": "fear_exposure", "type": "float", "default": 0.75, "choices": [0.50, 0.75, 1.00], "log_scale": False, "nullable": False},
        "normal_exposure": {"name": "normal_exposure", "type": "float", "default": 1.00, "choices": [0.75, 1.00], "log_scale": False, "nullable": False},
        "greed_exposure": {"name": "greed_exposure", "type": "float", "default": 0.75, "choices": [0.50, 0.75, 1.00], "log_scale": False, "nullable": False},
        "confirmation_days": {"name": "confirmation_days", "type": "int", "default": 2, "choices": [1, 2, 3], "log_scale": False, "nullable": False},
    },
    "superstock": {
        "max_positions": {"name": "max_positions", "type": "int", "default": 5, "choices": [3, 5, 8], "log_scale": False, "nullable": False},
        "price_min": {"name": "price_min", "type": "float", "default": 5.0, "choices": [3.0, 5.0], "log_scale": False, "nullable": False},
        "price_max": {"name": "price_max", "type": "float", "default": 15.0, "choices": [15.0, 20.0], "log_scale": False, "nullable": False},
        "min_dollar_volume": {"name": "min_dollar_volume", "type": "float", "default": 1_000_000.0, "choices": [500_000.0, 1_000_000.0, 2_000_000.0], "log_scale": False, "nullable": False},
        "above_52w_low_mult": {"name": "above_52w_low_mult", "type": "float", "default": 1.25, "choices": [1.25], "log_scale": False, "nullable": False},
        "near_52w_high_mult": {"name": "near_52w_high_mult", "type": "float", "default": 0.75, "choices": [0.75], "log_scale": False, "nullable": False},
        "rs_rank_26w_min": {"name": "rs_rank_26w_min", "type": "float", "default": 0.70, "choices": [0.60, 0.70, 0.80], "log_scale": False, "nullable": False},
        "rs_rank_52w_min": {"name": "rs_rank_52w_min", "type": "float", "default": 0.70, "choices": [0.60, 0.70, 0.80], "log_scale": False, "nullable": False},
        "base_depth_max": {"name": "base_depth_max", "type": "float", "default": 0.60, "choices": [0.40, 0.60, 0.80], "log_scale": False, "nullable": False},
        "weekly_range_median_max": {"name": "weekly_range_median_max", "type": "float", "default": 0.18, "choices": [0.15, 0.18, 0.22], "log_scale": False, "nullable": False},
        "weekly_volatility_max": {"name": "weekly_volatility_max", "type": "float", "default": 0.12, "choices": [0.10, 0.12, 0.15], "log_scale": False, "nullable": False},
        "volume_dryup_max": {"name": "volume_dryup_max", "type": "float", "default": 0.80, "choices": [0.80, 0.90, 1.00], "log_scale": False, "nullable": False},
        "vix_hard_cap": {"name": "vix_hard_cap", "type": "float", "default": 35.0, "choices": [30.0, 35.0, 40.0], "log_scale": False, "nullable": False},
        "vix_ma_multiplier": {"name": "vix_ma_multiplier", "type": "float", "default": 1.25, "choices": [1.10, 1.25, 1.50], "log_scale": False, "nullable": False},
        "breakout_extension_max": {"name": "breakout_extension_max", "type": "float", "default": 0.10, "choices": [0.08, 0.10, 0.15], "log_scale": False, "nullable": False},
        "daily_volume_expansion_mult": {"name": "daily_volume_expansion_mult", "type": "float", "default": 1.50, "choices": [1.25, 1.50, 2.00], "log_scale": False, "nullable": False},
        "daily_dollar_volume_expansion_mult": {"name": "daily_dollar_volume_expansion_mult", "type": "float", "default": 1.50, "choices": [1.25, 1.50, 2.00], "log_scale": False, "nullable": False},
        "parabolic_from_pivot_min": {"name": "parabolic_from_pivot_min", "type": "float", "default": 0.25, "choices": [0.20, 0.25, 0.30], "log_scale": False, "nullable": False},
        "parabolic_above_10w_min": {"name": "parabolic_above_10w_min", "type": "float", "default": 0.20, "choices": [0.15, 0.20, 0.25], "log_scale": False, "nullable": False},
        "late_stage_range_mult": {"name": "late_stage_range_mult", "type": "float", "default": 1.75, "choices": [1.50, 1.75, 2.00], "log_scale": False, "nullable": False},
        "late_stage_volume_mult": {"name": "late_stage_volume_mult", "type": "float", "default": 2.00, "choices": [1.50, 2.00, 2.50], "log_scale": False, "nullable": False},
    },
    "vix_regime_sector_tilt": {
        "cyclical_sectors_mode": {"name": "cyclical_sectors_mode", "type": "categorical", "default": "narrow", "choices": ["narrow", "broad"], "log_scale": False, "nullable": False},
        "high_vix_threshold": {"name": "high_vix_threshold", "type": "float", "default": 28.0, "choices": [25.0, 28.0, 32.0], "log_scale": False, "nullable": False},
        "low_vix_threshold": {"name": "low_vix_threshold", "type": "float", "default": 18.0, "choices": [15.0, 18.0, 20.0], "log_scale": False, "nullable": False},
        "vix_smooth_days": {"name": "vix_smooth_days", "type": "int", "default": 5, "choices": [3, 5, 10], "log_scale": False, "nullable": False},
    },
    "rl_bandit": {
        "policy_type": {"name": "policy_type", "type": "categorical", "default": "ucb", "choices": ["ucb", "epsilon_greedy"], "log_scale": False, "nullable": False},
        "lookback_days": {"name": "lookback_days", "type": "int", "default": 252, "choices": [126, 252], "log_scale": False, "nullable": False},
        "rebalance_days": {"name": "rebalance_days", "type": "int", "default": 5, "choices": [5, 10], "log_scale": False, "nullable": False},
        "epsilon": {"name": "epsilon", "type": "float", "default": 0.10, "choices": [0.05, 0.10, 0.20], "log_scale": False, "nullable": False},
        "ucb_bonus": {"name": "ucb_bonus", "type": "float", "default": 1.0, "choices": [0.5, 1.0, 2.0], "log_scale": False, "nullable": False},
        "max_positions": {"name": "max_positions", "type": "int", "default": 5, "choices": [3, 5, 8], "log_scale": False, "nullable": False},
        "momentum_top_pct": {"name": "momentum_top_pct", "type": "float", "default": 0.03, "choices": [0.025, 0.03, 0.05], "log_scale": False, "nullable": False},
        "superstock_top_pct": {"name": "superstock_top_pct", "type": "float", "default": 0.03, "choices": [0.025, 0.03, 0.05], "log_scale": False, "nullable": False},
        "use_vix_gate": {"name": "use_vix_gate", "type": "bool", "default": True, "choices": [False, True], "log_scale": False, "nullable": False},
        "use_fear_greed_gate": {"name": "use_fear_greed_gate", "type": "bool", "default": True, "choices": [False, True], "log_scale": False, "nullable": False},
    },
    "sector_breadth_overlay": {
        "breadth_window": {"name": "breadth_window", "type": "int", "default": 100, "choices": [80, 100, 150], "log_scale": False, "nullable": False},
        "sector_on_threshold": {"name": "sector_on_threshold", "type": "float", "default": 0.55, "choices": [0.50, 0.55, 0.60], "log_scale": False, "nullable": False},
        "sector_off_threshold": {"name": "sector_off_threshold", "type": "float", "default": 0.45, "choices": [0.40, 0.45, 0.50], "log_scale": False, "nullable": False},
        "broad_sector_fraction": {"name": "broad_sector_fraction", "type": "float", "default": 0.55, "choices": [0.50, 0.55, 0.65], "log_scale": False, "nullable": False},
        "risk_on_exposure": {"name": "risk_on_exposure", "type": "float", "default": 0.75, "choices": [0.50, 0.75], "log_scale": False, "nullable": False},
        "selective_exposure": {"name": "selective_exposure", "type": "float", "default": 0.50, "choices": [0.35, 0.50], "log_scale": False, "nullable": False},
        "defensive_exposure": {"name": "defensive_exposure", "type": "float", "default": 0.35, "choices": [0.25, 0.35], "log_scale": False, "nullable": False},
        "min_selective_sectors": {"name": "min_selective_sectors", "type": "int", "default": 3, "choices": [2, 3, 4], "log_scale": False, "nullable": False},
        "market_symbol": {"name": "market_symbol", "type": "categorical", "default": "SPY", "choices": ["SPY"], "log_scale": False, "nullable": False},
    },
    "volatility_compression_expansion": {
        "compression_window": {"name": "compression_window", "type": "int", "default": 20, "choices": [15, 20, 30], "log_scale": False, "nullable": False},
        "baseline_window": {"name": "baseline_window", "type": "int", "default": 100, "choices": [80, 100, 150], "log_scale": False, "nullable": False},
        "breakout_window": {"name": "breakout_window", "type": "int", "default": 20, "choices": [15, 20, 30], "log_scale": False, "nullable": False},
        "compression_ratio": {"name": "compression_ratio", "type": "float", "default": 0.70, "choices": [0.60, 0.70, 0.80], "log_scale": False, "nullable": False},
        "expansion_mult": {"name": "expansion_mult", "type": "float", "default": 1.20, "choices": [1.10, 1.20, 1.40], "log_scale": False, "nullable": False},
        "exit_expansion_mult": {"name": "exit_expansion_mult", "type": "float", "default": 1.50, "choices": [1.30, 1.50, 1.80], "log_scale": False, "nullable": False},
        "exposure": {"name": "exposure", "type": "float", "default": 0.50, "choices": [0.35, 0.50, 0.75], "log_scale": False, "nullable": False},
        "compression_lookback": {"name": "compression_lookback", "type": "int", "default": 10, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "market_symbol": {"name": "market_symbol", "type": "categorical", "default": "SPY", "choices": ["SPY"], "log_scale": False, "nullable": False},
    },
}


def list_searchable_families() -> list[str]:
    registry = build_family_search_status_registry()
    return sorted(
        family
        for family, entry in registry.items()
        if entry.get("canonical_status") in _CANONICAL_SEARCHABLE_STATUSES and bool(entry.get("searchable", True))
    )


def get_family_search_status(family: str) -> dict[str, Any]:
    normalized = family.strip().lower()
    if normalized in FAMILY_SEARCH_STATUS:
        return _normalized_status_entry(normalized, FAMILY_SEARCH_STATUS.get(normalized))
    if normalized in RETIRED_SEARCH_SPACES:
        return {
            "family": normalized,
            "status": "retired_for_now",
            "canonical_status": "retired_for_now",
            "searchable": False,
            "budget_scale": 0.0,
            "lane": "retired",
            "reason": "retired family is excluded from normal search budget",
        }
    if normalized not in SEARCH_SPACES:
        return {
            "family": normalized,
            "status": "active",
            "canonical_status": "active",
            "searchable": True,
            "budget_scale": 1.0,
            "lane": "primary",
            "reason": "active searchable family",
        }
    return _normalized_status_entry(normalized, FAMILY_SEARCH_STATUS.get(normalized))


def list_active_families() -> list[str]:
    return sorted(
        family
        for family, status in build_family_search_status_registry().items()
        if status.get("canonical_status") == "active"
    )


def list_controlled_families() -> list[str]:
    return sorted(
        family
        for family, status in build_family_search_status_registry().items()
        if status.get("canonical_status") == "controlled_candidate"
    )


def list_redesign_families() -> list[str]:
    return sorted(
        family
        for family, status in build_family_search_status_registry().items()
        if status.get("canonical_status") == "redesign_candidate"
    )


def list_paused_families() -> list[str]:
    return sorted(
        family
        for family, status in build_family_search_status_registry().items()
        if status.get("canonical_status") == "paused"
    )


def list_retired_families() -> list[str]:
    return sorted(
        family
        for family, status in build_family_search_status_registry().items()
        if status.get("canonical_status") == "retired_for_now"
    )


def get_family_search_space(family: str) -> dict[str, dict]:
    normalized = family.strip().lower()
    if normalized in SEARCH_SPACES:
        return SEARCH_SPACES[normalized]
    if normalized in RETIRED_SEARCH_SPACES:
        return RETIRED_SEARCH_SPACES[normalized]
    if normalized not in SEARCH_SPACES:
        raise ValueError(f"Unknown experiment family '{family}'.")
    return SEARCH_SPACES[normalized]


def get_family_default_config(family: str) -> dict[str, Any]:
    space = get_family_search_space(family)
    return {name: spec["default"] for name, spec in space.items()}


def _coerce_value(spec: dict[str, Any], value: Any) -> Any:
    if value is None:
        if spec.get("nullable", False):
            return None
        raise ValueError(f"{spec['name']} does not allow null.")

    value_type = spec["type"]
    if value_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be an int.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ValueError(f"{spec['name']} must be an int.")
    if value_type == "float":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be a float.")
        try:
            return round(float(value), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec['name']} must be a float.") from exc
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{spec['name']} must be a bool.")
    if value_type == "categorical":
        return value
    raise ValueError(f"Unsupported type '{value_type}' for {spec['name']}.")


def validate_experiment_config(family: str, config: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        normalize_experiment_config(family, config)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def normalize_experiment_config(family: str, config: dict[str, Any]) -> dict[str, Any]:
    family = family.strip().lower()
    space = get_family_search_space(family)
    config = dict(config or {})

    unknown = sorted(set(config) - set(space))
    if unknown:
        raise ValueError(f"Unknown config key(s) for {family}: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    for name, spec in space.items():
        raw_value = config[name] if name in config else spec["default"]
        value = _coerce_value(spec, raw_value)

        if value is None:
            normalized[name] = value
            continue

        choices = spec.get("choices")
        skip_choice_check = family == "superstock" and name in {"price_min", "price_max"}
        if choices is not None and not skip_choice_check and value not in choices:
            raise ValueError(f"{name} must be one of {choices}.")

        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}.")

        normalized[name] = value

    if family == "momentum":
        stop_type = normalized["STOP_TYPE"]
        if stop_type == "none":
            normalized["STOP_LOSS_PCT"] = None
            normalized["STOP_PARABOLIC"] = None
        elif stop_type == "fixed":
            normalized["STOP_PARABOLIC"] = None
        if normalized["EXIT_PCT_RANK"] is None:
            normalized["RANK_EXIT_CONFIRM"] = None

    if family == "superstock":
        price_min = float(normalized["price_min"])
        price_max = float(normalized["price_max"])
        normalized["price_min"] = min(price_min, price_max)
        normalized["price_max"] = max(price_min, price_max)

    if family == "breadth_timing_overlay":
        if normalized["exit_threshold"] >= normalized["entry_threshold"]:
            normalized["exit_threshold"] = round(normalized["entry_threshold"] - 0.05, 6)

    if family == "vix_regime_sector_tilt":
        if normalized["low_vix_threshold"] >= normalized["high_vix_threshold"]:
            normalized["high_vix_threshold"] = round(normalized["low_vix_threshold"] + 5.0, 6)

    if family == "fear_greed_contrarian":
        if normalized["fear_entry"] >= normalized["greed_exit"]:
            normalized["greed_exit"] = round(normalized["fear_entry"] + 15.0, 6)
        normalized["exposure"] = min(max(float(normalized["exposure"]), 0.0), 1.0)

    if family == "fear_greed_contrarian_overlay":
        if normalized["entry_threshold"] >= normalized["exit_threshold"]:
            normalized["exit_threshold"] = round(normalized["entry_threshold"] + 15.0, 6)
        normalized["exposure"] = min(max(float(normalized["exposure"]), 0.0), 1.0)

    if family == "momentum_fear_greed_overlay":
        if normalized["entry_threshold"] >= normalized["exit_threshold"]:
            normalized["exit_threshold"] = round(normalized["entry_threshold"] + 15.0, 6)
        if normalized["greed_threshold"] <= normalized["exit_threshold"]:
            normalized["greed_threshold"] = round(normalized["exit_threshold"] + 10.0, 6)
        for key in ("fear_exposure", "normal_exposure", "greed_exposure"):
            normalized[key] = min(max(float(normalized[key]), 0.0), 1.0)

    if family == "sector_breadth_overlay":
        if normalized["sector_off_threshold"] >= normalized["sector_on_threshold"]:
            normalized["sector_off_threshold"] = round(normalized["sector_on_threshold"] - 0.05, 6)
        for key in ("risk_on_exposure", "selective_exposure", "defensive_exposure"):
            normalized[key] = min(max(float(normalized[key]), 0.0), 1.0)

    if family == "volatility_compression_expansion":
        if normalized["compression_window"] >= normalized["baseline_window"]:
            normalized["baseline_window"] = normalized["compression_window"] * 3
        normalized["exposure"] = min(max(float(normalized["exposure"]), 0.0), 1.0)

    return dict(sorted(normalized.items()))


def enumerate_grid_candidates(family: str, limit: int | None = None) -> list[dict[str, Any]]:
    space = get_family_search_space(family)
    keys = sorted(space)
    candidate_lists = [space[key]["choices"] for key in keys]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for values in itertools.product(*candidate_lists):
        candidate = normalize_experiment_config(family, dict(zip(keys, values, strict=False)))
        token = tuple(sorted(candidate.items()))
        if token in seen:
            continue
        seen.add(token)
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def sample_random_candidates(family: str, n: int, seed: int) -> list[dict[str, Any]]:
    space = get_family_search_space(family)
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    max_attempts = max(100, n * 20)
    attempts = 0

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        raw = {name: rng.choice(spec["choices"]) for name, spec in space.items()}
        candidate = normalize_experiment_config(family, raw)
        token = tuple(sorted(candidate.items()))
        if token in seen:
            continue
        seen.add(token)
        candidates.append(candidate)

    if len(candidates) < n:
        for candidate in enumerate_grid_candidates(family):
            token = tuple(sorted(candidate.items()))
            if token in seen:
                continue
            seen.add(token)
            candidates.append(candidate)
            if len(candidates) >= n:
                break
    return candidates


def normalize_experiment_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    return normalize_experiment_config(family, params)

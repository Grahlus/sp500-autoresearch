from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from experiment_spaces import get_family_default_config, normalize_experiment_config


@dataclass(frozen=True)
class IdeaTemplate:
    family: str
    template_id: str
    strategy_type: str
    source_type: str
    hypothesis: str
    reason_selected: str
    config: dict[str, Any]
    exploration_mode: str
    source_family: str | None = None
    tags: tuple[str, ...] = ()


def _templates() -> dict[str, list[IdeaTemplate]]:
    return {
        "momentum": [
            IdeaTemplate(
                family="momentum",
                template_id="momentum_balanced_default",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Balanced JT baseline with adaptive exits and moderate hold discipline",
                reason_selected="Anchor around the known runnable momentum baseline.",
                config={
                    "LOOKBACK_WEEKS": 26,
                    "SKIP_WEEKS": 3,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.025,
                    "MA_WEEKS": 20,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.20,
                    "STOP_PARABOLIC": 0.30,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 5,
                    "FG_MIN": 10.0,
                    "EXIT_PCT_RANK": 0.97,
                    "RANK_EXIT_CONFIRM": None,
                },
                exploration_mode="local_refinement",
                tags=("baseline", "control"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_fast_rotation",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Faster rotation with shorter lookback and tighter ranking filters.",
                reason_selected="Tests whether momentum edge is more responsive to shorter cycles.",
                config={
                    "LOOKBACK_WEEKS": 20,
                    "SKIP_WEEKS": 2,
                    "REBAL_WEEKS": 2,
                    "TOP_PCT": 0.03,
                    "MA_WEEKS": 16,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.15,
                    "STOP_PARABOLIC": 0.25,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 5,
                    "FG_MIN": 10.0,
                    "EXIT_PCT_RANK": 0.95,
                    "RANK_EXIT_CONFIRM": None,
                },
                exploration_mode="template_expansion",
                tags=("rotation", "reversal"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_quality_slow",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Lower-churn quality rotation with longer lookback and tighter selection.",
                reason_selected="Explores a slower, more selective momentum regime.",
                config={
                    "LOOKBACK_WEEKS": 39,
                    "SKIP_WEEKS": 4,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.015,
                    "MA_WEEKS": 24,
                    "STOP_TYPE": "fixed",
                    "STOP_LOSS_PCT": 0.25,
                    "STOP_PARABOLIC": None,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 10,
                    "FG_MIN": 22.0,
                    "EXIT_PCT_RANK": 0.94,
                    "RANK_EXIT_CONFIRM": None,
                },
                exploration_mode="template_expansion",
                tags=("quality", "slow"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_turtle_breakout",
                strategy_type="classical",
                source_type="cross_family_hybrid",
                hypothesis="Momentum with breakout-like holding discipline and tighter regime gating.",
                reason_selected="Borrowed from breakout/turtle concepts to widen the search.",
                config={
                    "LOOKBACK_WEEKS": 20,
                    "SKIP_WEEKS": 2,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.015,
                    "MA_WEEKS": 20,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.25,
                    "STOP_PARABOLIC": 0.30,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 15,
                    "FG_MIN": 22.0,
                    "EXIT_PCT_RANK": 0.95,
                    "RANK_EXIT_CONFIRM": 1,
                },
                exploration_mode="cross_family_hybrid",
                source_family="superstock",
                tags=("breakout", "turtle", "regime"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_regime_conservative",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="More conservative regime gate with slower exits to reduce drawdown spikes.",
                reason_selected="Tests whether regime gating and slower exits stabilize the equity curve.",
                config={
                    "LOOKBACK_WEEKS": 26,
                    "SKIP_WEEKS": 3,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.025,
                    "MA_WEEKS": 24,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.30,
                    "STOP_PARABOLIC": 0.35,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 10,
                    "FG_MIN": 30.0,
                    "EXIT_PCT_RANK": 0.94,
                    "RANK_EXIT_CONFIRM": 2,
                },
                exploration_mode="template_expansion",
                tags=("regime", "risk-control"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_breakout_bias",
                strategy_type="classical",
                source_type="cross_family_hybrid",
                hypothesis="Momentum with breakout-style shorter holding and stronger confirmation.",
                reason_selected="Expands beyond local tuning toward breakout behavior.",
                config={
                    "LOOKBACK_WEEKS": 20,
                    "SKIP_WEEKS": 2,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.03,
                    "MA_WEEKS": 20,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.25,
                    "STOP_PARABOLIC": 0.30,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 5,
                    "FG_MIN": 10.0,
                    "EXIT_PCT_RANK": 0.95,
                    "RANK_EXIT_CONFIRM": 1,
                },
                exploration_mode="cross_family_hybrid",
                source_family="superstock",
                tags=("breakout", "confirmation"),
            ),
            IdeaTemplate(
                family="momentum",
                template_id="momentum_volatility_gate",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Momentum with a stricter volatility filter and slower rotation.",
                reason_selected="Checks whether a tighter volatility regime improves durability.",
                config={
                    "LOOKBACK_WEEKS": 26,
                    "SKIP_WEEKS": 4,
                    "REBAL_WEEKS": 4,
                    "TOP_PCT": 0.025,
                    "MA_WEEKS": 24,
                    "STOP_TYPE": "adaptive",
                    "STOP_LOSS_PCT": 0.20,
                    "STOP_PARABOLIC": 0.35,
                    "INV_VOL_DAYS": 15,
                    "MIN_HOLD_DAYS": 10,
                    "FG_MIN": 30.0,
                    "EXIT_PCT_RANK": 0.94,
                    "RANK_EXIT_CONFIRM": 2,
                },
                exploration_mode="template_expansion",
                tags=("volatility", "regime"),
            ),
        ],
        "superstock": [
            IdeaTemplate(
                family="superstock",
                template_id="superstock_balanced_default",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Balanced Superstock breakout with moderate liquidity and RS thresholds.",
                reason_selected="Anchor around the current runnable Superstock baseline.",
                config={
                    "max_positions": 5,
                    "price_min": 5.0,
                    "price_max": 15.0,
                    "min_dollar_volume": 1_000_000.0,
                    "above_52w_low_mult": 1.25,
                    "near_52w_high_mult": 0.75,
                    "rs_rank_26w_min": 0.70,
                    "rs_rank_52w_min": 0.70,
                    "base_depth_max": 0.60,
                    "weekly_range_median_max": 0.18,
                    "weekly_volatility_max": 0.12,
                    "volume_dryup_max": 0.80,
                    "vix_hard_cap": 35.0,
                    "vix_ma_multiplier": 1.25,
                    "breakout_extension_max": 0.10,
                    "daily_volume_expansion_mult": 1.50,
                    "daily_dollar_volume_expansion_mult": 1.50,
                    "parabolic_from_pivot_min": 0.25,
                    "parabolic_above_10w_min": 0.20,
                    "late_stage_range_mult": 1.75,
                    "late_stage_volume_mult": 2.00,
                },
                exploration_mode="local_refinement",
                tags=("baseline", "control"),
            ),
            IdeaTemplate(
                family="superstock",
                template_id="superstock_high_liquidity",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Higher liquidity and higher RS filters to reduce noise and failed breakouts.",
                reason_selected="Explores a cleaner breakout universe with tighter quality control.",
                config={
                    "max_positions": 5,
                    "price_min": 5.0,
                    "price_max": 20.0,
                    "min_dollar_volume": 2_000_000.0,
                    "above_52w_low_mult": 1.25,
                    "near_52w_high_mult": 0.75,
                    "rs_rank_26w_min": 0.80,
                    "rs_rank_52w_min": 0.80,
                    "base_depth_max": 0.40,
                    "weekly_range_median_max": 0.15,
                    "weekly_volatility_max": 0.10,
                    "volume_dryup_max": 0.80,
                    "vix_hard_cap": 30.0,
                    "vix_ma_multiplier": 1.10,
                    "breakout_extension_max": 0.08,
                    "daily_volume_expansion_mult": 2.00,
                    "daily_dollar_volume_expansion_mult": 2.00,
                    "parabolic_from_pivot_min": 0.20,
                    "parabolic_above_10w_min": 0.15,
                    "late_stage_range_mult": 1.50,
                    "late_stage_volume_mult": 1.50,
                },
                exploration_mode="template_expansion",
                tags=("liquidity", "quality"),
            ),
            IdeaTemplate(
                family="superstock",
                template_id="superstock_breakout_focused",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="More aggressive breakout capture with looser price band and stronger volume confirmation.",
                reason_selected="Tests whether stronger breakout confirmation improves follow-through.",
                config={
                    "max_positions": 8,
                    "price_min": 3.0,
                    "price_max": 20.0,
                    "min_dollar_volume": 500_000.0,
                    "above_52w_low_mult": 1.25,
                    "near_52w_high_mult": 0.75,
                    "rs_rank_26w_min": 0.60,
                    "rs_rank_52w_min": 0.70,
                    "base_depth_max": 0.80,
                    "weekly_range_median_max": 0.22,
                    "weekly_volatility_max": 0.15,
                    "volume_dryup_max": 1.00,
                    "vix_hard_cap": 40.0,
                    "vix_ma_multiplier": 1.50,
                    "breakout_extension_max": 0.15,
                    "daily_volume_expansion_mult": 1.25,
                    "daily_dollar_volume_expansion_mult": 1.25,
                    "parabolic_from_pivot_min": 0.30,
                    "parabolic_above_10w_min": 0.25,
                    "late_stage_range_mult": 2.00,
                    "late_stage_volume_mult": 2.50,
                },
                exploration_mode="template_expansion",
                tags=("breakout", "aggressive"),
            ),
            IdeaTemplate(
                family="superstock",
                template_id="superstock_volatility_conservative",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis="Lower volatility and tighter breakout extension to improve post-entry stability.",
                reason_selected="Explores whether cleaner volatility structure improves breakout durability.",
                config={
                    "max_positions": 3,
                    "price_min": 5.0,
                    "price_max": 15.0,
                    "min_dollar_volume": 1_000_000.0,
                    "above_52w_low_mult": 1.25,
                    "near_52w_high_mult": 0.75,
                    "rs_rank_26w_min": 0.70,
                    "rs_rank_52w_min": 0.80,
                    "base_depth_max": 0.40,
                    "weekly_range_median_max": 0.15,
                    "weekly_volatility_max": 0.10,
                    "volume_dryup_max": 0.80,
                    "vix_hard_cap": 30.0,
                    "vix_ma_multiplier": 1.25,
                    "breakout_extension_max": 0.08,
                    "daily_volume_expansion_mult": 1.50,
                    "daily_dollar_volume_expansion_mult": 1.50,
                    "parabolic_from_pivot_min": 0.20,
                    "parabolic_above_10w_min": 0.15,
                    "late_stage_range_mult": 1.50,
                    "late_stage_volume_mult": 2.00,
                },
                exploration_mode="template_expansion",
                tags=("volatility", "risk-control"),
            ),
            IdeaTemplate(
                family="superstock",
                template_id="superstock_momentum_hybrid",
                strategy_type="classical",
                source_type="cross_family_hybrid",
                hypothesis="Superstock with more momentum-like tighter follow-through and quicker exit pressure.",
                reason_selected="Borrowed from momentum-style ranking discipline to widen search.",
                config={
                    "max_positions": 5,
                    "price_min": 5.0,
                    "price_max": 20.0,
                    "min_dollar_volume": 1_000_000.0,
                    "above_52w_low_mult": 1.25,
                    "near_52w_high_mult": 0.75,
                    "rs_rank_26w_min": 0.80,
                    "rs_rank_52w_min": 0.80,
                    "base_depth_max": 0.60,
                    "weekly_range_median_max": 0.18,
                    "weekly_volatility_max": 0.12,
                    "volume_dryup_max": 0.90,
                    "vix_hard_cap": 35.0,
                    "vix_ma_multiplier": 1.25,
                    "breakout_extension_max": 0.10,
                    "daily_volume_expansion_mult": 2.00,
                    "daily_dollar_volume_expansion_mult": 2.00,
                    "parabolic_from_pivot_min": 0.25,
                    "parabolic_above_10w_min": 0.20,
                    "late_stage_range_mult": 1.75,
                    "late_stage_volume_mult": 2.50,
                },
                exploration_mode="cross_family_hybrid",
                source_family="momentum",
                tags=("momentum", "hybrid"),
            ),
        ],
        "ml_ranker": [
            IdeaTemplate(
                family="ml_ranker",
                template_id="ml_ranker_ridge_trend_volume",
                strategy_type="ml",
                source_type="model_based",
                hypothesis="Cross-sectional ridge ranker with trend and volume features.",
                reason_selected="CPU-light ML baseline for broad ranking research.",
                config={
                    "model_type": "ridge",
                    "lookback_days": 252,
                    "horizon_days": 10,
                    "rebalance_days": 5,
                    "top_pct": 0.03,
                    "max_positions": 10,
                    "feature_set": "trend_volume",
                    "allow_short": False,
                    "use_vix_gate": True,
                    "use_fear_greed_gate": False,
                },
                exploration_mode="template_expansion",
                tags=("ml", "ridge", "ranking"),
            ),
            IdeaTemplate(
                family="ml_ranker",
                template_id="ml_ranker_hgb_full",
                strategy_type="ml",
                source_type="model_based",
                hypothesis="HistGradientBoosting ranker with a richer feature set and wider horizon.",
                reason_selected="Tests whether nonlinear trees outperform the linear baseline.",
                config={
                    "model_type": "hgb",
                    "lookback_days": 504,
                    "horizon_days": 20,
                    "rebalance_days": 10,
                    "top_pct": 0.05,
                    "max_positions": 20,
                    "feature_set": "full",
                    "allow_short": False,
                    "use_vix_gate": True,
                    "use_fear_greed_gate": True,
                },
                exploration_mode="template_expansion",
                tags=("ml", "boosting", "ranking"),
            ),
            IdeaTemplate(
                family="ml_ranker",
                template_id="ml_ranker_short_horizon",
                strategy_type="ml",
                source_type="model_based",
                hypothesis="Short-horizon ranker emphasizing fast rotation and tighter selection.",
                reason_selected="Explores whether shorter prediction horizons reduce lag.",
                config={
                    "model_type": "ridge",
                    "lookback_days": 126,
                    "horizon_days": 5,
                    "rebalance_days": 5,
                    "top_pct": 0.025,
                    "max_positions": 5,
                    "feature_set": "trend",
                    "allow_short": False,
                    "use_vix_gate": False,
                    "use_fear_greed_gate": False,
                },
                exploration_mode="template_expansion",
                tags=("ml", "short-horizon"),
            ),
        ],
        "rl_bandit": [
            IdeaTemplate(
                family="rl_bandit",
                template_id="rl_bandit_ucb_balanced",
                strategy_type="rl",
                source_type="policy_learning",
                hypothesis="UCB bandit choosing among momentum, breakout, and cash arms.",
                reason_selected="Practical CPU-only RL-lite baseline for regime-aware allocation.",
                config={
                    "policy_type": "ucb",
                    "lookback_days": 252,
                    "rebalance_days": 5,
                    "epsilon": 0.10,
                    "ucb_bonus": 1.0,
                    "max_positions": 5,
                    "momentum_top_pct": 0.03,
                    "superstock_top_pct": 0.03,
                    "use_vix_gate": True,
                    "use_fear_greed_gate": True,
                },
                exploration_mode="template_expansion",
                tags=("rl", "ucb", "bandit"),
            ),
            IdeaTemplate(
                family="rl_bandit",
                template_id="rl_bandit_epsilon_aggressive",
                strategy_type="rl",
                source_type="policy_learning",
                hypothesis="Epsilon-greedy bandit with more exploratory allocation switching.",
                reason_selected="Tests whether extra policy exploration helps in unstable regimes.",
                config={
                    "policy_type": "epsilon_greedy",
                    "lookback_days": 126,
                    "rebalance_days": 10,
                    "epsilon": 0.20,
                    "ucb_bonus": 0.5,
                    "max_positions": 8,
                    "momentum_top_pct": 0.05,
                    "superstock_top_pct": 0.05,
                    "use_vix_gate": True,
                    "use_fear_greed_gate": False,
                },
                exploration_mode="template_expansion",
                tags=("rl", "epsilon", "bandit"),
            ),
        ],
    }


def list_idea_families() -> list[str]:
    return sorted(_templates())


def list_idea_templates(family: str) -> list[IdeaTemplate]:
    normalized = family.strip().lower()
    return list(_templates().get(normalized, []))


def materialize_template(family: str, template: IdeaTemplate) -> dict[str, Any]:
    normalized = family.strip().lower()
    if template.family != normalized:
        raise ValueError(f"Template '{template.template_id}' is not for family '{family}'.")
    merged = {**get_family_default_config(family), **template.config}
    return normalize_experiment_config(family, merged)


def build_template_payload(template: IdeaTemplate, family: str) -> dict[str, Any]:
    normalized = materialize_template(family, template)
    return {
        "config": normalized,
        "metadata": {
            "strategy_type": template.strategy_type,
            "source_type": template.source_type,
            "template_id": template.template_id,
            "hypothesis": template.hypothesis,
            "reason_selected": template.reason_selected,
            "exploration_mode": template.exploration_mode,
            "source_family": template.source_family,
            "tags": list(template.tags),
        },
    }


def expand_template_candidates(
    family: str,
    *,
    limit: int,
    seed: int,
    allow_cross_family: bool = True,
) -> list[dict[str, Any]]:
    templates = list_idea_templates(family)
    if not allow_cross_family:
        templates = [template for template in templates if template.source_type != "cross_family_hybrid"]
    if not templates:
        return []

    import random

    rng = random.Random(seed)
    ordered = templates[:]
    rng.shuffle(ordered)
    payloads: list[dict[str, Any]] = []
    for template in ordered[:limit]:
        payloads.append(build_template_payload(template, family))
    return payloads


def load_external_idea_seeds(enabled: bool = False) -> list[dict[str, Any]]:
    return [] if not enabled else []

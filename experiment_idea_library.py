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
        "amihud_illiquidity_premium": [
            IdeaTemplate(
                family="amihud_illiquidity_premium",
                template_id="amihud_illiquidity_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Monthly Amihud illiquidity premium: rank tradable stocks by 60-day "
                    "average abs(return) / dollar volume, apply $5M avg dollar-volume, "
                    "$10 price, top-500 dollar-volume, 3.5% daily-volatility, and "
                    "20% OHLC execution-sanity guards, and hold the top 10 equally weighted "
                    "on a two-month rebalance."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Edge source is compensated illiquidity "
                    "within a tradable universe, not momentum, breakout, or superstock behavior."
                ),
                config={
                    "lookback_days": 60,
                    "rebalance_days": 42,
                    "max_positions": 10,
                    "min_dollar_volume": 5_000_000.0,
                    "price_min": 10.0,
                    "max_dollar_volume_rank": 500,
                    "max_daily_volatility": 0.035,
                    "max_abs_open_gap": 0.20,
                },
                exploration_mode="template_expansion",
                tags=("amihud", "illiquidity", "retired", "debug-only"),
            ),
            IdeaTemplate(
                family="amihud_illiquidity_premium",
                template_id="amihud_illiquidity_quality_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Illiquidity premium with volatility quality guard: monthly Amihud rank "
                    "among stocks above $2M avg dollar volume and below 3.5% daily volatility."
                ),
                reason_selected=(
                    "Keeps the illiquidity signal while testing whether a simple risk-quality "
                    "guard reduces cost and drawdown sensitivity."
                ),
                config={
                    "lookback_days": 60,
                    "rebalance_days": 21,
                    "max_positions": 20,
                    "min_dollar_volume": 2_000_000.0,
                    "price_min": 5.0,
                    "max_dollar_volume_rank": None,
                    "max_daily_volatility": 0.035,
                    "max_abs_open_gap": 0.20,
                },
                exploration_mode="template_expansion",
                tags=("amihud", "illiquidity", "retired", "debug-only"),
            ),
            IdeaTemplate(
                family="amihud_illiquidity_premium",
                template_id="amihud_illiquidity_largecap_safe_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Large-cap-safe illiquidity: rank by Amihud only within the top 250 stocks "
                    "by average dollar volume, with $5M liquidity and $10 price floors."
                ),
                reason_selected=(
                    "Explicitly tests whether an illiquidity premium remains in highly tradable "
                    "large-cap names after conservative cost controls."
                ),
                config={
                    "lookback_days": 90,
                    "rebalance_days": 21,
                    "max_positions": 10,
                    "min_dollar_volume": 5_000_000.0,
                    "price_min": 10.0,
                    "max_dollar_volume_rank": 250,
                    "max_daily_volatility": 0.035,
                    "max_abs_open_gap": 0.15,
                },
                exploration_mode="template_expansion",
                tags=("amihud", "illiquidity", "retired", "debug-only"),
            ),
        ],
        "breadth_timing_overlay": [
            IdeaTemplate(
                family="breadth_timing_overlay",
                template_id="breadth_timing_overlay_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Market-level breadth timing: long top-100 liquid stocks when >55% of "
                    "universe is above 200-day MA; cash when <45%. Hysteresis prevents whipsaw."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Edge source is market participation rate, "
                    "not stock return rank. Fully distinct from momentum and superstock."
                ),
                config={
                    "breadth_window": 200,
                    "entry_threshold": 0.55,
                    "exit_threshold": 0.45,
                    "top_n": 100,
                },
                exploration_mode="template_expansion",
                tags=("breadth", "market-timing", "orthogonal"),
            ),
            IdeaTemplate(
                family="breadth_timing_overlay",
                template_id="breadth_timing_fast_signal",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Faster breadth signal using 100-day MA with wider stock selection. "
                    "Tests whether shorter breadth window improves regime detection speed."
                ),
                reason_selected=(
                    "Explores shorter breadth lookback for earlier regime transitions."
                ),
                config={
                    "breadth_window": 100,
                    "entry_threshold": 0.55,
                    "exit_threshold": 0.45,
                    "top_n": 200,
                },
                exploration_mode="template_expansion",
                tags=("breadth", "market-timing", "fast"),
            ),
            IdeaTemplate(
                family="breadth_timing_overlay",
                template_id="breadth_timing_selective",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Stricter breadth gate (>60% above MA to enter, <40% to exit) with "
                    "concentrated top-50 position. Tests whether higher conviction reduces whipsaw."
                ),
                reason_selected=(
                    "Tighter entry/exit thresholds may improve regime quality at cost of fewer signals."
                ),
                config={
                    "breadth_window": 200,
                    "entry_threshold": 0.60,
                    "exit_threshold": 0.40,
                    "top_n": 50,
                },
                exploration_mode="template_expansion",
                tags=("breadth", "market-timing", "selective"),
            ),
        ],
        "mean_reversion": [
            IdeaTemplate(
                family="mean_reversion",
                template_id="mr_sector_neutral_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Sector-relative mean reversion: long 30 stocks with the worst 5-day "
                    "return relative to their GICS sector. Liquidity-filtered (>$1M avg daily "
                    "dollar vol). VIX gate at 40. Rebalances every 5 days."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Edge source is short-horizon idiosyncratic "
                    "overreaction reversal — explicitly trades against return rank within sectors. "
                    "Not momentum, not superstock."
                ),
                config={
                    "lookback_days": 5,
                    "rebal_days": 5,
                    "max_positions": 30,
                    "min_dollar_volume": 1_000_000.0,
                    "vix_gate": 40.0,
                },
                exploration_mode="template_expansion",
                tags=("mean-reversion", "sector-neutral", "orthogonal"),
            ),
            IdeaTemplate(
                family="mean_reversion",
                template_id="mr_sector_neutral_fast",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Faster mean reversion using 3-day return dislocation and 3-day hold. "
                    "Tests whether very short-term sector overreaction is tradeable after costs."
                ),
                reason_selected=(
                    "Probes the cost boundary of mean reversion — shorter horizon = stronger signal, "
                    "higher turnover."
                ),
                config={
                    "lookback_days": 3,
                    "rebal_days": 3,
                    "max_positions": 20,
                    "min_dollar_volume": 2_000_000.0,
                    "vix_gate": 35.0,
                },
                exploration_mode="template_expansion",
                tags=("mean-reversion", "fast", "high-liquidity"),
            ),
            IdeaTemplate(
                family="mean_reversion",
                template_id="mr_sector_neutral_broad",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Broader 10-day lookback with 50 positions. Tests whether medium-horizon "
                    "sector dislocation (not noise, not momentum) captures a reversal premium."
                ),
                reason_selected=(
                    "10-day horizon reduces noise and turnover; 50 positions improves diversification."
                ),
                config={
                    "lookback_days": 10,
                    "rebal_days": 10,
                    "max_positions": 50,
                    "min_dollar_volume": 500_000.0,
                    "vix_gate": 40.0,
                },
                exploration_mode="template_expansion",
                tags=("mean-reversion", "medium-horizon", "diversified"),
            ),
        ],
        "fear_greed_contrarian": [
            IdeaTemplate(
                family="fear_greed_contrarian",
                template_id="fear_greed_contrarian_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Fear & Greed contrarian: enter a 75% gross liquid-stock basket when "
                    "Fear & Greed is <=25, exit to cash when it reaches >=55. Rebalance monthly."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Edge source is sentiment capitulation and "
                    "mean reversion in risk appetite, not return momentum or superstock screening."
                ),
                config={
                    "fear_entry": 25.0,
                    "greed_exit": 55.0,
                    "confirmation_days": 1,
                    "rebalance_days": 21,
                    "max_positions": 25,
                    "min_dollar_volume": 2_000_000.0,
                    "exposure": 0.75,
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "stock-basket", "retired", "debug-only"),
            ),
            IdeaTemplate(
                family="fear_greed_contrarian",
                template_id="fear_greed_contrarian_defensive_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Defensive sentiment contrarian: require confirmed extreme fear <=20, "
                    "hold 50% gross across 50 liquid stocks, and exit when sentiment normalizes."
                ),
                reason_selected=(
                    "Tests the conservative version of the same sentiment edge with lower exposure "
                    "and broader diversification."
                ),
                config={
                    "fear_entry": 20.0,
                    "greed_exit": 50.0,
                    "confirmation_days": 2,
                    "rebalance_days": 21,
                    "max_positions": 50,
                    "min_dollar_volume": 5_000_000.0,
                    "exposure": 0.50,
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "stock-basket", "retired", "debug-only"),
            ),
            IdeaTemplate(
                family="fear_greed_contrarian",
                template_id="fear_greed_contrarian_aggressive_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Aggressive sentiment contrarian: enter earlier when Fear & Greed <=35, "
                    "hold 100% gross across 15 liquid stocks, and exit only at greed >=70."
                ),
                reason_selected=(
                    "Tests whether a wider sentiment contrarian regime captures more rebound while "
                    "remaining distinct from stock momentum ranks."
                ),
                config={
                    "fear_entry": 35.0,
                    "greed_exit": 70.0,
                    "confirmation_days": 1,
                    "rebalance_days": 10,
                    "max_positions": 15,
                    "min_dollar_volume": 1_000_000.0,
                    "exposure": 1.00,
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "stock-basket", "retired", "debug-only"),
            ),
        ],
        "fear_greed_contrarian_overlay": [
            IdeaTemplate(
                family="fear_greed_contrarian_overlay",
                template_id="fear_greed_contrarian_overlay_v2",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Fear & Greed overlay v2: enter 75% SPY exposure when sentiment "
                    "falls to <=25, exit to cash at >=55. No stock ranking or basket "
                    "selection; regime transitions drive allocation."
                ),
                reason_selected=(
                    "Promoted from diagnostics: Fear & Greed worked cleaner as a low-turnover "
                    "market exposure overlay than as a stock-selection family."
                ),
                config={
                    "entry_threshold": 25.0,
                    "exit_threshold": 55.0,
                    "exposure": 0.75,
                    "confirmation_days": 1,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "overlay", "v2", "redesign-only"),
            ),
            IdeaTemplate(
                family="fear_greed_contrarian_overlay",
                template_id="fear_greed_contrarian_overlay_defensive_v2",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Defensive Fear & Greed overlay: require confirmed extreme fear <=20 "
                    "and hold 50% SPY exposure until sentiment normalizes at >=50."
                ),
                reason_selected=(
                    "Tests lower exposure and confirmation while preserving the diagnostic "
                    "finding that this should be an overlay, not a stock selector."
                ),
                config={
                    "entry_threshold": 20.0,
                    "exit_threshold": 50.0,
                    "exposure": 0.50,
                    "confirmation_days": 2,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "overlay", "defensive", "v2", "redesign-only"),
            ),
            IdeaTemplate(
                family="fear_greed_contrarian_overlay",
                template_id="fear_greed_contrarian_overlay_partial_v2",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Partial Fear & Greed overlay: enter earlier at <=35 with 50% SPY "
                    "exposure and exit at greed >=70. Keeps risk low while testing wider "
                    "sentiment regimes."
                ),
                reason_selected=(
                    "Limited v2 exploration of threshold breadth and partial exposure only; "
                    "not an optimization pass."
                ),
                config={
                    "entry_threshold": 35.0,
                    "exit_threshold": 70.0,
                    "exposure": 0.50,
                    "confirmation_days": 1,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("fear-greed", "sentiment", "overlay", "partial", "v2", "redesign-only"),
            ),
        ],
        "momentum_fear_greed_overlay": [
            IdeaTemplate(
                family="momentum_fear_greed_overlay",
                template_id="momentum_fear_greed_overlay_partial_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Momentum plus Fear & Greed partial overlay: preserve momentum "
                    "stock selection, but scale sleeve exposure to 75% during fear "
                    "and greed regimes."
                ),
                reason_selected=(
                    "Tests whether sentiment improves an existing strong sleeve as "
                    "an exposure modifier rather than standalone alpha."
                ),
                config={
                    "entry_threshold": 25.0,
                    "exit_threshold": 55.0,
                    "greed_threshold": 75.0,
                    "fear_exposure": 0.75,
                    "normal_exposure": 1.00,
                    "greed_exposure": 0.75,
                    "confirmation_days": 2,
                },
                exploration_mode="template_expansion",
                tags=("momentum", "fear-greed", "sleeve-overlay", "redesign-only"),
            ),
            IdeaTemplate(
                family="momentum_fear_greed_overlay",
                template_id="momentum_fear_greed_overlay_defensive_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Defensive momentum plus Fear & Greed overlay: require confirmed "
                    "extreme fear and cut exposure to 50% in fear or greed regimes."
                ),
                reason_selected=(
                    "First-test risk-control variant with slower transitions and no "
                    "change to momentum ranking."
                ),
                config={
                    "entry_threshold": 20.0,
                    "exit_threshold": 50.0,
                    "greed_threshold": 75.0,
                    "fear_exposure": 0.50,
                    "normal_exposure": 1.00,
                    "greed_exposure": 0.50,
                    "confirmation_days": 3,
                },
                exploration_mode="template_expansion",
                tags=("momentum", "fear-greed", "sleeve-overlay", "defensive", "redesign-only"),
            ),
            IdeaTemplate(
                family="momentum_fear_greed_overlay",
                template_id="momentum_fear_greed_overlay_asymmetric_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Asymmetric momentum plus Fear & Greed overlay: keep full exposure "
                    "after fear capitulation, but reduce exposure during greed."
                ),
                reason_selected=(
                    "Tests whether the useful signal is mainly avoiding euphoric regimes "
                    "instead of buying fear as a standalone sleeve."
                ),
                config={
                    "entry_threshold": 25.0,
                    "exit_threshold": 55.0,
                    "greed_threshold": 75.0,
                    "fear_exposure": 1.00,
                    "normal_exposure": 1.00,
                    "greed_exposure": 0.50,
                    "confirmation_days": 2,
                },
                exploration_mode="template_expansion",
                tags=("momentum", "fear-greed", "sleeve-overlay", "asymmetric", "redesign-only"),
            ),
        ],
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
        "vix_regime_sector_tilt": [
            IdeaTemplate(
                family="vix_regime_sector_tilt",
                template_id="vix_regime_sector_tilt_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "VIX regime sector rotation: cyclicals (Tech/Discretionary/Industrials) "
                    "when VIX < 18, defensives (Utilities/Staples/Health Care) when VIX > 28, "
                    "all sectors equal-weight in between. VIX smoothed 5-day. "
                    "Edge source: volatility regime, not return rank."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Stock selection is driven by VIX level + "
                    "GICS sector tag only — no return rank, no quality filter. "
                    "Fully distinct from momentum and superstock."
                ),
                config={
                    "low_vix_threshold": 18.0,
                    "high_vix_threshold": 28.0,
                    "vix_smooth_days": 5,
                    "cyclical_sectors_mode": "narrow",
                },
                exploration_mode="template_expansion",
                tags=("vix-regime", "sector-rotation", "orthogonal"),
            ),
            IdeaTemplate(
                family="vix_regime_sector_tilt",
                template_id="vix_regime_broad_cyclicals",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Broad cyclicals mode: includes Financials and Communication Services "
                    "in the risk-on basket. Tests whether wider cyclical exposure improves "
                    "low-VIX returns."
                ),
                reason_selected=(
                    "Tests whether adding Financials/Comm Services to cyclicals improves coverage."
                ),
                config={
                    "low_vix_threshold": 18.0,
                    "high_vix_threshold": 28.0,
                    "vix_smooth_days": 5,
                    "cyclical_sectors_mode": "broad",
                },
                exploration_mode="template_expansion",
                tags=("vix-regime", "sector-rotation", "broad-cyclicals"),
            ),
            IdeaTemplate(
                family="vix_regime_sector_tilt",
                template_id="vix_regime_tight_bands",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Tighter VIX bands (15/25) with 10-day smoothing. Tests whether "
                    "more sensitive regime detection with slower signal improves stability."
                ),
                reason_selected=(
                    "Tighter thresholds detect regime shifts earlier; smoothing prevents whipsaw."
                ),
                config={
                    "low_vix_threshold": 15.0,
                    "high_vix_threshold": 25.0,
                    "vix_smooth_days": 10,
                    "cyclical_sectors_mode": "narrow",
                },
                exploration_mode="template_expansion",
                tags=("vix-regime", "sector-rotation", "tight-bands"),
            ),
        ],
        "sector_breadth_overlay": [
            IdeaTemplate(
                family="sector_breadth_overlay",
                template_id="sector_breadth_overlay_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Sector breadth overlay: use 100-day sector participation to choose "
                    "between 75% SPY risk-on, 50% selective-on SPY exposure, "
                    "35% defensive SPY exposure, or cash. No stock return ranking."
                ),
                reason_selected=(
                    "First-test orthogonal overlay. Edge source is sector internal breadth "
                    "and participation, not momentum or superstock stock selection."
                ),
                config={
                    "breadth_window": 100,
                    "sector_on_threshold": 0.55,
                    "sector_off_threshold": 0.45,
                    "broad_sector_fraction": 0.55,
                    "risk_on_exposure": 0.75,
                    "selective_exposure": 0.50,
                    "defensive_exposure": 0.35,
                    "min_selective_sectors": 3,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("sector-breadth", "overlay", "redesign-only", "v1"),
            ),
            IdeaTemplate(
                family="sector_breadth_overlay",
                template_id="sector_breadth_overlay_defensive_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Defensive sector breadth overlay: require broader sector confirmation "
                    "for SPY risk-on and keep lower selective/defensive SPY exposure."
                ),
                reason_selected=(
                    "Tests a conservative version of sector participation timing without "
                    "expanding into stock picking."
                ),
                config={
                    "breadth_window": 150,
                    "sector_on_threshold": 0.60,
                    "sector_off_threshold": 0.45,
                    "broad_sector_fraction": 0.65,
                    "risk_on_exposure": 0.50,
                    "selective_exposure": 0.35,
                    "defensive_exposure": 0.25,
                    "min_selective_sectors": 4,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("sector-breadth", "overlay", "defensive", "redesign-only", "v1"),
            ),
            IdeaTemplate(
                family="sector_breadth_overlay",
                template_id="sector_breadth_overlay_partial_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Partial sector breadth overlay: shorter breadth window with earlier "
                    "sector-on threshold and partial exposure. Tests responsiveness while "
                    "keeping overlay construction simple."
                ),
                reason_selected=(
                    "Limited first-test variation around sector breadth responsiveness; "
                    "not a parameter optimization pass."
                ),
                config={
                    "breadth_window": 80,
                    "sector_on_threshold": 0.50,
                    "sector_off_threshold": 0.40,
                    "broad_sector_fraction": 0.50,
                    "risk_on_exposure": 0.50,
                    "selective_exposure": 0.50,
                    "defensive_exposure": 0.35,
                    "min_selective_sectors": 2,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("sector-breadth", "overlay", "partial", "redesign-only", "v1"),
            ),
        ],
        "volatility_compression_expansion": [
            IdeaTemplate(
                family="volatility_compression_expansion",
                template_id="volatility_compression_expansion_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Volatility compression/expansion overlay: enter 50% SPY exposure "
                    "when an upside range expansion follows recent ATR compression; exit "
                    "on downside expansion or volatility re-expansion. OHLCV-only, no stock ranking."
                ),
                reason_selected=(
                    "First-test orthogonal prototype. Edge source is volatility state transition "
                    "after compression, not momentum rank or superstock breakout screening."
                ),
                config={
                    "compression_window": 20,
                    "baseline_window": 100,
                    "breakout_window": 20,
                    "compression_ratio": 0.70,
                    "expansion_mult": 1.20,
                    "exit_expansion_mult": 1.50,
                    "exposure": 0.50,
                    "compression_lookback": 10,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("volatility-compression", "expansion", "overlay", "retire-for-now", "v1"),
            ),
            IdeaTemplate(
                family="volatility_compression_expansion",
                template_id="volatility_compression_expansion_defensive_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Defensive compression/expansion overlay: require tighter compression and "
                    "larger expansion trigger, then hold only 35% SPY exposure."
                ),
                reason_selected=(
                    "Tests a lower-risk first-test version while keeping the same simple "
                    "OHLCV-only overlay construction."
                ),
                config={
                    "compression_window": 20,
                    "baseline_window": 100,
                    "breakout_window": 30,
                    "compression_ratio": 0.60,
                    "expansion_mult": 1.40,
                    "exit_expansion_mult": 1.30,
                    "exposure": 0.35,
                    "compression_lookback": 10,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("volatility-compression", "expansion", "overlay", "defensive", "retire-for-now", "v1"),
            ),
            IdeaTemplate(
                family="volatility_compression_expansion",
                template_id="volatility_compression_expansion_partial_v1",
                strategy_type="classical",
                source_type="template_expansion",
                hypothesis=(
                    "Partial compression/expansion overlay: shorter compression window and "
                    "partial 50% SPY exposure to test a more responsive volatility transition."
                ),
                reason_selected=(
                    "Limited first-test responsiveness variant; not a broad parameter search."
                ),
                config={
                    "compression_window": 15,
                    "baseline_window": 80,
                    "breakout_window": 15,
                    "compression_ratio": 0.80,
                    "expansion_mult": 1.10,
                    "exit_expansion_mult": 1.50,
                    "exposure": 0.50,
                    "compression_lookback": 5,
                    "market_symbol": "SPY",
                },
                exploration_mode="template_expansion",
                tags=("volatility-compression", "expansion", "overlay", "partial", "retire-for-now", "v1"),
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
    return sorted(
        family
        for family in _templates()
        if list_idea_templates(family)
    )


def list_idea_templates(family: str, *, include_retired: bool = False) -> list[IdeaTemplate]:
    normalized = family.strip().lower()
    templates = list(_templates().get(normalized, []))
    if include_retired:
        return templates
    return [
        template
        for template in templates
        if not ({"retired", "debug-only", "redesign-only", "retire-for-now"} & set(template.tags))
    ]


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

import logging

import pandas as pd

from .screening import ScreeningResult, combine_masks
from .superstock_trend import build_superstock_trend_template
from .superstock_weekly import build_superstock_weekly_features, to_daily_feature_map, SuperstockWeeklyFeatures

LOGGER = logging.getLogger(__name__)


def build_liquidity_and_price_mask(data: dict) -> ScreeningResult:
    close = data["close"]
    volume = data["volume"]

    avg_dollar_volume_20d = (close * volume).rolling(20, min_periods=20).mean()
    history_260d = close.rolling(260, min_periods=260).count() >= 260

    rule_masks = {
        "price_in_superstock_band": (close >= 5.0) & (close <= 15.0),
        "min_dollar_volume": avg_dollar_volume_20d >= 1_000_000.0,
        "has_min_daily_history": history_260d,
    }

    return ScreeningResult(
        eligible=combine_masks(rule_masks),
        rule_masks=rule_masks,
        diagnostics={"avg_dollar_volume_20d": avg_dollar_volume_20d},
    )


def build_base_building_mask(
    data: dict,
    weekly: SuperstockWeeklyFeatures,
) -> ScreeningResult:
    aligned = to_daily_feature_map(weekly, data["close"].index)

    base_depth = aligned["base_high_26w"] / aligned["base_low_26w"] - 1.0
    has_min_weekly_history = aligned["dist_from_52w_high"].notna() & aligned["dist_from_52w_low"].notna()
    volume_ratio = aligned["weekly_volume_ratio_26w"]

    rule_masks = {
        "has_min_weekly_history": has_min_weekly_history,
        "has_26w_base_window": aligned["base_high_26w"].notna() & aligned["base_low_26w"].notna(),
        "base_depth_not_too_wide": base_depth <= 0.60,
        "weekly_range_not_exploding": aligned["range_pct_median_26w"] <= 0.18,
        "weekly_volatility_not_exploding": aligned["weekly_volatility_26w"] <= 0.12,
        "volume_dryup_present": volume_ratio <= 0.80,
    }

    return ScreeningResult(
        eligible=combine_masks(rule_masks),
        rule_masks=rule_masks,
        diagnostics={
            "base_depth_26w": base_depth,
            "weekly_volume_ratio_26w": volume_ratio,
            "range_pct_median_26w": aligned["range_pct_median_26w"],
            "weekly_volatility_26w": aligned["weekly_volatility_26w"],
        },
    )


def build_superstock_screen(
    data: dict,
    weekly: SuperstockWeeklyFeatures | None = None,
    fundamentals: dict | None = None,
) -> ScreeningResult:
    if fundamentals:
        LOGGER.info("Optional fundamentals were provided but are ignored in Superstock v1 screening.")

    weekly = weekly or build_superstock_weekly_features(data)
    aligned = to_daily_feature_map(weekly, data["close"].index)

    trend = build_superstock_trend_template(data, weekly)
    liquidity_price = build_liquidity_and_price_mask(data)
    base = build_base_building_mask(data, weekly)

    benchmark_ma_10w = aligned["benchmark_weekly_close"].rolling(10, min_periods=10).mean()
    benchmark_not_broken = aligned["benchmark_weekly_close"] > benchmark_ma_10w

    if aligned["vix_weekly_close"].isna().all():
        vix_not_extreme = pd.Series(True, index=data["close"].index, name="vix_not_extreme")
        vix_not_extreme = pd.DataFrame(
            {col: vix_not_extreme for col in data["close"].columns},
            index=data["close"].index,
        )
    else:
        vix_threshold = pd.concat(
            [
                aligned["vix_ma_10w"] * 1.25,
                pd.Series(35.0, index=data["close"].index, name="hard_vix_cap"),
            ],
            axis=1,
        ).max(axis=1)
        vix_not_extreme_series = aligned["vix_weekly_close"] <= vix_threshold
        vix_not_extreme = pd.DataFrame(
            {col: vix_not_extreme_series for col in data["close"].columns},
            index=data["close"].index,
        )

    benchmark_not_broken_df = pd.DataFrame(
        {col: benchmark_not_broken for col in data["close"].columns},
        index=data["close"].index,
    )

    combined_rule_masks = {
        **trend.rule_masks,
        **liquidity_price.rule_masks,
        **base.rule_masks,
        "benchmark_not_broken": benchmark_not_broken_df,
        "vix_not_extreme": vix_not_extreme,
    }

    eligible_rule_names = [
        *trend.rule_masks.keys(),
        "price_in_superstock_band",
        "min_dollar_volume",
        "has_min_daily_history",
        "has_min_weekly_history",
        "has_26w_base_window",
        "base_depth_not_too_wide",
        "weekly_range_not_exploding",
        "weekly_volatility_not_exploding",
        "volume_dryup_present",
        "vix_not_extreme",
    ]
    eligible = combine_masks({name: combined_rule_masks[name] for name in eligible_rule_names})

    diagnostics: dict[str, pd.DataFrame | pd.Series | str | None] = {
        **trend.diagnostics,
        **liquidity_price.diagnostics,
        **base.diagnostics,
        "benchmark_not_broken": benchmark_not_broken_df,
        "vix_weekly_close": aligned["vix_weekly_close"],
        "vix_ma_10w": aligned["vix_ma_10w"],
        "fundamentals_used": None,
    }

    return ScreeningResult(
        eligible=eligible,
        rule_masks=combined_rule_masks,
        diagnostics=diagnostics,
    )

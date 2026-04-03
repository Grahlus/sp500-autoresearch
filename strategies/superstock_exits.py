import pandas as pd

from .screening import ScreeningResult, combine_masks
from .superstock_entries import build_superstock_breakout_entries
from .superstock_weekly import SuperstockWeeklyFeatures, build_superstock_weekly_features, to_daily_feature_map


def build_superstock_exit_signals(
    data: dict,
    weekly: SuperstockWeeklyFeatures | None = None,
    entries: ScreeningResult | None = None,
    config: dict | None = None,
) -> ScreeningResult:
    config = config or {}
    close = data["close"]
    low = data["low"]

    weekly = weekly or build_superstock_weekly_features(data)
    entries = entries or build_superstock_breakout_entries(data, weekly=weekly, config=config)
    weekly_daily = to_daily_feature_map(weekly, close.index)

    weekly_close = weekly_daily["weekly_close"]
    ma_10w = weekly_daily["ma_10w"]
    ma_30w = weekly_daily["ma_30w"]
    range_pct_1w = weekly_daily["range_pct_1w"]
    range_pct_median_26w = weekly_daily["range_pct_median_26w"]
    weekly_volume_ratio_26w = weekly_daily["weekly_volume_ratio_26w"]
    breakout_pct_above_pivot = entries.diagnostics["breakout_pct_above_pivot"]
    entry_event = entries.diagnostics["entry_event"]

    close_below_20d_low_excl_today = close < low.shift(1).rolling(20, min_periods=20).min()
    extension_above_10w = close / ma_10w - 1.0

    weekly_close_below_10w = weekly_close < ma_10w
    prior_weekly_close_not_below_10w = weekly_close.shift(1) >= ma_10w.shift(1)
    support_break_exit = weekly_close_below_10w & prior_weekly_close_not_below_10w

    parabolic_extension_from_pivot = breakout_pct_above_pivot >= float(config.get("parabolic_from_pivot_min", 0.25))
    parabolic_extension_above_10w = extension_above_10w >= float(config.get("parabolic_above_10w_min", 0.20))
    parabolic_extension_exit = parabolic_extension_from_pivot | parabolic_extension_above_10w

    weekly_range_expansion_vs_base = range_pct_1w >= (range_pct_median_26w * float(config.get("late_stage_range_mult", 1.75)))
    weekly_volume_surge_vs_base = weekly_volume_ratio_26w >= float(config.get("late_stage_volume_mult", 2.0))
    late_stage_hv_exit = (
        weekly_range_expansion_vs_base
        & weekly_volume_surge_vs_base
        & close_below_20d_low_excl_today
    )

    same_day_entry = entry_event.fillna(False).astype(bool)

    rule_masks = {
        "weekly_close_below_10w": weekly_close_below_10w,
        "prior_weekly_close_not_below_10w": prior_weekly_close_not_below_10w,
        "support_break_exit": support_break_exit,
        "weekly_close_below_30w": weekly_close < ma_30w,
        "parabolic_extension_from_pivot": parabolic_extension_from_pivot,
        "parabolic_extension_above_10w": parabolic_extension_above_10w,
        "parabolic_extension_exit": parabolic_extension_exit,
        "weekly_range_expansion_vs_base": weekly_range_expansion_vs_base,
        "weekly_volume_surge_vs_base": weekly_volume_surge_vs_base,
        "close_breaks_below_20d_low_excl_today": close_below_20d_low_excl_today,
        "late_stage_hv_exit": late_stage_hv_exit,
        "not_same_day_as_entry": ~same_day_entry,
    }

    eligible = combine_masks(
        {
            "any_exit_family": support_break_exit | parabolic_extension_exit | late_stage_hv_exit,
            "not_same_day_as_entry": rule_masks["not_same_day_as_entry"],
        }
    )

    diagnostics: dict[str, pd.DataFrame | pd.Series | str | None] = {
        "benchmark_source": entries.diagnostics.get("benchmark_source"),
        "benchmark_weekly_close": entries.diagnostics.get("benchmark_weekly_close"),
        "spy_weekly_close": entries.diagnostics.get("spy_weekly_close"),
        "entry_event": entry_event,
        "weekly_close": weekly_close,
        "ma_10w": ma_10w,
        "ma_30w": ma_30w,
        "breakout_pct_above_pivot": breakout_pct_above_pivot,
        "extension_above_10w": extension_above_10w,
        "range_pct_1w": range_pct_1w,
        "range_pct_median_26w": range_pct_median_26w,
        "weekly_volume_ratio_26w": weekly_volume_ratio_26w,
        "close_20d_low_excl_today": low.shift(1).rolling(20, min_periods=20).min(),
        "exit_reason_support": support_break_exit,
        "exit_reason_parabolic": parabolic_extension_exit,
        "exit_reason_late_stage_hv": late_stage_hv_exit,
        "exit_reason_any": support_break_exit | parabolic_extension_exit | late_stage_hv_exit,
    }

    return ScreeningResult(eligible=eligible, rule_masks=rule_masks, diagnostics=diagnostics)


def build_superstock_position_mask(
    entries: ScreeningResult,
    exits: ScreeningResult,
    execution_lag_days: int = 1,
) -> pd.DataFrame:
    if execution_lag_days != 1:
        raise ValueError("Superstock v1 only supports execution_lag_days=1.")

    columns = entries.eligible.columns
    index = entries.eligible.index
    positions = pd.DataFrame(False, index=index, columns=columns)
    current = pd.Series(False, index=columns)

    for i, dt in enumerate(index):
        if i > 0:
            prior_dt = index[i - execution_lag_days]
            exit_mask = exits.eligible.loc[prior_dt].fillna(False).astype(bool)
            current.loc[exit_mask[exit_mask].index] = False

            entry_mask = entries.eligible.loc[prior_dt].fillna(False).astype(bool)
            current.loc[entry_mask[entry_mask].index] = True
        positions.loc[dt] = current

    return positions

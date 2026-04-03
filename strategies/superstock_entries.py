import pandas as pd

from .screening import ScreeningResult, combine_masks
from .superstock_screen import build_superstock_screen
from .superstock_weekly import SuperstockWeeklyFeatures, build_superstock_weekly_features, to_daily_feature_map


def build_superstock_breakout_entries(
    data: dict,
    weekly: SuperstockWeeklyFeatures | None = None,
    screen: ScreeningResult | None = None,
    config: dict | None = None,
) -> ScreeningResult:
    config = config or {}
    close = data["close"]
    high = data["high"]
    volume = data["volume"]

    weekly = weekly or build_superstock_weekly_features(data)
    screen = screen or build_superstock_screen(data, weekly=weekly, config=config)
    weekly_daily = to_daily_feature_map(weekly, close.index)

    base_high_26w = weekly_daily["base_high_26w"]
    rolling_20d_high_excl_today = high.shift(1).rolling(20, min_periods=20).max()
    avg_volume_20d_excl_today = volume.shift(1).rolling(20, min_periods=20).mean()
    avg_dollar_volume_20d_excl_today = (close * volume).shift(1).rolling(20, min_periods=20).mean()
    breakout_pct_above_pivot = close / base_high_26w - 1.0

    rule_masks = {
        "screen_eligible": screen.eligible,
        "close_breaks_above_base_high": close > base_high_26w,
        "prior_close_not_already_above_base_high": close.shift(1) <= base_high_26w.shift(1),
        "close_above_recent_daily_high": close > rolling_20d_high_excl_today,
        "not_too_extended_from_pivot": breakout_pct_above_pivot <= float(config.get("breakout_extension_max", 0.10)),
        "daily_volume_expansion": volume >= (avg_volume_20d_excl_today * float(config.get("daily_volume_expansion_mult", 1.5))),
        "daily_dollar_volume_expansion": (close * volume) >= (
            avg_dollar_volume_20d_excl_today * float(config.get("daily_dollar_volume_expansion_mult", 1.5))
        ),
    }

    eligible = combine_masks(rule_masks)
    diagnostics: dict[str, pd.DataFrame | pd.Series | str | None] = {
        "benchmark_source": screen.diagnostics.get("benchmark_source"),
        "benchmark_weekly_close": screen.diagnostics.get("benchmark_weekly_close"),
        "spy_weekly_close": screen.diagnostics.get("spy_weekly_close"),
        "entry_event": eligible,
        "base_high_26w": base_high_26w,
        "rolling_20d_high_excl_today": rolling_20d_high_excl_today,
        "avg_volume_20d_excl_today": avg_volume_20d_excl_today,
        "avg_dollar_volume_20d_excl_today": avg_dollar_volume_20d_excl_today,
        "breakout_pct_above_pivot": breakout_pct_above_pivot,
        "rs_rank_26w": screen.diagnostics.get("rs_rank_26w"),
        "rs_rank_52w": screen.diagnostics.get("rs_rank_52w"),
    }

    return ScreeningResult(eligible=eligible, rule_masks=rule_masks, diagnostics=diagnostics)


def build_superstock_entry_weights(
    data: dict,
    entries: ScreeningResult | None = None,
    max_positions: int = 5,
    weight_mode: str = "equal",
    config: dict | None = None,
) -> pd.DataFrame:
    if weight_mode != "equal":
        raise ValueError("Only equal weighting is supported in Superstock v1 entries.")

    entries = entries or build_superstock_breakout_entries(data, config=config)
    close = data["close"]
    rs_rank_26w = entries.diagnostics.get("rs_rank_26w")
    if not isinstance(rs_rank_26w, pd.DataFrame):
        raise ValueError("Entry diagnostics must include rs_rank_26w for ranking.")

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    max_positions = max(int(max_positions), 1)

    for dt in close.index:
        candidates = entries.eligible.loc[dt]
        if not candidates.any():
            continue
        ranked = rs_rank_26w.loc[dt, candidates[candidates].index].sort_values(ascending=False)
        selected = ranked.head(max_positions).index
        weights.loc[dt, selected] = 1.0 / len(selected)

    return weights

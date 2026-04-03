from dataclasses import dataclass, fields

import pandas as pd

from .weekly import WeeklyBars, align_completed_weekly_to_daily, build_weekly_bars, rolling_weekly_mean, rolling_weekly_std


@dataclass(frozen=True)
class SuperstockWeeklyFeatures:
    weekly_open: pd.DataFrame
    weekly_high: pd.DataFrame
    weekly_low: pd.DataFrame
    weekly_close: pd.DataFrame
    weekly_volume: pd.DataFrame
    week_end_mask: pd.Series
    week_key_by_day: pd.Series
    ma_10w: pd.DataFrame
    ma_30w: pd.DataFrame
    avg_volume_10w: pd.DataFrame
    base_high_26w: pd.DataFrame
    base_low_26w: pd.DataFrame
    range_pct_1w: pd.DataFrame
    range_pct_median_26w: pd.DataFrame
    weekly_volatility_26w: pd.DataFrame
    weekly_volume_ratio_26w: pd.DataFrame
    dist_from_52w_high: pd.DataFrame
    dist_from_52w_low: pd.DataFrame
    rs_ratio_spy: pd.DataFrame
    rs_rank_26w: pd.DataFrame
    rs_rank_52w: pd.DataFrame
    spy_weekly_close: pd.Series
    vix_weekly_close: pd.Series
    vix_ma_10w: pd.Series


def _build_series_weekly(values: pd.Series, name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError(f"Expected a single-column DataFrame for {name}.")
        values = values.iloc[:, 0]
    frame = values.to_frame(name)
    weekly = build_weekly_bars(frame, frame, frame, frame, frame)
    return weekly.close[name]


def _resolve_benchmark_close(close: pd.DataFrame, data: dict) -> pd.Series:
    if "SPY" in close.columns:
        return close["SPY"]
    for key in ("spy_close", "benchmark_close"):
        series = data.get(key)
        if isinstance(series, pd.Series):
            return series.reindex(close.index).ffill()
    return close.ffill().pct_change().mean(axis=1).fillna(0.0).add(1.0).cumprod()


def build_superstock_weekly_features(data: dict) -> SuperstockWeeklyFeatures:
    close = data["close"]
    open_ = data["open"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    vix = data["vix"]

    spy_close = _resolve_benchmark_close(close, data)
    weekly: WeeklyBars = build_weekly_bars(open_, high, low, close, volume)

    weekly_open = weekly.open
    weekly_high = weekly.high
    weekly_low = weekly.low
    weekly_close = weekly.close
    weekly_volume = weekly.volume

    weekly_return = weekly_close.pct_change()
    ma_10w = rolling_weekly_mean(weekly_close, 10)
    ma_30w = rolling_weekly_mean(weekly_close, 30)
    avg_volume_10w = rolling_weekly_mean(weekly_volume.shift(1), 10)

    base_high_26w = weekly_high.shift(1).rolling(26, min_periods=26).max()
    base_low_26w = weekly_low.shift(1).rolling(26, min_periods=26).min()

    range_pct_1w = (weekly_high - weekly_low) / weekly_close.replace(0, pd.NA)
    range_pct_median_26w = range_pct_1w.shift(1).rolling(26, min_periods=26).median()
    weekly_volatility_26w = rolling_weekly_std(weekly_return.shift(1), 26)

    volume_baseline_26w = weekly_volume.shift(1).rolling(26, min_periods=26).median()
    weekly_volume_ratio_26w = weekly_volume / volume_baseline_26w.replace(0, pd.NA)

    high_52w = weekly_high.rolling(52, min_periods=52).max()
    low_52w = weekly_low.rolling(52, min_periods=52).min()
    dist_from_52w_high = (weekly_close / high_52w) - 1.0
    dist_from_52w_low = (weekly_close / low_52w) - 1.0

    spy_weekly_close = _build_series_weekly(spy_close, "spy_proxy")
    rs_ratio_spy = weekly_close.div(spy_weekly_close, axis=0)

    ret_26w = weekly_close / weekly_close.shift(26) - 1.0
    ret_52w = weekly_close / weekly_close.shift(52) - 1.0
    rs_rank_26w = ret_26w.rank(axis=1, pct=True)
    rs_rank_52w = ret_52w.rank(axis=1, pct=True)

    vix_weekly_close = _build_series_weekly(vix, "vix")
    vix_ma_10w = rolling_weekly_mean(vix_weekly_close, 10)

    return SuperstockWeeklyFeatures(
        weekly_open=weekly_open,
        weekly_high=weekly_high,
        weekly_low=weekly_low,
        weekly_close=weekly_close,
        weekly_volume=weekly_volume,
        week_end_mask=weekly.week_end_mask,
        week_key_by_day=weekly.week_key_by_day,
        ma_10w=ma_10w,
        ma_30w=ma_30w,
        avg_volume_10w=avg_volume_10w,
        base_high_26w=base_high_26w,
        base_low_26w=base_low_26w,
        range_pct_1w=range_pct_1w,
        range_pct_median_26w=range_pct_median_26w,
        weekly_volatility_26w=weekly_volatility_26w,
        weekly_volume_ratio_26w=weekly_volume_ratio_26w,
        dist_from_52w_high=dist_from_52w_high,
        dist_from_52w_low=dist_from_52w_low,
        rs_ratio_spy=rs_ratio_spy,
        rs_rank_26w=rs_rank_26w,
        rs_rank_52w=rs_rank_52w,
        spy_weekly_close=spy_weekly_close,
        vix_weekly_close=vix_weekly_close,
        vix_ma_10w=vix_ma_10w,
    )


def to_daily_feature_map(features: SuperstockWeeklyFeatures, daily_index: pd.Index) -> dict[str, pd.DataFrame | pd.Series]:
    aligned: dict[str, pd.DataFrame | pd.Series] = {}
    for field in fields(features):
        value = getattr(features, field.name)
        if field.name in {"week_end_mask", "week_key_by_day"}:
            aligned[field.name] = value
        else:
            aligned[field.name] = align_completed_weekly_to_daily(
                value, daily_index, features.weekly_close.index
            )
    return aligned

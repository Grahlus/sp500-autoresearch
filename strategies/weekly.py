from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WeeklyBars:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    week_end_dates: pd.Index
    week_end_mask: pd.Series
    week_key_by_day: pd.Series


def build_weekly_bars(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> WeeklyBars:
    if not (
        open_.index.equals(high.index)
        and open_.index.equals(low.index)
        and open_.index.equals(close.index)
        and open_.index.equals(volume.index)
    ):
        raise ValueError("OHLCV inputs must share the same daily index.")

    if not (
        open_.columns.equals(high.columns)
        and open_.columns.equals(low.columns)
        and open_.columns.equals(close.columns)
        and open_.columns.equals(volume.columns)
    ):
        raise ValueError("OHLCV inputs must share the same ticker columns.")

    week_period = open_.index.to_period("W-FRI")
    week_key_by_day = pd.Series(week_period.astype(str), index=open_.index, name="week_key")

    weekly_open = open_.groupby(week_period).first()
    weekly_high = high.groupby(week_period).max()
    weekly_low = low.groupby(week_period).min()
    weekly_close = close.groupby(week_period).last()
    weekly_volume = volume.groupby(week_period).sum(min_count=1)

    week_end_dates = close.groupby(week_period).apply(lambda frame: frame.index[-1])
    week_end_dates.index = weekly_close.index

    weekly_open.index = week_end_dates.values
    weekly_high.index = week_end_dates.values
    weekly_low.index = week_end_dates.values
    weekly_close.index = week_end_dates.values
    weekly_volume.index = week_end_dates.values

    week_end_mask = open_.index.isin(week_end_dates.values)
    week_end_mask = pd.Series(week_end_mask, index=open_.index, name="is_week_end")

    return WeeklyBars(
        open=weekly_open.sort_index(),
        high=weekly_high.sort_index(),
        low=weekly_low.sort_index(),
        close=weekly_close.sort_index(),
        volume=weekly_volume.sort_index(),
        week_end_dates=pd.Index(week_end_dates.values, name="week_end"),
        week_end_mask=week_end_mask,
        week_key_by_day=week_key_by_day,
    )


def align_completed_weekly_to_daily(
    weekly_feature: pd.DataFrame | pd.Series,
    daily_index: pd.Index,
    week_end_dates: pd.Index,
) -> pd.DataFrame | pd.Series:
    if not weekly_feature.index.isin(week_end_dates).all():
        raise ValueError("Weekly feature index must be the actual week-end trading dates.")
    return weekly_feature.reindex(daily_index).ffill()


def rolling_weekly_mean(
    weekly_values: pd.DataFrame | pd.Series,
    weeks: int,
    min_periods: int | None = None,
) -> pd.DataFrame | pd.Series:
    return weekly_values.rolling(weeks, min_periods=min_periods or weeks).mean()


def rolling_weekly_std(
    weekly_values: pd.DataFrame | pd.Series,
    weeks: int,
    min_periods: int | None = None,
) -> pd.DataFrame | pd.Series:
    return weekly_values.rolling(weeks, min_periods=min_periods or weeks).std()

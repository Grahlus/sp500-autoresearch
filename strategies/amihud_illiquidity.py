from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyFamily

METRIC = "sharpe"
HYPOTHESIS = (
    "amihud_illiquidity_v1: Monthly long-only selection of tradable stocks ranked "
    "by rolling Amihud illiquidity, abs(return) / dollar volume, with explicit "
    "liquidity and price floors. Edge source: illiquidity premium, not momentum "
    "or superstock breakout behavior."
)


def amihud_fragility_penalty(data: dict, lookback_days: int = 60) -> pd.DataFrame:
    """Recoverable hook for future use as a liquidity-fragility penalty, not a standalone family."""
    close = data["close"]
    volume = data["volume"]
    daily_return = close.pct_change()
    dollar_volume = (close * volume).replace(0.0, np.nan)
    amihud = (daily_return.abs() / dollar_volume).rolling(
        lookback_days,
        min_periods=lookback_days // 2,
    ).mean()
    return amihud.rank(axis=1, pct=True, ascending=True)


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    open_ = data["open"]
    volume = data["volume"]

    lookback_days = int(config.get("lookback_days", 60))
    rebalance_days = int(config.get("rebalance_days", 42))
    max_positions = int(config.get("max_positions", 10))
    min_dollar_volume = float(config.get("min_dollar_volume", 5_000_000.0))
    price_min = float(config.get("price_min", 10.0))
    max_dollar_volume_rank = config.get("max_dollar_volume_rank", 500)
    max_dollar_volume_rank = int(max_dollar_volume_rank) if max_dollar_volume_rank is not None else None
    max_daily_volatility = config.get("max_daily_volatility", 0.035)
    max_daily_volatility = float(max_daily_volatility) if max_daily_volatility is not None else None
    max_abs_open_gap = config.get("max_abs_open_gap", 0.20)
    max_abs_open_gap = float(max_abs_open_gap) if max_abs_open_gap is not None else None

    daily_return = close.pct_change()
    open_gap = (open_ / close.shift(1) - 1.0).abs()
    intraday_move = (close / open_ - 1.0).abs()
    dollar_volume = (close * volume).replace(0.0, np.nan)
    avg_dollar_volume = dollar_volume.rolling(20, min_periods=5).mean()
    amihud = (daily_return.abs() / dollar_volume).rolling(lookback_days, min_periods=lookback_days // 2).mean()
    daily_volatility = daily_return.rolling(lookback_days, min_periods=lookback_days // 2).std()
    daily_execution_move = open_gap.where(open_gap >= intraday_move, intraday_move)
    max_execution_move = daily_execution_move.rolling(
        lookback_days,
        min_periods=lookback_days // 2,
    ).max()
    dollar_volume_rank = avg_dollar_volume.rank(axis=1, ascending=False, method="first")

    n_dates = len(close.index)
    n_tickers = len(close.columns)
    weights = np.zeros((n_dates, n_tickers), dtype=np.float64)
    current = np.zeros(n_tickers, dtype=np.float64)

    for i in range(lookback_days // 2, n_dates):
        is_rebalance = (i - lookback_days // 2) % rebalance_days == 0
        if is_rebalance:
            eligible = (
                close.iloc[i].notna()
                & (close.iloc[i] >= price_min)
                & (avg_dollar_volume.iloc[i] >= min_dollar_volume)
                & amihud.iloc[i].notna()
            )
            if max_dollar_volume_rank is not None:
                eligible &= dollar_volume_rank.iloc[i] <= max_dollar_volume_rank
            if max_daily_volatility is not None:
                eligible &= daily_volatility.iloc[i] <= max_daily_volatility
            if max_abs_open_gap is not None:
                eligible &= max_execution_move.iloc[i] <= max_abs_open_gap

            ranked = amihud.iloc[i][eligible].dropna().nlargest(max_positions)
            current = np.zeros(n_tickers, dtype=np.float64)
            if len(ranked) >= max(3, min(max_positions, 10) // 2):
                current[close.columns.get_indexer(ranked.index)] = 1.0 / len(ranked)

        weights[i] = current

    return pd.DataFrame(weights, index=close.index, columns=close.columns)


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="amihud_illiquidity_premium",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

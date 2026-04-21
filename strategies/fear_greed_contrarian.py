from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyFamily

METRIC = "sharpe"
HYPOTHESIS = (
    "fear_greed_contrarian_v1: Long liquid broad baskets only after Fear & Greed "
    "falls into fear/extreme-fear regimes; exit to cash as sentiment normalizes or "
    "reaches greed. Edge source: sentiment-regime contrarian behavior, not stock "
    "momentum or superstock screening."
)


def _fear_greed_series(data: dict) -> pd.Series:
    fear_greed = data["fear_greed"]
    if isinstance(fear_greed, pd.DataFrame):
        if "fg_value" in fear_greed.columns:
            return fear_greed["fg_value"]
        return fear_greed.iloc[:, 0]
    return fear_greed


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    volume = data["volume"]
    fear_greed = _fear_greed_series(data).reindex(close.index).ffill()

    fear_entry = float(config.get("fear_entry", 25.0))
    greed_exit = float(config.get("greed_exit", 55.0))
    confirmation_days = int(config.get("confirmation_days", 1))
    rebalance_days = int(config.get("rebalance_days", 21))
    max_positions = int(config.get("max_positions", 25))
    min_dollar_volume = float(config.get("min_dollar_volume", 2_000_000.0))
    exposure = float(config.get("exposure", 0.75))

    dollar_volume = (close * volume).rolling(20, min_periods=5).mean()
    liquid = dollar_volume >= min_dollar_volume
    liquidity_rank = dollar_volume.rank(axis=1, ascending=False, method="first")

    fear_signal = (fear_greed <= fear_entry).fillna(False)
    greed_signal = (fear_greed >= greed_exit).fillna(False)
    if confirmation_days > 1:
        fear_signal = fear_signal.rolling(confirmation_days, min_periods=confirmation_days).sum() >= confirmation_days
        greed_signal = greed_signal.rolling(confirmation_days, min_periods=confirmation_days).sum() >= confirmation_days

    n_dates = len(close.index)
    n_tickers = len(close.columns)
    weights = np.zeros((n_dates, n_tickers), dtype=np.float64)
    current = np.zeros(n_tickers, dtype=np.float64)

    in_market = False
    for i in range(n_dates):
        fg_value = fear_greed.iloc[i]
        if np.isnan(fg_value):
            in_market = False
            current = np.zeros(n_tickers, dtype=np.float64)
        elif not in_market and bool(fear_signal.iloc[i]):
            in_market = True
        elif in_market and bool(greed_signal.iloc[i]):
            in_market = False
            current = np.zeros(n_tickers, dtype=np.float64)

        is_rebalance = i == 0 or i % rebalance_days == 0
        if in_market and is_rebalance:
            eligible = (liquid.iloc[i] & close.iloc[i].notna()).fillna(False)
            ranked = liquidity_rank.iloc[i][eligible].dropna().nsmallest(max_positions)
            current = np.zeros(n_tickers, dtype=np.float64)
            if len(ranked) > 0:
                current[close.columns.get_indexer(ranked.index)] = exposure / len(ranked)

        weights[i] = current if in_market else 0.0

    return pd.DataFrame(weights, index=close.index, columns=close.columns)


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="fear_greed_contrarian",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

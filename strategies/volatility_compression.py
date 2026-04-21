from __future__ import annotations

import pandas as pd

from .base import StrategyFamily

METRIC = "sharpe"
HYPOTHESIS = (
    "volatility_compression_expansion_v1: OHLCV-only overlay that enters partial "
    "SPY exposure when an upside range expansion follows recent volatility "
    "compression; exits on downside expansion or volatility re-expansion. Edge "
    "source: compression/expansion regime, not momentum rank or superstock screening."
)


def _true_range_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range / close.replace(0.0, pd.NA)


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    high = data["high"]
    low = data["low"]

    market_symbol = str(config.get("market_symbol", "SPY"))
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if market_symbol not in close.columns:
        return weights

    compression_window = int(config.get("compression_window", 20))
    baseline_window = int(config.get("baseline_window", 100))
    breakout_window = int(config.get("breakout_window", 20))
    compression_ratio = float(config.get("compression_ratio", 0.70))
    expansion_mult = float(config.get("expansion_mult", 1.20))
    exit_expansion_mult = float(config.get("exit_expansion_mult", 1.50))
    exposure = float(config.get("exposure", 0.50))
    compression_lookback = int(config.get("compression_lookback", 10))

    px_close = close[market_symbol]
    px_high = high[market_symbol]
    px_low = low[market_symbol]

    tr_pct = _true_range_pct(px_high, px_low, px_close)
    atr_short = tr_pct.rolling(compression_window, min_periods=compression_window // 2).mean()
    atr_base = tr_pct.rolling(baseline_window, min_periods=baseline_window // 2).median()

    compressed = atr_short <= (atr_base * compression_ratio)
    recently_compressed = (
        compressed.rolling(compression_lookback, min_periods=1).max().fillna(False).astype(bool)
    )

    prior_high = px_high.rolling(breakout_window, min_periods=breakout_window // 2).max().shift(1)
    prior_low = px_low.rolling(breakout_window, min_periods=breakout_window // 2).min().shift(1)
    upside_expansion = (
        recently_compressed
        & (px_close > prior_high)
        & (tr_pct >= atr_short * expansion_mult)
    )
    downside_expansion = (
        (px_close < prior_low)
        | (tr_pct >= atr_short * exit_expansion_mult)
    )

    in_market = False
    market_col = weights.columns.get_loc(market_symbol)
    for i in range(len(weights.index)):
        if not in_market and bool(upside_expansion.iloc[i]):
            in_market = True
        elif in_market and bool(downside_expansion.iloc[i]):
            in_market = False

        if in_market:
            weights.iat[i, market_col] = exposure

    return weights


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="volatility_compression_expansion",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

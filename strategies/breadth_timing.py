from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyFamily

METRIC = "sharpe"
HYPOTHESIS = (
    "breadth_timing_overlay_v1: Long equal-weight top-N liquid stocks when market breadth "
    "(% stocks above 200d MA) exceeds entry threshold; cash otherwise. "
    "Hysteresis band prevents whipsaw. Edge source: market participation rate, not return rank."
)


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    volume = data["volume"]

    breadth_window = int(config.get("breadth_window", 200))
    entry_threshold = float(config.get("entry_threshold", 0.55))
    exit_threshold = float(config.get("exit_threshold", 0.45))
    top_n = int(config.get("top_n", 100))

    # Breadth = fraction of stocks currently trading above their N-day MA
    ma = close.rolling(breadth_window, min_periods=breadth_window // 2).mean()
    above_ma = (close > ma).fillna(False)
    breadth_num = above_ma.sum(axis=1).astype(float)
    breadth_den = (~close.isna()).sum(axis=1).astype(float).clip(lower=1)
    breadth = breadth_num / breadth_den
    breadth.iloc[: breadth_window // 2] = np.nan

    # Hysteresis state machine: enter market on strength, exit on weakness
    in_market = np.zeros(len(close), dtype=bool)
    state = False
    for i in range(len(breadth)):
        b = breadth.iloc[i]
        if np.isnan(b):
            state = False
        elif not state and b >= entry_threshold:
            state = True
        elif state and b <= exit_threshold:
            state = False
        in_market[i] = state

    in_market_s = pd.Series(in_market, index=close.index)

    # Select top_n stocks by 20-day average dollar volume (most liquid)
    avg_dvol = (close * volume).rolling(20, min_periods=5).mean().fillna(0.0)
    vol_rank = avg_dvol.rank(axis=1, ascending=False, method="first")
    selected_mask = (vol_rank <= top_n).astype(float)

    n_selected = selected_mask.sum(axis=1).clip(lower=1)
    w_base = selected_mask.div(n_selected, axis=0)

    # Zero out days not in market
    weights = w_base.multiply(in_market_s.astype(float), axis=0)
    return weights


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="breadth_timing_overlay",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

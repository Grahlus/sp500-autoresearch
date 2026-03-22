"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 151: Self-normalizing vol percentile filter.
Instead of vol_60 vs vol_240 ratio, use vol_60 vs its own rolling 25th percentile
over a 480-bar window. Enter when current vol is in the top 75% of recent vol history.
Hypothesis: percentile is more adaptive to regime vol changes than a fixed ratio.
Asymmetric: longs vol > 25th pct (top 75%), shorts vol > 40th pct (top 60%).
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    # Rolling percentile thresholds over 480-bar history
    vol_pct25 = vol_series.rolling(480).quantile(0.25).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct25)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

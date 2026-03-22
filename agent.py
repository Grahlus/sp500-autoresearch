"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 321: Median of HL2 EMA(380, 425, 470) slow ensemble — wider spread.
Hypothesis: exp_320 used narrow span range (400/425/450, ±25).
A wider range (380/425/470, ±45) might provide more meaningful averaging.
The median would be the same as 425 when all 3 are in agreement,
but diverges more on edge cases, potentially improving robustness.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow_380 = hl2.ewm(span=380, adjust=False).mean().values
    ema_slow_425 = hl2.ewm(span=425, adjust=False).mean().values
    ema_slow_470 = hl2.ewm(span=470, adjust=False).mean().values
    ema_slow = np.median([ema_slow_380, ema_slow_425, ema_slow_470], axis=0)

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    long_sig = ema_fast > ema_slow
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

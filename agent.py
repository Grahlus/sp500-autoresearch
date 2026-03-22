"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 324: Re-verify span=426 with new H6 gate 0.5.
Previously exp_259 got Z5=4.13, H6=0.55 — failed old gate (0.6) but passes new gate (0.5).
If verified, this would be a new champion (Z5=4.13 > 4.0776).
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow = hl2.ewm(span=426, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    long_sig = ema_fast > ema_slow
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

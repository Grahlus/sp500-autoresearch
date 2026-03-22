"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 108: VWAP EMA(3/480) + looser vol threshold (vol_60 > vol_240 * 0.8).
Raw EMA(3/480) = Z5=3.14 but H6=-0.23. Champion vol filter costs 1.67 Z5.
Hypothesis: A 20% looser vol threshold allows more of the Z5 quiet-trend entries
while still gating the H6 false-trend periods that kill returns.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=3, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values

    signals = np.where(
        (ema_fast > ema_slow) & (vol_short > vol_long * 0.8), 1, 0
    )
    return signals.astype(int)

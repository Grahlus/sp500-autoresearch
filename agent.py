"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 040: VWAP EMA(8/480) + volume rising (vol_60 > vol_240) dual confirmation.
Only long when VWAP trend AND volume increasing (institutional accumulation).
Thesis: rising volume in uptrend = real demand; volume decline = distribution.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Long when EMA(8/480) vwap uptrend AND recent volume above long-term average.
    vol_60 > vol_240 means 60-bar avg volume is above 240-bar avg volume.
    """
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=8, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    # Volume trend: rolling 60-bar mean > rolling 240-bar mean
    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values
    vol_rising = vol_short > vol_long

    trend_on = ema_fast > ema_slow

    signals = np.where(trend_on & vol_rising, 1, 0)
    return signals.astype(int)

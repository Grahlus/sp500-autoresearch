"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 120: VWAP EMA(6/480) bidirectional + vol*0.8.
Fast span sweep: EMA(3)=2.06, EMA(4)=2.35 (champion), EMA(5)=2.26, EMA(6)=?
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values
    vol_ok = vol_short > vol_long * 0.8

    long_sig = (ema_fast > ema_slow) & vol_ok
    short_sig = (ema_fast < ema_slow) & vol_ok

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

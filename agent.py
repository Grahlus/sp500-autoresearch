"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 114: VWAP EMA(3/480) bidirectional + vol_60>vol_240*0.8.
Long when EMA3>EMA480 + vol. Short when EMA3<EMA480 + vol.
Hypothesis: With vol=0.8x filter (previously tested only with long-only),
shorts in declining confirmed trends might add alpha,
especially in H6 which has more volatility than Z5.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=3, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values
    vol_ok = vol_short > vol_long * 0.8

    long_sig = (ema_fast > ema_slow) & vol_ok
    short_sig = (ema_fast < ema_slow) & vol_ok

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

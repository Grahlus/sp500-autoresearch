"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 139: Asymmetric vol — longs: vol*0.6 (very loose), shorts: vol*0.8 (current).
Key insight: long-only with 0.8x = Z5=1.61. Shorts add 0.97 Z5.
Hypothesis: Much looser vol for longs (enter more quiet uptrends) while
keeping quality shorts (0.8x). Z5 might improve by adding Z5 quiet-trend longs.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_long * 0.6)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_long * 0.8)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

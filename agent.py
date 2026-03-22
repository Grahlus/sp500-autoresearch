"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 320: Median of HL2 EMA(400, 425, 450) as slow line (ensemble).
Hypothesis: a single EMA span can overfit to one regime. The median of three spans
(below, at, and above the champion span) is more robust.
The median filters out extreme behavior from spans 400 and 450 while retaining
the central behavior of span 425. Should reduce regime-specific fitting.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow_400 = hl2.ewm(span=400, adjust=False).mean().values
    ema_slow_425 = hl2.ewm(span=425, adjust=False).mean().values
    ema_slow_450 = hl2.ewm(span=450, adjust=False).mean().values
    # Median ensemble: more robust than any single span
    ema_slow = np.median([ema_slow_400, ema_slow_425, ema_slow_450], axis=0)

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    long_sig = ema_fast > ema_slow
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

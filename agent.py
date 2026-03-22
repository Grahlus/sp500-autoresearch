"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 257: Longs vol-free + shorts vol>40th, HL2 EMA(425) slow.
420: Z5=3.99, H6=0.65 (champion, passes). 430: Z5=4.20, H6=0.46 (fails).
Testing 425 — the crossover point between passing and failing H6 gate.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow = hl2.ewm(span=425, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    long_sig = ema_fast > ema_slow
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

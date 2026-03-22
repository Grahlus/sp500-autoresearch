"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 185: Fast EMA on OHLC4, slow EMA on close.
OHLC4 fast captures bar activity (including open); close slow tracks pure price.
Hypothesis: divergence between OHLC4 trend and close-price trend provides
cleaner signal — OHLC4 fast reacts to all bar info, close slow filters out wicks.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values          # OHLC4 fast
    ema_slow = df["close"].ewm(span=420, adjust=False).mean().values  # close slow

    vol_series = df["volume"].rolling(60).mean()
    vol_pct30 = vol_series.rolling(480).quantile(0.30).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct30)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

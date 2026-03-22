"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 253: Longs vol-filter-free, shorts require vol > 40th pct.
Champion uses vol for both. HL2 EMA(420) slow line already filters noise for
longs. In NQ's uptrend-biased market, the EMA crossover alone might be
sufficient for longs — vol filter may cut valid quiet-trend long entries.
Shorts remain vol-gated to filter false shorts in the uptrend bias.
Hypothesis: removing vol requirement for longs captures more of the uptrend
alpha without adding significant noise.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow = hl2.ewm(span=420, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    # Longs: EMA crossover only (no vol filter)
    long_sig = ema_fast > ema_slow
    # Shorts: EMA crossover + vol > 40th pct
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

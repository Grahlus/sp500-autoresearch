"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 164: EMA(6/420) + vol pct longs>30th, shorts>40th.
Sweep slow span: EMA(6/360)→Z5=2.91, testing EMA(6/420).
Hypothesis: 420 might be a sweet spot between 360 (Z5=2.91) and 480 (Z5=2.82 with 25th/40th).
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=420, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct30 = vol_series.rolling(480).quantile(0.30).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct30)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 161: EMA(6/360) + vol percentile, longs>30th pct (between 25th and 35th).
Champion (exp_160): longs>35th → Z5=2.8635, H6=1.32.
With H6=1.32 headroom, try slightly looser longs (30th pct = top 70%).
Hypothesis: 30th pct might increase Z5 further while keeping H6 > 0.6.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=360, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct30 = vol_series.rolling(480).quantile(0.30).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct30)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

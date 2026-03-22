"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 160: EMA(6/360) + vol percentile, tighter longs>35th pct.
Exp_158: longs>25th → Z5=2.66, H6=1.15. Loosening (15th) hurt Z5.
Maybe tighter longs (35th pct = top 65%) improves quality for EMA(360)?
With EMA(360)'s strong H6 structure, we might be able to select
the best Z5 entries without hurting H6.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=6, adjust=False).mean().values
    ema_slow = vwap.ewm(span=360, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct35 = vol_series.rolling(480).quantile(0.35).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct35)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 214: OHLC4 EMA(6) fast vs HL2 = (high+low)/2 EMA(420) slow.
HL2 is the bar midpoint (range center), capturing intrabar extremes without
settlement (close) or opening (open) bias. Vol filter unchanged.
Hypothesis: HL2 represents structural support/resistance (range midpoint history)
rather than just settlement price. Different structural anchor for the slow line.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    ema_slow = hl2.ewm(span=420, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct30 = vol_series.rolling(480).quantile(0.30).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_cur > vol_pct30)
    short_sig = (ema_fast < ema_slow) & (vol_cur > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

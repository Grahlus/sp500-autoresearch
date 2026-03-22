"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 174: OHLC4 = (open+high+low+close)/4 as price metric instead of VWAP proxy.
VWAP proxy = (H+L+C)/3 excludes open. OHLC4 includes open which carries
overnight gap/session-start information. Structural difference.
EMA(6/420) + vol pct 30th/40th pct (champion setup).
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    # OHLC4 includes open unlike VWAP proxy (H+L+C)/3
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values
    ema_slow = ohlc4.ewm(span=420, adjust=False).mean().values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct30 = vol_series.rolling(480).quantile(0.30).values
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values

    vol_short = vol_series.values

    long_sig = (ema_fast > ema_slow) & (vol_short > vol_pct30)
    short_sig = (ema_fast < ema_slow) & (vol_short > vol_pct40)

    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return signals.astype(int)

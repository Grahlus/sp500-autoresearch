"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 047: VWAP EMA(5/480) + vol_60>vol_240 — re-tune fast span on correct data.
Previous best was EMA(8); test if faster EMA works better on properly deduplicated Z5.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long when VWAP EMA(5) > VWAP EMA(480) AND vol_60 > vol_240."""
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=5, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    vol_short = df["volume"].rolling(60).mean().values
    vol_long = df["volume"].rolling(240).mean().values

    signals = np.where(
        (ema_fast > ema_slow) & (vol_short > vol_long), 1, 0
    )
    return signals.astype(int)

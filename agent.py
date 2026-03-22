"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 037: EMA(8) vs EMA(480) both on VWAP proxy (typical price).
Consistent signal using (H+L+C)/3 for both fast and slow EMAs.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long when EMA(8) of vwap_proxy > EMA(480) of vwap_proxy."""
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=8, adjust=False).mean().values
    ema_slow = vwap.ewm(span=480, adjust=False).mean().values

    signals = np.where(ema_fast > ema_slow, 1, 0)
    return signals.astype(int)

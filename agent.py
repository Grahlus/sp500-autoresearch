"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 035: EMA(8) of close vs EMA(480) of VWAP proxy (typical price).
Slow EMA uses (H+L+C)/3 — more representative of each bar's true price.
Thesis: VWAP-based reference level is less noisy than close-based.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long when EMA(8) of close > EMA(480) of vwap_proxy."""
    ema_fast = df["close"].ewm(span=8, adjust=False).mean().values
    ema_slow = df["vwap_proxy"].ewm(span=480, adjust=False).mean().values

    signals = np.where(ema_fast > ema_slow, 1, 0)
    return signals.astype(int)

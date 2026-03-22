"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 019: EMA(8) vs EMA(480) long-only crossover.
Slightly faster fast EMA to test if entry timing improves.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long-only EMA crossover: long when ema_8 > ema_480, flat otherwise."""
    close = df["close"]
    ema_fast = close.ewm(span=8, adjust=False).mean().values
    ema_slow = close.ewm(span=480, adjust=False).mean().values

    signals = np.where(ema_fast > ema_slow, 1, 0)
    return signals.astype(int)

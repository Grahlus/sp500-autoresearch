"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 013: EMA(15) vs EMA(240) long-only crossover.
EMA responds faster to recent price → better timing than SMA crossover.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long-only EMA crossover: long when ema_15 > ema_240, flat otherwise."""
    close = df["close"]
    ema_fast = close.ewm(span=15, adjust=False).mean().values
    ema_slow = close.ewm(span=240, adjust=False).mean().values

    signals = np.where(ema_fast > ema_slow, 1, 0)
    return signals.astype(int)

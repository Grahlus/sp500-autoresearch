"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 017: EMA(10) vs EMA(480) long-only crossover.
Slower slow EMA (8hr) holds through more noise, fewer exits.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long-only EMA crossover: long when ema_10 > ema_480, flat otherwise."""
    close = df["close"]
    ema_fast = close.ewm(span=10, adjust=False).mean().values
    ema_slow = close.ewm(span=480, adjust=False).mean().values

    signals = np.where(ema_fast > ema_slow, 1, 0)
    return signals.astype(int)

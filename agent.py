"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 003: SMA(15) vs SMA(240) long-only crossover.
Wider MA gap → fewer false crossovers → less commission drag.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Long-only SMA crossover: long when sma_15 > sma_240, flat otherwise.
    """
    fast = df["sma_15"].values
    slow = df["sma_240"].values

    signals = np.where(fast > slow, 1, 0)
    return signals.astype(int)

"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 001: SMA(5) vs SMA(60) crossover trend-following.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """
    SMA crossover: long when sma_5 > sma_60, short when sma_5 < sma_60.
    """
    fast = df["sma_5"].values
    slow = df["sma_60"].values

    signals = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return signals.astype(int)

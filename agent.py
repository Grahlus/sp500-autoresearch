"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

The agent may:
  - Implement any strategy it wants (rules-based, ML, RL, hybrid)
  - Add/remove imports
  - Define a train(df) function if it needs to fit on training data
  - Change literally anything in here

The contract with prepare.py is simple:
  - get_signals(df) must exist
  - get_signals(df) must return a numpy array of length len(df) with values in {-1, 0, 1}
  - Everything must complete within the 15-minute wall clock budget

Current implementation: random baseline (Calmar ≈ 0). Replace entirely.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of NQ 1-min bars with pre-computed features,
    return a signal array: +1 (long), -1 (short), 0 (flat).

    Available columns (from prepare.add_basic_features):
      open, high, low, close, volume
      returns, log_ret, hl_range, vwap_proxy
      sma_5, sma_15, sma_60, sma_240
      roc_5, roc_15, roc_60, roc_240
      vol_5, vol_15, vol_60, vol_240
      rsi_14, atr_14
    """
    # BASELINE: random signals — agent should replace this entirely
    rng = np.random.default_rng(seed=42)
    signals = rng.choice([-1, 0, 1], size=len(df))
    return signals

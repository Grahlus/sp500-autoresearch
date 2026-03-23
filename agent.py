"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 388: Two-tier boundary test — DIP_MULT2=3.95.
Hypothesis: DIP_MULT2 sweep shows H6=0.608→0.602 going from 4.2→4.0. Each 0.1 ATR
decrease in DIP_MULT2 reduces H6 by ~0.003. At DIP_MULT2=3.95, expect H6~0.600 (still
passing gate). If true, Z5 should be ~4.27 (more Tier2 entries). This verifies whether
the H6/Z5 tradeoff is smooth (linear) or whether 3.95 suddenly crashes H6 below 0.6
(like a cliff edge). Critical boundary test.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_fast = ohlc4.ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    e380 = hl2.ewm(span=380, adjust=False).mean().values
    e425 = hl2.ewm(span=425, adjust=False).mean().values
    e470 = hl2.ewm(span=470, adjust=False).mean().values
    ema_slow = np.median([e380, e425, e470], axis=0)

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    bar_range = (df["high"] - df["low"]).values
    atr = pd.Series(bar_range).rolling(20, min_periods=1).mean().values
    close_arr = df["close"].values

    n = len(close_arr)
    signals = np.zeros(n, dtype=np.int32)
    position = 0
    dip_tier = 0

    DIP_MULT1 = 3.9
    LOOKBACK1 = 120
    STOP1 = 5.5
    DIP_MULT2 = 3.95
    LOOKBACK2 = 60
    STOP2 = 5.0
    EXIT_ABOVE_SLOW = 0.25

    for i in range(n):
        close = close_arr[i]
        slow = ema_slow[i]
        atr_val = atr[i]
        base_long = ema_fast[i] > slow
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev1 = ema_slow[max(0, i - LOOKBACK1)]
        slow_prev2 = ema_slow[max(0, i - LOOKBACK2)]
        tier1_ok = (slow > slow_prev1) and (close < slow - DIP_MULT1 * atr_val)
        tier2_ok = (slow > slow_prev2) and (close < slow - DIP_MULT2 * atr_val)
        dip_entry = (not base_long) and (not base_short) and (tier1_ok or tier2_ok)

        if position == 1:
            if dip_tier > 0:
                stop_mult = STOP1 if dip_tier == 1 else STOP2
                if base_long:
                    dip_tier = 0
                elif close >= slow + EXIT_ABOVE_SLOW * atr_val:
                    position = 0
                    dip_tier = 0
                elif close < slow - stop_mult * atr_val:
                    position = 0
                    dip_tier = 0
                elif base_short:
                    position = 0
                    dip_tier = 0
            else:
                if not base_long:
                    position = 0

        elif position == -1:
            if not base_short:
                position = 0

        if position == 0:
            if base_long:
                position = 1
                dip_tier = 0
            elif base_short:
                position = -1
            elif dip_entry:
                position = 1
                dip_tier = 1 if tier1_ok else 2

        signals[i] = position

    return signals

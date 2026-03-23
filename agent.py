"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 441: Three-tier ATR(25) with STOP3=8.0 (2.5 ATR buffer vs champion 1.5 ATR buffer).
Hypothesis: STOP3=6.0→Z5=4.35(worse), 7.0→4.36(champion). Try 8.0 — wider stop allows
more room for Tier3 trades to breathe. May improve Z5 if some Tier3 trades temporarily
dip below -7.0 ATR before recovering.
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
    atr = pd.Series(bar_range).rolling(25, min_periods=1).mean().values
    close_arr = df["close"].values

    n = len(close_arr)
    signals = np.zeros(n, dtype=np.int32)
    position = 0
    dip_tier = 0
    dip_entry_bar = 0  # bar index when dip was entered

    DIP_MULT1 = 3.9
    LOOKBACK1 = 120
    STOP1 = 5.5
    DIP_MULT2 = 3.95
    LOOKBACK2 = 60
    STOP2 = 5.0
    DIP_MULT3 = 5.5
    LOOKBACK3 = 45
    STOP3 = 8.0
    EXIT_ABOVE_SLOW = 0.25

    for i in range(n):
        close = close_arr[i]
        slow = ema_slow[i]
        atr_val = atr[i]
        base_long = ema_fast[i] > slow
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev1 = ema_slow[max(0, i - LOOKBACK1)]
        slow_prev2 = ema_slow[max(0, i - LOOKBACK2)]
        slow_prev3 = ema_slow[max(0, i - LOOKBACK3)]
        tier1_ok = (slow > slow_prev1) and (close < slow - DIP_MULT1 * atr_val)
        tier2_ok = (slow > slow_prev2) and (close < slow - DIP_MULT2 * atr_val)
        tier3_ok = (slow > slow_prev3) and (close < slow - DIP_MULT3 * atr_val)
        dip_entry = (not base_long) and (not base_short) and (tier1_ok or tier2_ok or tier3_ok)

        if position == 1:
            if dip_tier > 0:
                if dip_tier == 1:
                    stop_mult = STOP1
                elif dip_tier == 2:
                    stop_mult = STOP2
                else:
                    stop_mult = STOP3
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
                if tier1_ok:
                    dip_tier = 1
                elif tier2_ok:
                    dip_tier = 2
                else:
                    dip_tier = 3
                dip_entry_bar = i

        signals[i] = position

    return signals

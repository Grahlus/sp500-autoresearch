"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 698: RSI<40 hold condition. NEW CHAMPION!
Z5=5.2556, H6=0.8610. Hold when: roc_240>0.0005 OR roc_60>0.0002 OR rsi_14<40.
RSI sweep: <35→5.025/0.83, <38→5.11/0.85, <39→5.25/0.83, <40→5.2556(peak)/0.86, <42→5.10/0.83, <45→5.21/0.81.
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
    roc_240_arr = df["roc_240"].fillna(0).values
    roc_60_arr = df["roc_60"].fillna(0).values
    rsi_14_arr = df["rsi_14"].fillna(50).values

    n = len(close_arr)
    signals = np.zeros(n, dtype=np.int32)
    position = 0
    dip_tier = 0
    dip_entry_bar = 0  # bar index when dip was entered
    dip_entry_fast = 0.0  # fast EMA value at dip entry

    DIP_MULT1 = 3.9
    LOOKBACK1 = 130
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
        base_long_enter = ema_fast[i] > slow + 0.05 * atr_val
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev1 = ema_slow[max(0, i - LOOKBACK1)]
        slow_prev2 = ema_slow[max(0, i - LOOKBACK2)]
        slow_prev3 = ema_slow[max(0, i - LOOKBACK3)]
        tier1_ok = (slow > slow_prev1) and (close < slow - DIP_MULT1 * atr_val)
        tier2_ok = (slow > slow_prev2) and (close < slow - DIP_MULT2 * atr_val)
        tier3_ok = (slow > slow_prev3) and (close < slow - DIP_MULT3 * atr_val)
        fast_declining = (ema_fast[max(0, i - 5)] - ema_fast[i]) > 0.02 * atr_val
        dip_entry = (not base_long) and (not base_short) and fast_declining and (tier1_ok or tier2_ok or tier3_ok)

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
                elif (i - dip_entry_bar) > (80 if ema_fast[i] > dip_entry_fast else 60):
                    position = 0
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
                if (not base_long and roc_240_arr[i] <= 0.0005 and roc_60_arr[i] <= 0.0002 and rsi_14_arr[i] > 40) or base_short or (close < slow - 3.6 * atr_val):
                    position = 0

        elif position == -1:
            if not base_short:
                position = 0

        if position == 0:
            if base_long_enter:
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
                dip_entry_fast = ema_fast[i]

        signals[i] = position

    return signals

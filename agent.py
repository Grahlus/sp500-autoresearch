"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 1185: single EMA(425) instead of 3-span ensemble.
"""

import numpy as np
import pandas as pd


def get_signals(df: pd.DataFrame) -> np.ndarray:
    ohlc4_arr = ((df["open"] + df["high"] + df["low"] + df["close"]) / 4.0).values
    ema_fast = ((df["high"] + df["low"]) / 2.0).ewm(span=6, adjust=False).mean().values

    hl2 = (df["high"] + df["low"]) / 2.0
    slow_basis = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ema_slow = slow_basis.ewm(span=425, adjust=False).mean().values

    vol_realized = df["vol_240"].bfill()
    vol_realized_pct80 = vol_realized.rolling(480).quantile(0.85).values
    vol_realized_arr = vol_realized.values

    vol_series = df["volume"].rolling(60).mean()
    vol_pct40 = vol_series.rolling(480).quantile(0.40).values
    vol_cur = vol_series.values

    atr = df["atr_14"].bfill().values
    close_arr = df["close"].values
    roc_240_arr = df["roc_240"].fillna(0).values
    roc_60_arr = df["roc_60"].fillna(0).values
    roc_5_arr = df["roc_5"].fillna(0).values
    roc_15_arr = df["roc_15"].fillna(0).values
    rsi_14_arr = df["rsi_14"].fillna(50).values

    n = len(close_arr)
    signals = np.zeros(n, dtype=np.int32)
    position = 0
    dip_tier = 0
    dip_entry_bar = 0  # bar index when dip was entered
    dip_entry_fast = 0.0  # fast EMA value at dip entry

    DIP_MULT1 = 5.0
    LOOKBACK1 = 240
    STOP1 = 5.25
    DIP_MULT2 = 3.95
    LOOKBACK2 = 60
    STOP2 = 3.5
    DIP_MULT3 = 5.5
    LOOKBACK3 = 45
    STOP3 = 10.0
    EXIT_ABOVE_SLOW = 0.12
    EXIT_RSI = 0.19

    for i in range(n):
        close = close_arr[i]
        slow = ema_slow[i]
        atr_val = atr[i]
        base_long = ema_fast[i] > slow
        low_vol = vol_realized_arr[i] < vol_realized_pct80[i]
        slow_rising10 = slow > ema_slow[max(0, i - 10)]
        base_long_enter = ema_fast[i] > slow + 0.05 * atr_val and roc_240_arr[i] > 0.0001 and roc_60_arr[i] > 0.0025 and low_vol and slow_rising10
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev1 = ema_slow[max(0, i - LOOKBACK1)]
        slow_prev2 = ema_slow[max(0, i - LOOKBACK2)]
        slow_prev3 = ema_slow[max(0, i - LOOKBACK3)]
        tier1_ok = (slow > slow_prev1) and (close < slow - DIP_MULT1 * atr_val)
        tier2_ok = (slow > slow_prev2) and (close < slow - DIP_MULT2 * atr_val)
        tier3_ok = (slow > slow_prev3) and (close < slow - DIP_MULT3 * atr_val)
        fast_declining = (ema_fast[max(0, i - 3)] - ema_fast[i]) > 0.02 * atr_val
        rsi_dip_ok = (rsi_14_arr[i] < 32) and fast_declining and (not base_long) and (not base_short)
        dip_entry = (not base_long) and (not base_short) and fast_declining and (tier1_ok or tier2_ok or tier3_ok)
        dip_entry = dip_entry or rsi_dip_ok

        if position == 1:
            if dip_tier > 0:
                if dip_tier == 1:
                    stop_mult = STOP1
                elif dip_tier == 2:
                    stop_mult = STOP2
                elif dip_tier == 3:
                    stop_mult = STOP3
                else:
                    stop_mult = STOP1  # tier4 RSI: same stop as tier1
                if base_long and roc_60_arr[i] > 0.0003:
                    dip_tier = 0
                elif (i - dip_entry_bar) > (90 if ema_fast[i] > dip_entry_fast else 70):
                    position = 0
                    dip_tier = 0
                elif close >= slow + (EXIT_RSI if dip_tier == 4 else EXIT_ABOVE_SLOW) * atr_val:
                    position = 0
                    dip_tier = 0
                elif close < slow - stop_mult * atr_val:
                    position = 0
                    dip_tier = 0
                elif base_short:
                    position = 0
                    dip_tier = 0
            else:
                if (not base_long and roc_240_arr[i] <= 0.0003 and roc_60_arr[i] <= 0.0001 and roc_15_arr[i] <= 0.0001 and roc_5_arr[i] <= 0.0001 and rsi_14_arr[i] > 40) or base_short or (ohlc4_arr[i] < slow - 3.75 * atr_val):
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
                elif tier3_ok:
                    dip_tier = 3
                else:
                    dip_tier = 4  # RSI-only: tier4 with wider exit
                dip_entry_bar = i
                dip_entry_fast = ema_fast[i]

        signals[i] = position

    return signals

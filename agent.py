"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 386: Two-tier dip with tier-specific stops.
Tier1: DIP_MULT=3.9, LOOKBACK=120, STOP=5.5 ATR (stable uptrend, wider stop)
Tier2: DIP_MULT=4.0, LOOKBACK=60, STOP=5.0 ATR (recent reversal, tighter stop)
Hypothesis: Champion uses single stop=5.5 for both tiers. Tier2 entries (60-bar
rising, recently reversed) are riskier — tighter stop at 5.0 ATR should cut bad
H6 Tier2 trades earlier without affecting Tier1. This should improve H6 safety
buffer while minimally impacting Z5 (in Z5's bull market, Tier2 dips recover
quickly and rarely hit the 5.0 ATR stop). Net: better H6 with similar Z5.
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
    dip_tier = 0  # 0=no dip, 1=Tier1, 2=Tier2

    DIP_MULT1 = 3.9
    LOOKBACK1 = 120
    STOP1 = 5.5
    DIP_MULT2 = 4.0
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
                    dip_tier = 0  # smooth transition to champion long
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

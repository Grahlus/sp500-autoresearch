"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 383: Two-tier dip — Tier2 DIP_MULT=4.0.
Hypothesis: exp_382 champion uses Tier2=4.2. Try 4.0 — even lower threshold. Tier1
captures slow_rising_120 dips at 3.9 ATR. Tier2 at 4.0 would overlap significantly with
Tier1 (only difference: slow_rising_60 not slow_rising_120). New entries only come from
cases where slow_rising_60=True but slow_rising_120=False (slow EMA recently started
rising, was flat/declining 120 bars ago but rising 60 bars ago). This is a "recently
turned upward" category. DIP_MULT2=4.0 for these cases is a test of this subset.

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
    is_dip_trade = False

    # Tier 1: shallow dip in strong uptrend
    DIP_MULT1 = 3.9
    LOOKBACK1 = 120
    # Tier 2: deeper dip in shorter-term uptrend
    DIP_MULT2 = 4.0
    LOOKBACK2 = 60
    EXIT_ABOVE_SLOW = 0.25
    DIP_STOP_MULT = 5.5

    for i in range(n):
        close = close_arr[i]
        slow = ema_slow[i]
        atr_val = atr[i]
        base_long = ema_fast[i] > slow
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev1 = ema_slow[max(0, i - LOOKBACK1)]
        slow_prev2 = ema_slow[max(0, i - LOOKBACK2)]
        slow_rising1 = slow > slow_prev1
        slow_rising2 = slow > slow_prev2
        dip_entry = (
            (not base_long)
            and (not base_short)
            and (
                (slow_rising1 and (close < slow - DIP_MULT1 * atr_val))
                or (slow_rising2 and (close < slow - DIP_MULT2 * atr_val))
            )
        )

        if position == 1:
            if is_dip_trade:
                if base_long:
                    is_dip_trade = False  # smooth transition to champion long
                elif close >= slow + EXIT_ABOVE_SLOW * atr_val:
                    position = 0
                    is_dip_trade = False
                elif close < slow - DIP_STOP_MULT * atr_val:
                    position = 0
                    is_dip_trade = False
                elif base_short:
                    position = 0
                    is_dip_trade = False
            else:
                if not base_long:
                    position = 0

        elif position == -1:
            if not base_short:
                position = 0

        if position == 0:
            if base_long:
                position = 1
                is_dip_trade = False
            elif base_short:
                position = -1
            elif dip_entry:
                position = 1
                is_dip_trade = True

        signals[i] = position

    return signals

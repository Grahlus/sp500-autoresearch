"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 368: Dip 3.9 ATR / 120-bar rising + smooth exit at slow+0.25 ATR
         + dip stop-loss at slow - 5.5 ATR.
Hypothesis: exp_366 champion uses EXIT_ABOVE_SLOW=0.5 (exit at slow+0.5 ATR). DIP_MULT=3.9
confirmed optimal. Test EXIT_ABOVE_SLOW=0.25 — exit sooner, before the trend ema stabilizes.
This books profits earlier and allows re-entry on subsequent base_long signal at a better price.
If profit target is the constraint, earlier exit = more closed winning trades = lower drawdown.
Or it might reduce Z5 if dips often recover further than slow+0.25.
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

    atr14 = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().values
    close_arr = df["close"].values

    n = len(close_arr)
    signals = np.zeros(n, dtype=np.int32)
    position = 0
    is_dip_trade = False

    DIP_MULT = 3.9
    SLOW_LOOKBACK = 120
    EXIT_ABOVE_SLOW = 0.25
    DIP_STOP_MULT = 5.5

    for i in range(n):
        close = close_arr[i]
        slow = ema_slow[i]
        atr = atr14[i]
        base_long = ema_fast[i] > slow
        base_short = (ema_fast[i] < slow) and (vol_cur[i] > vol_pct40[i])
        slow_prev = ema_slow[max(0, i - SLOW_LOOKBACK)]
        slow_rising = slow > slow_prev
        dip_entry = (
            (not base_long)
            and (not base_short)
            and slow_rising
            and (close < slow - DIP_MULT * atr)
        )

        if position == 1:
            if is_dip_trade:
                if base_long:
                    is_dip_trade = False  # smooth transition to champion long
                elif close >= slow + EXIT_ABOVE_SLOW * atr:
                    position = 0
                    is_dip_trade = False
                elif close < slow - DIP_STOP_MULT * atr:
                    position = 0  # stop-loss: dip is getting worse, not recovering
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

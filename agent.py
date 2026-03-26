#!/usr/bin/env python3
"""
agent.py — AGENT EDITS THIS FILE each experiment.

WALK-FORWARD VALIDATION (Session 5):
  7 independent 1-year test windows: 2018, 2019, 2020, 2021, 2022, 2023, 2024-H1
  Score = MEAN Sharpe across all windows.
  Commit only if mean Sharpe improves AND no more than 2 windows are negative.

  This prevents gaming a single val window — a real edge must show up
  across bull markets (2019, 2021), COVID (2020), bear (2022), recovery (2023).

HARD CONSTRAINTS (enforced by run.py):
  1. trades_per_year > 150        → REVERT
  2. mean_sharpe < 0.3            → REVERT
  3. neg_windows > 2              → REVERT (fails in more than 2 of 7 windows)
  4. min_sharpe < -0.5            → REVERT (catastrophic in any single window)
  5. ONE change per experiment

STARTING POINT: SP500 champion (best OOS result across all sessions)
  val=1.643 train=1.066 OOS=0.564 — survived 431 unseen trading days
  Config: JT 26w skip=3w rebal=4w top=2.5% stop=20%close-HIGH MA=20w F&G≥22

SESSION 5 GOAL:
  Find a strategy with mean walk-forward Sharpe > 0.8 across all 7 windows.
  That is a genuinely robust strategy, not a regime-specific one.
  Start simple — add complexity only if it consistently helps across windows.
"""
import numpy as np
import pandas as pd

METRIC     = "sharpe"
HYPOTHESIS = "S5-001: SP500 champion baseline — walk-forward across 7 windows"

LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.20
INV_VOL_DAYS   = 6
MIN_HOLD_DAYS  = 5
FG_MIN         = 22.0


def generate_signals(data: dict) -> pd.DataFrame:
    """
    SP500 champion config: JT momentum + F&G entry filter + trailing stop.
    Walk-forward engine will test this across 7 independent windows.
    """
    close   = data["close"]
    volume  = data["volume"]
    fg_raw  = data["fear_greed"]
    dates   = close.index
    tickers = close.columns
    n       = len(dates)

    fg      = fg_raw.reindex(dates).ffill().fillna(50.0).values

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)
    entry_day   = pd.Series(-999,   index=tickers)
    current_pos = pd.Series(0.0,    index=tickers)

    _stops = _enters = _exits = 0

    for i in range(lb_days, n):
        today  = close.iloc[i]
        fg_val = float(fg[i])

        # ── Trailing stop from close HIGH ─────────────────────────────────────
        for tkr in current_pos[current_pos > 0].index:
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                if today[tkr] > ph:
                    pos_high[tkr] = today[tkr]
                    ph = today[tkr]
                days_held = i - int(entry_day.get(tkr, i))
                if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - STOP_LOSS_PCT):
                    current_pos[tkr] = 0.0
                    pos_high[tkr]    = np.nan
                    entry_day[tkr]   = -999
                    _stops          += 1

        # ── Rebalance every rebal_days when F&G >= threshold ─────────────────
        if i % rebal_days == 0 and fg_val >= FG_MIN:
            mom = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
            mom = mom.replace([np.inf, -np.inf], np.nan)
            ma       = close.iloc[max(0, i - ma_days):i].mean()
            above_ma = today > ma

            # Volume filter: top 50% by 20d avg share volume
            avg_vol  = volume.iloc[max(0, i - 20):i].mean()
            high_vol = avg_vol >= avg_vol.median()

            combo = mom.rank(pct=True)
            filt  = combo.where(above_ma & high_vol).dropna()

            if not filt.empty:
                # Adaptive: top 1% bear, top 1.5% greed, top 2.5% normal
                ma_200  = close.iloc[max(0, i - 200):i].mean()
                breadth = float((today > ma_200).mean())
                if breadth < 0.40:
                    eff_pct = 0.010
                elif fg_val > 70:
                    eff_pct = 0.015
                else:
                    eff_pct = TOP_PCT

                n_top       = max(1, int(len(filt) * eff_pct))
                top_tickers = filt.nlargest(n_top).index

                vol_ret      = close.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr]  = today[tkr]
                        entry_day[tkr] = i
                        _enters += 1
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        pos_high[tkr]  = np.nan
                        entry_day[tkr] = -999
                        _exits += 1

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    n_years = max((n - lb_days) / 252, 0.01)
    print(f"  [DIAG] stops/yr={_stops/n_years:.1f}  "
          f"enters/yr={_enters/n_years:.1f}  exits/yr={_exits/n_years:.1f}",
          flush=True)
    return weights

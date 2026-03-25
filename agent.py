#!/usr/bin/env python3
"""
agent.py — AGENT EDITS THIS FILE each experiment.

Rules:
  - generate_signals() receives the full data dict (see prepare.py: load_data)
  - Return a DataFrame: index=dates, columns=tickers, values in [-1, 1]
  - NO lookahead: at row T use only data up to T.
  - Do NOT edit prepare.py, run.py, or evaluate.py.

─────────────────────────────────────────────────────────────────────────────
ENGINE STATUS: FIXED (session 3)
  - Daily drift rebalancing bug removed — trades only on weight changes
  - Trailing stop uses close-price HIGH (not intraday — intraday not tradeable)
  - All session 1/2 numbers were inflated. Session 3 is the truth.

COST MODEL:
  Starting capital : $100,000
  Commission       : $20 per ticker traded
  Slippage         : 5bps one-way
  Execution        : open[T+1] ± slippage

HARD CONSTRAINTS — auto-checked by run.py, revert if violated:
  1. trades_per_year > 150  → REVERT (rebalance_days >= 20 minimum)
  2. train sharpe < 0.5     → REVERT
  3. train_sharpe < val_sharpe / 1.6  → REVERT (overfit)
  4. ONE change per experiment

SESSION 3 BASELINE (exp140, fixed engine):
  Val Sharpe 1.643 | Calmar 3.918 | MaxDD -16.9% | +175.6% | ~55 trades/yr
  Train Sharpe 1.066 | ratio 1.54 (passes 1.6 threshold)
  OOS Sharpe 0.564 (2024-07 → 2026-03) — weak in low-dispersion regime

  Beat val Sharpe 1.643 on chosen METRIC.
  Primary goal: strategy that also works in low-dispersion/high-VIX regimes.
  Do NOT retest anything already in program.md.
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd

METRIC     = "sharpe"
HYPOTHESIS = "S3-053: inv-vol window 20d (medium — between 10d best and 16w)"

LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.13
VOL_MA_DAYS    = 10
MIN_HOLD_DAYS  = 5


def generate_signals(data: dict) -> pd.DataFrame:
    close   = data["close"]
    high    = data["high"]
    volume  = data["volume"]
    fg_raw  = data["fear_greed"]
    dates   = close.index
    fg      = fg_raw.reindex(dates).ffill().fillna(50.0).values
    tickers = close.columns
    n       = len(dates)

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    entry_price = pd.Series(np.nan, index=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)
    entry_day   = pd.Series(-999,   index=tickers)
    current_pos = pd.Series(0.0,    index=tickers)

    _stop_exits = _rebal_exits = _rebal_enters = 0

    for i in range(lb_days, n):
        today    = close.iloc[i]
        fg_val   = float(fg[i])
        ma_200   = close.iloc[max(0, i - 200):i].mean()
        breadth  = float((today > ma_200).mean())

        # ── Trailing stop (close-price HIGH only — no intraday) ───────────────
        for tkr in current_pos[current_pos > 0].index:
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                if today[tkr] > ph:
                    pos_high[tkr] = today[tkr]
                    ph = today[tkr]
                days_held = i - int(entry_day.get(tkr, i))
                if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - STOP_LOSS_PCT):
                    current_pos[tkr] = 0.0
                    entry_price[tkr] = np.nan
                    pos_high[tkr]    = np.nan
                    entry_day[tkr]   = -999
                    _stop_exits     += 1

        # ── Rebalance every rebal_days ─────────────────────────────────────────
        if i % rebal_days == 0:
            mom      = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
            mom      = mom.replace([np.inf, -np.inf], np.nan)
            ma       = close.iloc[max(0, i - ma_days):i].mean()
            above_ma = today > ma

            # Dollar-volume filter: price × shares — levels playing field across
            # universe sizes. A $5 miner and a $500 megacap compete equally.
            dollar_vol  = (close.iloc[max(0, i - VOL_MA_DAYS):i] *
                           volume.iloc[max(0, i - VOL_MA_DAYS):i]).mean()
            high_vol    = dollar_vol >= dollar_vol.quantile(0.70)  # top 30%

            mom_13w     = (close.iloc[i - skip_days] / close.iloc[max(0, i - 80)] - 1)
            mom_13w     = mom_13w.replace([np.inf, -np.inf], np.nan)
            combo       = mom.rank(pct=True) + mom_13w.rank(pct=True)
            combo_filt  = combo.where(above_ma & high_vol).dropna()

            if not combo_filt.empty:
                if breadth < 0.40:
                    eff_pct = 0.010
                else:
                    eff_pct = TOP_PCT

                n_top       = max(1, int(len(combo_filt) * eff_pct))
                top_tickers = combo_filt.nlargest(n_top).index

                vol_ret      = close.iloc[max(0, i - 20):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        entry_price[tkr] = today[tkr]
                        pos_high[tkr]    = today[tkr]
                        entry_day[tkr]   = i
                        _rebal_enters   += 1
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        entry_price[tkr] = np.nan
                        pos_high[tkr]    = np.nan
                        entry_day[tkr]   = -999
                        _rebal_exits    += 1

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    n_years = (n - lb_days) / 252
    print(f"  [DIAG] stop_exits/yr={_stop_exits/n_years:.1f}  "
          f"rebal_exits/yr={_rebal_exits/n_years:.1f}  "
          f"rebal_enters/yr={_rebal_enters/n_years:.1f}", flush=True)
    return weights

#!/usr/bin/env python3
"""
agent.py — AGENT EDITS THIS FILE each experiment.

Rules:
  - generate_signals() returns DataFrame: index=dates, columns=tickers, [-1,1]
  - NO lookahead: at row T use only data up to T
  - Do NOT edit prepare.py, run.py, or evaluate.py

─────────────────────────────────────────────────────────────────────────────
SESSION 4 GOAL: Dual-mode regime classifier
  Switch between momentum (RSL) and mean reversion based on market regime.
  Root cause of OOS failure: momentum broke down in 2024-2026 low-dispersion regime.

HARD CONSTRAINTS:
  1. trades_per_year > 150  → REVERT
  2. train sharpe < 0.5     → REVERT
  3. train_sharpe < val_sharpe / 1.6  → REVERT
  4. ONE change per experiment

CONFIRMED MOMENTUM PARAMS (R1000, session 3 — do not re-tune):
  26w+16w sum, skip=3w, rebal=4w, top=2.5%, stop=13% close-HIGH,
  inv-vol=12d, dollar-vol top 30%, no vol_accel, no F&G gate, MA=20w

MEAN REVERSION SIGNAL (to build and tune in session 4):
  Target: stocks down 8-20% in past 15 days, above 200d MA, liquid
  Rank by drop magnitude → long top 2-3%
  No trailing stop in MR mode (exit at next rebalance)

REGIME DETECTOR (start with cross-sectional dispersion):
  Compute std of 13w returns across universe each rebalance
  Rolling 52w median of that std = baseline
  dispersion > median → MOMENTUM regime
  dispersion < median → MEAN REVERSION regime
  Print regime each rebalance in DIAG line for verification
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd

METRIC     = "sharpe"
HYPOTHESIS = "S4-031: blend 40/60 MOM/MR"

# ── Momentum params (confirmed optimal, do not change) ───────────────────────
MOM_LOOKBACK_WEEKS = 26
MOM_SECONDARY_WEEKS = 16
MOM_SKIP_WEEKS     = 3
REBAL_WEEKS        = 4
MOM_TOP_PCT        = 0.025
MA_WEEKS           = 20
STOP_LOSS_PCT      = 0.13
INV_VOL_DAYS       = 12
DOLLAR_VOL_PCT     = 0.30
MIN_HOLD_DAYS      = 5

# ── Regime detector params ────────────────────────────────────────────────────
DISPERSION_LOOKBACK = 65    # 13 weeks for momentum std measurement
DISPERSION_WINDOW   = 252   # 52 weeks rolling median baseline


def _regime(close: pd.DataFrame, i: int) -> str:
    """
    Returns 'MOM' or 'MR' based on cross-sectional return dispersion.
    High dispersion (stocks diverging) → momentum works.
    Low dispersion (stocks converging) → mean reversion works.
    """
    if i < DISPERSION_WINDOW + DISPERSION_LOOKBACK:
        return "MOM"   # not enough history — default to momentum

    # Current cross-sectional dispersion: std of 13w returns across universe
    ret_13w    = close.iloc[i] / close.iloc[i - DISPERSION_LOOKBACK] - 1
    current_disp = ret_13w.std()

    # Rolling 52-week median of that dispersion
    dispersions = []
    for j in range(i - DISPERSION_WINDOW, i, 5):   # sample every week
        r = close.iloc[j] / close.iloc[j - DISPERSION_LOOKBACK] - 1
        dispersions.append(r.std())
    median_disp = np.percentile(dispersions, 40)

    return "MOM" if current_disp >= median_disp else "MR"


def _momentum_signal(close, volume, i, tickers, lb_days, skip_days,
                     ma_days, rebal_days) -> pd.Series:
    """R1000 champion momentum signal."""
    today    = close.iloc[i]
    mom_26w  = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
    mom_16w  = (close.iloc[i - skip_days] / close.iloc[max(0, i - MOM_SECONDARY_WEEKS*5)] - 1)
    mom      = (mom_26w + mom_16w).replace([np.inf, -np.inf], np.nan)

    ma       = close.iloc[max(0, i - ma_days):i].mean()
    above_ma = today > ma

    dv       = (close.iloc[max(0, i - 20):i] * volume.iloc[max(0, i - 20):i]).mean()
    high_dv  = dv >= dv.quantile(1 - DOLLAR_VOL_PCT)

    combo    = mom.rank(pct=True)
    filt     = combo.where(above_ma & high_dv).dropna()
    if filt.empty:
        return pd.Series(0.0, index=tickers)

    n_top       = max(1, int(len(filt) * MOM_TOP_PCT))
    top_tkrs    = filt.nlargest(n_top).index
    vol_ret     = close.iloc[max(0, i - INV_VOL_DAYS):i][top_tkrs].pct_change().std()
    inv_vol     = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
    inv_vol_n   = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

    w = pd.Series(0.0, index=tickers)
    for tkr in top_tkrs:
        w[tkr] = inv_vol_n.get(tkr, 0.0)
    return w


def _mr_signal(close, volume, i, tickers, ma_days) -> pd.Series:
    """
    Mean reversion signal: long recent losers above 200d MA.
    Stocks down 8-20% in past 15 days → candidates for reversal.
    """
    today    = close.iloc[i]
    ret_15d  = today / close.iloc[max(0, i - 20)] - 1

    # Only stocks that dropped 8-20% (oversold but not broken)
    oversold = (ret_15d <= -0.12) & (ret_15d >= -0.30)

    # Must be above 200d MA (not in structural downtrend)
    ma_200   = close.iloc[max(0, i - 200):i].mean()
    above_ma = today > ma_200

    # Dollar-volume filter: top 50% (need liquidity for MR)
    dv       = (close.iloc[max(0, i - 20):i] * volume.iloc[max(0, i - 20):i]).mean()
    liquid   = dv >= dv.median()

    candidates = ret_15d.where(oversold & above_ma & liquid).dropna()
    if candidates.empty:
        return pd.Series(0.0, index=tickers)

    # Rank by drop magnitude (biggest drop = strongest MR candidate)
    mr_rank  = candidates.rank(pct=True)   # lowest return = rank 0 = best MR
    n_top    = max(1, int(len(candidates) * 0.03))
    top_tkrs = mr_rank.nsmallest(n_top).index   # most oversold

    # Equal weight in MR mode (inv-vol rewards low-vol stocks, not what we want here)
    w = pd.Series(0.0, index=tickers)
    for tkr in top_tkrs:
        w[tkr] = 1.0 / n_top
    return w


def generate_signals(data: dict) -> pd.DataFrame:
    close   = data["close"]
    volume  = data["volume"]
    dates   = close.index
    tickers = close.columns
    n       = len(dates)

    lb_days    = MOM_LOOKBACK_WEEKS * 5
    skip_days  = MOM_SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)
    entry_day   = pd.Series(-999,   index=tickers)
    current_pos = pd.Series(0.0,    index=tickers)

    _stop_exits  = 0
    _mom_rebal   = 0
    _mr_rebal    = 0
    _mom_days    = 0
    _mr_days     = 0

    for i in range(lb_days, n):
        today = close.iloc[i]

        # ── Trailing stop (MOM positions only) ───────────────────────────────
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
                    _stop_exits     += 1

        # ── Rebalance every rebal_days ────────────────────────────────────────
        if i % rebal_days == 0:
            regime = _regime(close, i)

            mom_pos = _momentum_signal(close, volume, i, tickers,
                                       lb_days, skip_days, ma_days, rebal_days)
            mr_pos  = _mr_signal(close, volume, i, tickers, ma_days)

            if regime == "MOM":
                new_pos = mom_pos * 0.4 + mr_pos * 0.6
                # Set entry tracking for MOM-weighted positions
                for tkr in tickers:
                    if new_pos[tkr] > 0 and current_pos[tkr] == 0.0:
                        pos_high[tkr]  = today[tkr]
                        entry_day[tkr] = i
                    elif new_pos[tkr] == 0.0:
                        pos_high[tkr]  = np.nan
                        entry_day[tkr] = -999
                _mom_rebal += 1
                _mom_days  += rebal_days
            else:
                new_pos = mr_pos * 0.55 + mom_pos * 0.45
                # No trailing stop in MR mode — clear all tracking
                pos_high[:]  = np.nan
                entry_day[:] = -999
                _mr_rebal += 1
                _mr_days  += rebal_days

            current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    total_rebal = max(_mom_rebal + _mr_rebal, 1)
    n_years     = (n - lb_days) / 252
    print(f"  [DIAG] stop_exits/yr={_stop_exits/n_years:.1f}  "
          f"MOM_rebal={_mom_rebal}({100*_mom_rebal//total_rebal}%)  "
          f"MR_rebal={_mr_rebal}({100*_mr_rebal//total_rebal}%)",
          flush=True)
    return weights

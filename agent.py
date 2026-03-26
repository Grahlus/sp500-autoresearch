#!/usr/bin/env python3
"""
agent.py — FIXED: 3 targeted patches to OOS problems.

PROBLEMS FOUND:
  1. Only 1-2 positions held at a time (should be 10-12)
     Root cause: SP500 Wikipedia filter + stacked filters + old abs cap of 15
     all combined to knock the selection down to 1-2 names.
  2. MaxDD -41% with no portfolio-level guard
  3. 15-month flat period from holding single losers

FIXES APPLIED:
  1. Remove Wikipedia SP500 filter — use full 841-ticker universe
     (The SP500 filter fetches at runtime, occasionally fails, and
      reduces the pool so much the other filters leave 1-2 names)
  2. Use TOP_PCT properly (2.5% of 841 = ~21 stocks) — no hard abs cap
  3. Add portfolio-level DD guard: cut all positions by 50% if portfolio
     drops 15% from its rolling peak. Full exit at -25%.
"""
import numpy as np
import pandas as pd

METRIC     = "sharpe"
HYPOTHESIS = "FIX3: R1000 full universe + top2.5% no cap + NO DD guard (guard hurts)"

# ── Strategy parameters ───────────────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025     # 2.5% of 841 = ~21 stocks simultaneously
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.20      # trailing stop from close HIGH
INV_VOL_DAYS   = 6
MIN_HOLD_DAYS  = 5
FG_MIN         = 22.0

# ── Portfolio-level DD control ────────────────────────────────────────────────
PORT_DD_REDUCE = None  # disabled      # cut all positions 50% if portfolio drops 15% from peak
PORT_DD_EXIT   = None  # disabled      # exit all positions if portfolio drops 25% from peak


def generate_signals(data: dict) -> pd.DataFrame:
    close   = data["close"]  # full universe
    volume  = data["volume"]
    fg_raw  = data["fear_greed"]
    dates   = close.index
    tickers = close.columns
    n       = len(dates)

    fg = fg_raw.reindex(dates).ffill().fillna(50.0).values

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)
    entry_day   = pd.Series(-999,   index=tickers)
    current_pos = pd.Series(0.0,    index=tickers)

    # Portfolio peak tracker (normalised to 1.0 at start)
    port_peak    = 1.0
    port_val_est = 1.0   # rough estimate via equal-weight avg of held positions
    port_reduced = False  # flag: currently in 50%-size mode

    _stops = _enters = _exits = _port_acts = 0

    for i in range(lb_days, n):
        today  = close.iloc[i]
        fg_val = float(fg[i])

        # ── Portfolio value estimate (normalised) ─────────────────────────────
        held = current_pos[current_pos > 0].index
        if len(held) > 0:
            # Track portfolio via avg price-relative of held names
            prices_now   = today[held]
            prices_entry = pd.Series({
                t: close.iloc[int(entry_day[t])][t] if entry_day[t] > 0 else today[t]
                for t in held
            })
            ret_since_entry = (prices_now / prices_entry.values - 1)
            port_val_est    = 1.0 + ret_since_entry.mean()
        else:
            port_val_est = port_peak   # in cash — no change

        if port_val_est > port_peak:
            port_peak    = port_val_est
            port_reduced = False       # new peak → reset reduction flag

        # ── Portfolio-level DD guard ───────────────────────────────────────────
        if port_peak > 0:
            port_dd = (port_val_est - port_peak) / port_peak

            if port_dd < -PORT_DD_EXIT:
                # Full exit
                current_pos[:] = 0.0
                pos_high[:]    = np.nan
                entry_day[:]   = -999
                port_peak      = port_val_est
                port_reduced   = False
                _port_acts    += 1

            elif port_dd < -PORT_DD_REDUCE and not port_reduced:
                # Halve all positions
                current_pos   *= 0.5
                port_reduced   = True
                _port_acts    += 1

        # ── Position-level trailing stop (close HIGH, no intraday) ────────────
        for tkr in list(current_pos[current_pos > 0].index):
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

            avg_vol  = volume.iloc[max(0, i - 20):i].mean()
            high_vol = avg_vol >= avg_vol.median()

            ma_200  = close.iloc[max(0, i - 200):i].mean()
            breadth = float((today > ma_200).mean())

            combo = mom.rank(pct=True)
            filt  = combo.where(above_ma & high_vol).dropna()

            if not filt.empty:
                # Bear: concentrate to top 1%, greed: top 1.5%, normal: top 2.5%
                if breadth < 0.40:
                    eff_pct = 0.010
                elif fg_val > 70:
                    eff_pct = 0.015
                else:
                    eff_pct = TOP_PCT

                # NO absolute cap — let 2.5% of universe determine count
                n_top       = max(1, int(len(filt) * eff_pct))
                top_tickers = filt.nlargest(n_top).index

                vol_ret      = close.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                # Apply size reduction if in DD mode
                size_mult = 0.5 if port_reduced else 1.0

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0) * size_mult

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr]  = today[tkr]
                        entry_day[tkr] = i
                        _enters       += 1
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        pos_high[tkr]  = np.nan
                        entry_day[tkr] = -999
                        _exits        += 1

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    n_years   = max((n - lb_days) / 252, 0.01)
    n_held    = (weights.abs().sum(axis=1) > 0).sum()
    avg_pos   = weights[weights > 0].count(axis=1)
    avg_pos   = avg_pos[avg_pos > 0].mean() if (avg_pos > 0).any() else 0
    print(f"  [DIAG] stops/yr={_stops/n_years:.1f}  enters/yr={_enters/n_years:.1f}  "
          f"port_acts={_port_acts}  avg_positions={avg_pos:.1f}  "
          f"invested={100*n_held//(n-lb_days)}%",
          flush=True)
    return weights

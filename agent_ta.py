#!/usr/bin/env python3
"""
agent_ta.py — TA-filter session.  S10-005 base + technical indicator gates.
S10-005 agent.py is LOCKED — all experiments run against this file only.

SESSION GOAL:
  Find if any TA filter improves WF Sharpe above 0.722 AND OOS Sharpe above 1.548.
  Decision rule: WF > 0.722, OOS > 1.548, ta_blocks/yr <= 6, neg_windows <= 4.

HARD CONSTRAINTS (enforced by run_ta.py):
  1. trades_per_year > 150        → REVERT
  2. mean_sharpe < 0.3            → REVERT
  3. neg_windows > 4              → REVERT
  4. min_sharpe < -1.2            → REVERT
"""
import numpy as np
import pandas as pd

METRIC     = "sharpe"
HYPOTHESIS = "TA-C04: Double-top entry block + BB exit >0.95 — best entry + best exit filter"

LOOKBACK_WEEKS  = 26
SKIP_WEEKS      = 3
REBAL_WEEKS     = 4
TOP_PCT         = 0.025
MA_WEEKS        = 20
STOP_LOSS_PCT   = 0.20
STOP_PARABOLIC  = 0.30
STOP_TYPE       = "adaptive"
INV_VOL_DAYS    = 15
MIN_HOLD_DAYS   = 5
FG_MIN          = 10.0

EXIT_PCT_RANK   = 0.97
RANK_EXIT_CONFIRM = None

# ── TA ENTRY FILTERS ──────────────────────────────────────────────────────────
USE_BB_GATE        = False   # TA-001/002: block entry if bb_position > threshold
BB_GATE_THRESHOLD  = 0.85

USE_SMA50_SLOPE    = False   # TA-003/004: block entry if sma50_slope < threshold
SMA50_SLOPE_MIN    = 0.0

USE_SMA50_POSITION = False   # TA-005: block entry if price below SMA50
USE_TOP_PATTERN    = True    # TA-006/007/008: block entry on double/triple-top
TOP_PATTERN_MODE   = "double"  # "double" | "triple" | "any"
USE_BOTTOM_PATTERN = False   # TA-009: only enter if double-bottom detected

# ── TA EXIT FILTERS ───────────────────────────────────────────────────────────
USE_SMA50_EXIT     = False   # TA-010/011: exit if close below SMA50
SMA50_EXIT_CROSS   = True    # True=fresh cross only, False=any close below

USE_TOP_EXIT       = False   # TA-012: exit if double-top forms after entry
USE_BB_EXIT        = True    # TA-013: exit if bb_position > threshold (overbought)
BB_EXIT_THRESHOLD  = 0.95


def generate_signals(data: dict) -> pd.DataFrame:
    """
    S10-005 base + TA filter gates.  All flags default False = exact S10-005.
    """
    from ta_filters import (bb_position, sma50_slope, sma50_position,
                            detect_double_top, detect_triple_top,
                            detect_double_bottom)

    import requests
    from io import StringIO
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        tbl  = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})[0]
        sp500_tkrs = set(tbl["Symbol"].str.replace(".", "-", regex=False).tolist())
        sp500_tkrs.add("SPY")
    except Exception:
        sp500_tkrs = set(data["close"].columns)

    close   = data["close"][[c for c in data["close"].columns if c in sp500_tkrs]]
    volume  = data["volume"][close.columns]
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
    confirm_count = pd.Series(0, index=tickers)

    _stops = _enters = _exits = _rank_exits = 0
    _ta_entry_blocks = 0   # entries blocked by TA filter
    _ta_exits = 0          # exits triggered by TA filter

    confirm_needed = int(RANK_EXIT_CONFIRM) if RANK_EXIT_CONFIRM else 0

    for i in range(lb_days, n):
        today  = close.iloc[i]
        fg_val = float(fg[i])

        # ── Trailing stop from close HIGH ─────────────────────────────────────
        if STOP_TYPE != "none":
            for tkr in current_pos[current_pos > 0].index:
                ph = pos_high.get(tkr, np.nan)
                if not np.isnan(ph) and ph > 0:
                    if today[tkr] > ph:
                        pos_high[tkr] = today[tkr]
                        ph = today[tkr]
                    days_held = i - int(entry_day.get(tkr, i))
                    if STOP_TYPE == "adaptive":
                        prev_20   = close.iloc[max(0, i - 20)][tkr]
                        short_mom = (today[tkr] / prev_20 - 1) if prev_20 > 0 else 0.0
                        eff_stop  = STOP_PARABOLIC if short_mom > 0.05 else STOP_LOSS_PCT
                    else:
                        eff_stop = STOP_LOSS_PCT
                    if days_held >= MIN_HOLD_DAYS and today[tkr] < ph * (1 - eff_stop):
                        current_pos[tkr]    = 0.0
                        pos_high[tkr]       = np.nan
                        entry_day[tkr]      = -999
                        confirm_count[tkr]  = 0
                        _stops             += 1

        # ── Rank-based exit ───────────────────────────────────────────────────
        if EXIT_PCT_RANK is not None and current_pos[current_pos > 0].any():
            held = current_pos[current_pos > 0].index.tolist()
            mom_now = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
            mom_now = mom_now.replace([np.inf, -np.inf], np.nan)
            ranks   = mom_now.rank(pct=True)

            for tkr in held:
                if current_pos.get(tkr, 0.0) == 0.0:
                    continue
                tkr_rank = ranks.get(tkr, np.nan)
                if np.isnan(tkr_rank):
                    continue
                if tkr_rank < EXIT_PCT_RANK:
                    confirm_count[tkr] += 1
                    if confirm_count[tkr] > confirm_needed:
                        current_pos[tkr]    = 0.0
                        pos_high[tkr]       = np.nan
                        entry_day[tkr]      = -999
                        confirm_count[tkr]  = 0
                        _rank_exits        += 1
                else:
                    confirm_count[tkr] = 0

        # ── TA EXIT FILTERS (applied after rank/stop, before rebalance) ───────
        if (USE_SMA50_EXIT or USE_TOP_EXIT or USE_BB_EXIT):
            held = current_pos[current_pos > 0].index.tolist()
            for tkr in held:
                if current_pos.get(tkr, 0.0) == 0.0:
                    continue
                days_held = i - int(entry_day.get(tkr, i))
                if days_held < MIN_HOLD_DAYS:
                    continue

                price_today = float(today[tkr])

                if USE_SMA50_EXIT:
                    close_slice = close.iloc[max(0, i - 60):i][tkr]
                    sma50_val   = float(close_slice.iloc[-50:].mean()) if len(close_slice) >= 50 else float(close_slice.mean())
                    if SMA50_EXIT_CROSS:
                        # Fresh cross: today below, yesterday above
                        prev_price = float(close.iloc[i - 1][tkr])
                        prev_slice = close.iloc[max(0, i - 61):i - 1][tkr]
                        sma50_prev = float(prev_slice.iloc[-50:].mean()) if len(prev_slice) >= 50 else float(prev_slice.mean())
                        if price_today < sma50_val and prev_price >= sma50_prev:
                            current_pos[tkr]    = 0.0
                            pos_high[tkr]       = np.nan
                            entry_day[tkr]      = -999
                            confirm_count[tkr]  = 0
                            _ta_exits          += 1
                            continue
                    else:
                        if price_today < sma50_val:
                            current_pos[tkr]    = 0.0
                            pos_high[tkr]       = np.nan
                            entry_day[tkr]      = -999
                            confirm_count[tkr]  = 0
                            _ta_exits          += 1
                            continue

                if USE_TOP_EXIT and current_pos.get(tkr, 0.0) > 0:
                    close_slice = close.iloc[max(0, i - 60):i + 1][tkr]
                    if detect_double_top(close_slice):
                        current_pos[tkr]    = 0.0
                        pos_high[tkr]       = np.nan
                        entry_day[tkr]      = -999
                        confirm_count[tkr]  = 0
                        _ta_exits          += 1
                        continue

                if USE_BB_EXIT and current_pos.get(tkr, 0.0) > 0:
                    close_slice = close.iloc[max(0, i - 20):i + 1][tkr]
                    bb_pos = bb_position(close_slice, window=20)
                    if bb_pos > BB_EXIT_THRESHOLD:
                        current_pos[tkr]    = 0.0
                        pos_high[tkr]       = np.nan
                        entry_day[tkr]      = -999
                        confirm_count[tkr]  = 0
                        _ta_exits          += 1
                        continue

        # ── Rebalance every rebal_days when F&G >= threshold ─────────────────
        ma_200  = close.iloc[max(0, i - 100):i].mean()
        breadth = float((today > ma_200).mean())

        if i % rebal_days == 0 and fg_val >= FG_MIN and breadth <= 0.85:
            mom = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
            mom = mom.replace([np.inf, -np.inf], np.nan)
            ma       = close.iloc[max(0, i - ma_days):i].mean()
            above_ma = today > ma

            avg_dvol = (close.iloc[max(0, i - 20):i] *
                        volume.iloc[max(0, i - 20):i]).mean()
            high_vol = avg_dvol >= avg_dvol.quantile(0.70)

            combo = mom.rank(pct=True)
            filt  = combo.where(above_ma & high_vol).dropna()

            if not filt.empty:
                if breadth < 0.40:
                    eff_pct = 0.010
                elif fg_val > 80:
                    eff_pct = 0.015
                else:
                    eff_pct = TOP_PCT

                n_top       = min(max(1, int(len(filt) * eff_pct)), 15)
                top_tickers = list(filt.nlargest(n_top).index)

                # ── TA ENTRY FILTERS ─────────────────────────────────────────
                filtered_out = []
                for tkr in top_tickers:
                    blocked = False

                    if USE_BB_GATE:
                        close_slice = close.iloc[max(0, i - 20):i][tkr]
                        bb_pos = bb_position(close_slice, window=20)
                        if bb_pos > BB_GATE_THRESHOLD:
                            blocked = True

                    if not blocked and USE_SMA50_SLOPE:
                        close_slice = close.iloc[max(0, i - 60):i][tkr]
                        slope = sma50_slope(close_slice)
                        if slope < SMA50_SLOPE_MIN:
                            blocked = True

                    if not blocked and USE_SMA50_POSITION:
                        close_slice = close.iloc[max(0, i - 60):i][tkr]
                        pos = sma50_position(float(today[tkr]), close_slice)
                        if pos < 0:
                            blocked = True

                    if not blocked and USE_TOP_PATTERN:
                        if TOP_PATTERN_MODE in ("double", "any"):
                            close_slice = close.iloc[max(0, i - 60):i + 1][tkr]
                            if detect_double_top(close_slice):
                                blocked = True
                        if not blocked and TOP_PATTERN_MODE in ("triple", "any"):
                            close_slice = close.iloc[max(0, i - 80):i + 1][tkr]
                            if detect_triple_top(close_slice):
                                blocked = True

                    if not blocked and USE_BOTTOM_PATTERN:
                        close_slice = close.iloc[max(0, i - 60):i + 1][tkr]
                        if not detect_double_bottom(close_slice):
                            blocked = True

                    if blocked:
                        filtered_out.append(tkr)
                        _ta_entry_blocks += 1

                for tkr in filtered_out:
                    top_tickers.remove(tkr)

                if not top_tickers:
                    # All candidates blocked — skip rebalance this period
                    weights.iloc[i] = current_pos
                    continue

                top_tickers_idx = pd.Index(top_tickers)
                vol_ret      = close.iloc[max(0, i - INV_VOL_DAYS):i][top_tickers_idx].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers_idx:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers_idx:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        pos_high[tkr]       = today[tkr]
                        entry_day[tkr]      = i
                        confirm_count[tkr]  = 0
                        _enters            += 1
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        pos_high[tkr]       = np.nan
                        entry_day[tkr]      = -999
                        confirm_count[tkr]  = 0
                        _exits             += 1

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    n_years = max((n - lb_days) / 252, 0.01)
    n_held  = (weights.abs().sum(axis=1) > 0).sum()
    avg_pos = weights[weights > 0].count(axis=1)
    avg_pos = avg_pos[avg_pos > 0].mean() if (avg_pos > 0).any() else 0
    print(f"  [DIAG] stops/yr={_stops/n_years:.1f}  "
          f"enters/yr={_enters/n_years:.1f}  exits/yr={_exits/n_years:.1f}  "
          f"rank_exits/yr={_rank_exits/n_years:.1f}  "
          f"ta_blocks/yr={_ta_entry_blocks/n_years:.1f}  "
          f"ta_exits/yr={_ta_exits/n_years:.1f}  "
          f"avg_positions={avg_pos:.1f}  invested={100*n_held//(n-lb_days)}%",
          flush=True)
    return weights

#!/usr/bin/env python3
"""
agent.py — AGENT EDITS THIS FILE each experiment.
Implement generate_signals() and set METRIC + HYPOTHESIS.

Rules:
  - generate_signals() receives the full data dict (see prepare.py: load_data)
  - Return a DataFrame: index=dates, columns=tickers, values in [-1, 1]
    Positive = long, Negative = short. Gross exposure normalised to 1.0 by engine.
  - NO lookahead: at row T you may only use data up to and including row T.
    (open[T+1] is the execution price — never use it as a signal input.)
  - Do NOT edit prepare.py or run.py.

─────────────────────────────────────────────────────────────────────────────
KNOWN BASELINE (from prior 10-year backtest on ~400 SP500 stocks, $100k):
  RSL / Jegadeesh-Titman momentum:
    skip=4, top3%, inv-vol sizing, 20-week MA filter, 20% stop-loss
    → +4,455% | Sharpe 1.37 | MaxDD -33.7% | 343 trades

  This is the benchmark to beat. Do NOT retest configs already in program.md.
  Focus on: adding shorts, VIX/Fear&Greed filters, multi-factor combos, ML.
─────────────────────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd

# ── Experiment config (agent sets these each run) ────────────────────────────
METRIC     = "sharpe"
HYPOTHESIS = (
    "RSL + vol top50% + vol accel + inv-vol 20d + greed-only short leg: "
    "short bottom 1.5% JT below MA when F&G>75 (greed regime only)"
)

# ── Strategy parameters ──────────────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025
SHORT_PCT      = 0.015
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.15
VOL_MA_DAYS    = 20


def generate_signals(data: dict) -> pd.DataFrame:
    """
    RSL + vol filter + vol accel composite + F&G regime:
    - Extreme Greed (F&G > 75): tighten to top 1.5%
    - Extreme Fear (F&G < 25): skip new entries (hold existing until stop-loss)
    - Normal: top 2.5%
    """
    close   = data["close"]
    volume  = data["volume"]
    fg_raw  = data["fear_greed"]
    dates   = close.index

    fg = fg_raw.reindex(dates).ffill().fillna(50.0).values
    tickers = close.columns
    n       = len(dates)

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    entry_price = pd.Series(np.nan, index=tickers)
    pos_high    = pd.Series(np.nan, index=tickers)  # rolling high since entry
    current_pos = pd.Series(0.0,   index=tickers)

    for i in range(lb_days, n):
        today  = close.iloc[i]
        fg_val = float(fg[i])

        # ── Trailing stop from position high (not entry) ──────────────────────
        for tkr in current_pos[current_pos > 0].index:
            ph = pos_high.get(tkr, np.nan)
            if not np.isnan(ph) and ph > 0:
                # Update rolling high
                if today[tkr] > ph:
                    pos_high[tkr] = today[tkr]
                    ph = today[tkr]
                # Trailing stop: exit if price drops 15% from rolling high
                if today[tkr] < ph * (1 - STOP_LOSS_PCT):
                    current_pos[tkr] = 0.0
                    entry_price[tkr] = np.nan
                    pos_high[tkr]    = np.nan

        # ── Rebalance every rebal_days ────────────────────────────────────────
        if i % rebal_days == 0 and fg_val >= 25.0:   # skip new entries in extreme fear
            mom = (close.iloc[i - skip_days] / close.iloc[i - lb_days] - 1)
            mom = mom.replace([np.inf, -np.inf], np.nan)
            ma       = close.iloc[max(0, i - ma_days):i].mean()
            above_ma = today > ma

            # Volume filter: top 50% by 20-day avg volume
            avg_vol     = volume.iloc[max(0, i - VOL_MA_DAYS):i].mean()
            high_volume = avg_vol >= avg_vol.median()

            # Volume acceleration: recent 5-day avg / 20-day avg
            recent_vol  = volume.iloc[max(0, i - 5):i].mean()
            vol_accel   = (recent_vol / avg_vol.replace(0, np.nan)).fillna(1.0)

            # Composite score: percentile ranks multiplied
            mom_rank    = mom.rank(pct=True)
            accel_rank  = vol_accel.rank(pct=True)
            combo       = mom_rank * accel_rank

            combo_filtered = combo.where(above_ma & high_volume).dropna()

            new_pos = pd.Series(0.0, index=tickers)

            if not combo_filtered.empty:
                # F&G regime: tighten concentration in extreme greed (>70)
                eff_pct = 0.015 if fg_val > 70.0 else TOP_PCT
                n_top   = max(1, int(len(combo_filtered) * eff_pct))
                top_tickers = combo_filtered.nlargest(n_top).index

                vol_ret      = close.iloc[max(0, i - 20):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        entry_price[tkr] = today[tkr]
                        pos_high[tkr]    = today[tkr]
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        entry_price[tkr] = np.nan
                        pos_high[tkr]    = np.nan

            # ── Greed-only short leg (F&G > 75): bottom 1.5% JT, below MA ───────
            if fg_val > 75.0:
                below_ma    = today < ma
                short_cands = mom.where(below_ma).dropna()
                if not short_cands.empty:
                    n_short = max(1, int(len(short_cands) * SHORT_PCT))
                    short_tickers = short_cands.nsmallest(n_short).index
                    vol_s = close.iloc[max(0, i - 20):i][short_tickers].pct_change().std()
                    inv_s = (1.0 / vol_s.replace(0, np.nan)).fillna(0.0)
                    inv_s_norm = inv_s / inv_s.sum() if inv_s.sum() > 0 else inv_s
                    for tkr in short_tickers:
                        new_pos[tkr] = -inv_s_norm.get(tkr, 0.0)

            current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights

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
    "exp079: adaptive breadth — top 1% when <40% SP500 above 200d MA (bear), "
    "top 2.5% normal, top 1.5% greed>70"
)

# ── Strategy parameters ──────────────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 3
REBAL_WEEKS    = 4
TOP_PCT        = 0.025
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.20
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

            if not combo_filtered.empty:
                # Market breadth: fraction of stocks above 200d MA
                ma_200  = close.iloc[max(0, i - 200):i].mean()
                breadth = float((today > ma_200).mean())

                # Adaptive concentration: defensive in bear, tight in greed
                if breadth < 0.40:
                    eff_pct = 0.010   # bear market — top 1%, very selective
                elif fg_val > 70.0:
                    eff_pct = 0.015   # greed — top 1.5%
                else:
                    eff_pct = TOP_PCT  # normal — top 2.5%
                n_top   = max(1, int(len(combo_filtered) * eff_pct))
                top_tickers = combo_filtered.nlargest(n_top).index

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
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        entry_price[tkr] = np.nan
                        pos_high[tkr]    = np.nan

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights

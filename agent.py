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
    "RSL + vol top50% + vol accel composite 15%stop: "
    "top 2.5% (between 2% and 3%)"
)

# ── Strategy parameters ──────────────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 4
REBAL_WEEKS    = 4
TOP_PCT        = 0.025  # 2.5% (between 2% and 3%)
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.15
VOL_MA_DAYS    = 20


def generate_signals(data: dict) -> pd.DataFrame:
    """
    RSL + volume filter + volume acceleration composite:
    - Volume filter: top 50% by 20-day avg volume
    - Composite rank = JT_momentum_rank × volume_acceleration_rank
      where vol_accel = recent 5-day avg vol / 20-day avg vol
    """
    close   = data["close"]
    volume  = data["volume"]
    dates   = close.index
    tickers = close.columns
    n       = len(dates)

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    entry_price = pd.Series(np.nan, index=tickers)
    current_pos = pd.Series(0.0,   index=tickers)

    for i in range(lb_days, n):
        today = close.iloc[i]

        # ── Stop-loss check every day ─────────────────────────────────────────
        for tkr in current_pos[current_pos > 0].index:
            ep = entry_price.get(tkr, np.nan)
            if not np.isnan(ep) and ep > 0:
                if today[tkr] < ep * (1 - STOP_LOSS_PCT):
                    current_pos[tkr] = 0.0
                    entry_price[tkr] = np.nan

        # ── Rebalance every rebal_days ────────────────────────────────────────
        if i % rebal_days == 0:
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
                n_top       = max(1, int(len(combo_filtered) * TOP_PCT))
                top_tickers = combo_filtered.nlargest(n_top).index

                vol_ret      = close.iloc[max(0, i - 60):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol_ret.replace(0, np.nan)).fillna(0.0)
                inv_vol_norm = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else inv_vol

                new_pos = pd.Series(0.0, index=tickers)
                for tkr in top_tickers:
                    new_pos[tkr] = inv_vol_norm.get(tkr, 0.0)

                for tkr in top_tickers:
                    if current_pos.get(tkr, 0.0) == 0.0:
                        entry_price[tkr] = today[tkr]
                for tkr in current_pos[current_pos > 0].index:
                    if new_pos[tkr] == 0.0:
                        entry_price[tkr] = np.nan

                current_pos = new_pos.copy()

        weights.iloc[i] = current_pos

    return weights

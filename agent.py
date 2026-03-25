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
    "RSL baseline + VIX regime sizing: VIX<20=100%, VIX20-30=60%, VIX>30=20% exposure"
)

# ── Strategy parameters ──────────────────────────────────────────────────────
LOOKBACK_WEEKS = 26
SKIP_WEEKS     = 4
REBAL_WEEKS    = 4
TOP_PCT        = 0.03
MA_WEEKS       = 20
STOP_LOSS_PCT  = 0.20


def generate_signals(data: dict) -> pd.DataFrame:
    """
    RSL momentum + VIX-based position scaling.
    VIX < 20: full exposure; VIX 20-30: 60%; VIX > 30: 20%.
    """
    close   = data["close"]
    vix_raw = data["vix"]
    dates   = close.index
    tickers = close.columns
    n       = len(dates)

    # Align VIX to close index
    vix = vix_raw.reindex(dates).ffill().fillna(20.0).values

    lb_days    = LOOKBACK_WEEKS * 5
    skip_days  = SKIP_WEEKS * 5
    ma_days    = MA_WEEKS * 5
    rebal_days = REBAL_WEEKS * 5

    weights     = pd.DataFrame(0.0, index=dates, columns=tickers)
    entry_price = pd.Series(np.nan, index=tickers)
    current_pos = pd.Series(0.0,   index=tickers)

    for i in range(lb_days, n):
        today   = close.iloc[i]
        vix_val = float(vix[i])

        # VIX regime scalar
        if vix_val < 20.0:
            vix_scale = 1.0
        elif vix_val < 30.0:
            vix_scale = 0.6
        else:
            vix_scale = 0.2

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
            ma           = close.iloc[max(0, i - ma_days):i].mean()
            above_ma     = today > ma
            mom_filtered = mom.where(above_ma).dropna()

            if not mom_filtered.empty:
                n_top       = max(1, int(len(mom_filtered) * TOP_PCT))
                top_tickers = mom_filtered.nlargest(n_top).index

                vol          = close.iloc[max(0, i - 60):i][top_tickers].pct_change().std()
                inv_vol      = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
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

        # Apply VIX scaling daily (weights output, not position tracking)
        weights.iloc[i] = current_pos * vix_scale

    return weights

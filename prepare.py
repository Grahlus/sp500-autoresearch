#!/usr/bin/env python3
"""
prepare.py — FROZEN. Do not edit.
Provides: load_data(), run_backtest(), print_metrics()

Execution model:
  Signal generated at close[T]  →  filled at open[T+1]  →  exited at open[T+2]
  holding_ret[T+1] = open[T+2] / open[T+1] - 1
  No lookahead: generate_signals() never sees open[T+1] when deciding at close[T].

Backtest: daily rebalance, long/short, equal-notional, 5bps round-trip cost.
"""
import signal, sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
EXPERIMENT_TIMEOUT_SECS = 60 * 60  # 60 minutes hard cap


# ── Timeout guard ─────────────────────────────────────────────────────────────
def _timeout_handler(signum, frame):
    print("\n[TIMEOUT] Experiment exceeded time budget. Aborting.", flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(EXPERIMENT_TIMEOUT_SECS)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_data() -> dict:
    """
    Returns dict with keys:
      close, open, high, low, volume  — DataFrame[dates × tickers]
      vix                             — Series[dates]
      fear_greed                      — Series[dates]
      train_end                       — Timestamp (cutoff date)
      index                           — DatetimeIndex of all trading days
    """
    prices = pd.read_parquet(DATA_DIR / "sp500_prices.parquet")
    vix    = pd.read_parquet(DATA_DIR / "vix.parquet")
    fg     = pd.read_parquet(DATA_DIR / "fear_greed.parquet")

    close  = prices["close"]
    open_  = prices["open"]
    high   = prices["high"]
    low    = prices["low"]
    volume = prices["volume"]

    # Align macro series to trading calendar
    idx            = close.index
    vix_aligned    = vix["vix"].reindex(idx).ffill()
    fg_aligned     = fg["fg_value"].reindex(idx).ffill()

    # Train / validation / test split: fixed dates per program.md
    # Train: 2014-01-01 → 2022-06-30
    # Validation: 2022-07-01 → 2024-06-30  (~504 days, includes 2022 bear)
    # Test: 2024-07-01 → present           (hidden — evaluate.py only)
    cutoff  = pd.Timestamp("2022-07-01")
    val_end = pd.Timestamp("2024-07-01")
    print(f"  Universe  : {close.shape[1]} stocks")
    print(f"  Train     : {idx[0].date()} → {cutoff.date()}")
    print(f"  Validation: {cutoff.date()} → {val_end.date()} (~{(idx[(idx >= cutoff) & (idx < val_end)]).shape[0]} days)")

    return dict(
        close=close, open=open_, high=high, low=low, volume=volume,
        vix=vix_aligned, fear_greed=fg_aligned,
        train_end=cutoff, val_end=val_end, index=idx,
    )


# ── Backtest engine ───────────────────────────────────────────────────────────
def run_backtest(weights: pd.DataFrame, data: dict, split: str = "validation") -> dict:
    """
    weights : DataFrame aligned to data index, columns=tickers, values in [-1, 1]
              Positive = long, Negative = short. Rows need not sum to 1;
              gross exposure is normalised to 1.0 internally.
    split   : 'train' or 'validation'
    Returns : dict of performance metrics

    Execution timeline (no lookahead):
      Day T  : observe close[T], generate weight[T]
      Day T+1: enter at open[T+1]   ← execution price
      Day T+2: exit / rebalance at open[T+2]
      P&L[T+1] = weight[T] * (open[T+2] / open[T+1] - 1)
    """
    close   = data["close"]
    open_   = data["open"]
    cutoff  = data["train_end"]
    val_end = data.get("val_end", None)

    if split == "validation":
        if val_end is not None:
            close   = close.loc[cutoff:val_end]
            open_   = open_.loc[cutoff:val_end]
        else:
            close   = close.loc[cutoff:]
            open_   = open_.loc[cutoff:]
        weights = (
            weights.loc[cutoff:]
            if cutoff in weights.index
            else weights.iloc[-int(252 * 2):]
        )
        if val_end is not None:
            weights = weights.loc[:val_end]
    else:
        close   = close.loc[:cutoff]
        open_   = open_.loc[:cutoff]
        weights = weights.loc[:cutoff]

    # Align weights to the slice
    weights = (
        weights
        .reindex(close.index)
        .reindex(columns=close.columns)
        .fillna(0.0)
    )

    # ── Holding return: open[T+1] → open[T+2] ────────────────────────────────
    # open_.shift(-1) = next-day open  (used only in P&L calc, not in signal gen)
    holding_ret = (open_.shift(-1) / open_ - 1).fillna(0.0)

    # weight[T-1] is decided at close[T-1], filled at open[T]
    # so lag weights by 1 to avoid lookahead
    w_lag  = weights.shift(1).fillna(0.0)

    # Normalise: gross exposure capped at 1.0
    gross  = w_lag.abs().sum(axis=1).replace(0, 1)
    w_norm = w_lag.div(gross, axis=0)

    # Strategy daily P&L
    strat_ret = (w_norm * holding_ret).sum(axis=1)

    # Transaction costs: 5 bps per unit of turnover (round-trip)
    turnover   = w_norm.diff().abs().sum(axis=1)
    costs      = turnover * 0.0005
    strat_ret -= costs

    # Approximate trade count: days where any position changed × avg active positions
    ann_local = 252
    position_changes = (w_norm.diff().abs() > 1e-6).sum(axis=1)
    total_trades = int(position_changes.sum())
    trades_per_year = round(total_trades / (len(strat_ret) / ann_local), 1)

    # Benchmark: SPY open-to-open (consistent with strategy execution)
    if "SPY" in open_.columns:
        bench_ret = (open_["SPY"].shift(-1) / open_["SPY"] - 1).fillna(0.0)
    else:
        bench_ret = holding_ret.mean(axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    ann = 252

    def sharpe(r):
        return (r.mean() / r.std() * np.sqrt(ann)) if r.std() > 1e-9 else 0.0

    def calmar(r):
        cum     = (1 + r).cumprod()
        dd      = (cum / cum.cummax() - 1).min()
        ann_ret = (cum.iloc[-1] ** (ann / len(r)) - 1)
        return (ann_ret / abs(dd)) if dd < -1e-6 else 0.0

    def max_dd(r):
        cum = (1 + r).cumprod()
        return (cum / cum.cummax() - 1).min()

    strat_cum = (1 + strat_ret).cumprod()
    bench_cum = (1 + bench_ret).cumprod()
    total_ret = strat_cum.iloc[-1] - 1
    bench_tot = bench_cum.iloc[-1] - 1
    alpha_ann = (strat_ret - bench_ret).mean() * ann

    start_cap   = 100_000
    final_value = round(start_cap * strat_cum.iloc[-1], 0)
    total_cost  = round(costs.sum() * start_cap, 0)
    cost_pct    = round(total_cost / start_cap * 100, 2)

    metrics = {
        "split"           : split,
        "sharpe"          : round(sharpe(strat_ret), 3),
        "calmar"          : round(calmar(strat_ret), 3),
        "total_return"    : round(total_ret, 4),
        "bench_return"    : round(bench_tot, 4),
        "alpha_ann"       : round(alpha_ann, 4),
        "max_drawdown"    : round(max_dd(strat_ret), 4),
        "win_rate"        : round((strat_ret > 0).mean(), 3),
        "ann_vol"         : round(strat_ret.std() * np.sqrt(ann), 4),
        "n_days"          : len(strat_ret),
        "trades_per_year" : trades_per_year,
        "final_value"     : f"${final_value:,.0f}",
        "total_cost"      : f"${total_cost:,.0f}",
        "cost_pct_capital": cost_pct,
    }
    return metrics


def print_metrics(m: dict):
    print("\n── Backtest Results ─────────────────────────────")
    for k, v in m.items():
        print(f"  {k:<16}: {v}")
    print("─────────────────────────────────────────────────\n")

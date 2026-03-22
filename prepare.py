"""
prepare.py — FIXED. Do not modify.

Handles:
  - Loading NQ 1-min bar data from .pkl files
  - Train/validation split (time-based, OOS on Z5 contract)
  - Feature engineering utilities
  - Backtest engine with position tracking
  - Calmar ratio evaluation (the single metric)
  - 15-minute wall-clock experiment timeout

The agent only touches agent.py. This file is the ground truth.
"""

import os
import time
import signal
import importlib
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("NQ_DATA_DIR", "./data"))
EXPERIMENT_TIMEOUT_SECS = 15 * 60  # 15-minute hard wall clock limit

# Z5 contract starts 2025-09-01 — use it exclusively as the OOS validation set
# Train = H5 + M5 + U5 (Nov 2024 → Aug 2025)
# Val   = Z5           (Sep 2025 → Dec 2025)
VALIDATION_START = "2025-09-01"

# Backtest parameters
POINT_VALUE          = 20       # NQ: $20 per point
COMMISSION_PER_TRADE = 2.50     # per side, in USD
SLIPPAGE_POINTS      = 0.25     # 1 tick slippage each way
MAX_POSITION         = 1        # max 1 contract long or short

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load all NQ 1-min pkl files from DATA_DIR, return sorted DataFrame.

    Each pkl is a DataFrame with a DatetimeIndex named 'date' and columns:
    open, close, high, low, volume.

    Contracts overlap (e.g. H5 and M5 both have Feb 2025 bars).
    We deduplicate by keeping the front-month bar at each timestamp
    (earliest contract alphabetically = soonest expiry).
    """
    import pickle

    files = sorted(DATA_DIR.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {DATA_DIR}")

    frames = []
    for f in files:
        with open(f, "rb") as fh:
            df = pickle.load(fh)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "timestamp"
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["_contract"] = f.stem
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "_contract"])

    # Keep first contract alphabetically at each timestamp = front month
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    combined = combined.drop(columns=["_contract"])
    combined = combined.reset_index(drop=True)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return combined


def train_val_split(df: pd.DataFrame):
    """Split DataFrame into (train_df, val_df) by VALIDATION_START date."""
    cutoff = pd.Timestamp(VALIDATION_START)
    train = df[df["timestamp"] < cutoff].reset_index(drop=True)
    val   = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"[prepare] Train: {len(train):,} bars  "
          f"({train['timestamp'].iloc[0]} → {train['timestamp'].iloc[-1]})")
    print(f"[prepare] Val:   {len(val):,} bars  "
          f"({val['timestamp'].iloc[0]} → {val['timestamp'].iloc[-1]})")
    return train, val


# ─── FEATURE UTILITIES ────────────────────────────────────────────────────────

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of standard features. Agent may or may not use these."""
    df = df.copy()
    df["returns"]    = df["close"].pct_change()
    df["log_ret"]    = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"]   = df["high"] - df["low"]
    df["vwap_proxy"] = (df["high"] + df["low"] + df["close"]) / 3.0
    for w in [5, 15, 60, 240]:
        df[f"sma_{w}"]  = df["close"].rolling(w).mean()
        df[f"roc_{w}"]  = df["close"].pct_change(w)
        df[f"vol_{w}"]  = df["returns"].rolling(w).std()
    df["rsi_14"] = _rsi(df["close"], 14)
    df["atr_14"] = _atr(df, 14)
    return df.dropna().reset_index(drop=True)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift(1)).abs()
    lc  = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signals: np.ndarray) -> dict:
    """
    Vectorised backtest given a DataFrame and a signal array.

    signals: array of length len(df), values in {-1, 0, 1}
      +1 = long, -1 = short, 0 = flat
    """
    signals = np.clip(np.round(signals).astype(int), -MAX_POSITION, MAX_POSITION)
    closes  = df["close"].values
    n       = len(closes)

    equity      = np.zeros(n)
    position    = 0
    entry_price = 0.0
    cash        = 0.0

    for i in range(1, n):
        sig = signals[i - 1]

        if position != 0 and sig != position:
            pnl   = (closes[i] - entry_price) * position * POINT_VALUE
            cost  = (COMMISSION_PER_TRADE + SLIPPAGE_POINTS * POINT_VALUE)
            cash += pnl - cost
            position    = 0
            entry_price = 0.0

        if sig != 0 and position == 0:
            position    = sig
            entry_price = closes[i]
            cash       -= COMMISSION_PER_TRADE + SLIPPAGE_POINTS * POINT_VALUE

        open_pnl  = (closes[i] - entry_price) * position * POINT_VALUE if position else 0.0
        equity[i] = cash + open_pnl

    return {"equity": equity, "signals": signals, "closes": closes}


# ─── CALMAR RATIO ─────────────────────────────────────────────────────────────

def calmar_ratio(equity: np.ndarray, bars_per_year: int = 252 * 390) -> float:
    """
    Calmar = Annualised Return / Max Drawdown.
    Returns 0.0 for flat strategies, -999.0 on degenerate inputs.
    """
    if len(equity) < 2:
        return -999.0
    total_return = equity[-1] - equity[0]
    n_bars       = len(equity)
    ann_return   = total_return * (bars_per_year / n_bars)

    peak   = np.maximum.accumulate(equity)
    dd     = peak - equity
    max_dd = dd.max()

    if max_dd < 1e-6:
        return 0.0 if total_return < 1.0 else 999.0

    return ann_return / max_dd


# ─── EVALUATION ENTRY POINT ───────────────────────────────────────────────────

def evaluate_agent(agent_module) -> float:
    """Load agent, run on validation set, return Calmar ratio."""
    df_all        = load_data()
    _, val_df     = train_val_split(df_all)
    val_df_feat   = add_basic_features(val_df)

    signals = agent_module.get_signals(val_df_feat)

    if len(signals) != len(val_df_feat):
        raise ValueError(
            f"Agent returned {len(signals)} signals but val set has {len(val_df_feat)} bars"
        )

    result = run_backtest(val_df_feat, signals)
    return calmar_ratio(result["equity"])


# ─── TIMEOUT ──────────────────────────────────────────────────────────────────

class ExperimentTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise ExperimentTimeout(f"Experiment exceeded {EXPERIMENT_TIMEOUT_SECS}s")

class time_limited_experiment:
    def __enter__(self):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXPERIMENT_TIMEOUT_SECS)
        return self
    def __exit__(self, *args):
        signal.alarm(0)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_experiment() -> float:
    import sys

    print(f"\n{'='*60}")
    print(f"Experiment started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Timeout: {EXPERIMENT_TIMEOUT_SECS // 60} minutes")
    print(f"{'='*60}")

    start = time.time()

    try:
        with time_limited_experiment():
            if "agent" in sys.modules:
                del sys.modules["agent"]
            agent = importlib.import_module("agent")

            if hasattr(agent, "train"):
                df_all = load_data()
                train_df, _ = train_val_split(df_all)
                train_df_feat = add_basic_features(train_df)
                print("[prepare] Running agent.train() on training data...")
                agent.train(train_df_feat)

            print("[prepare] Evaluating on validation set...")
            score = evaluate_agent(agent)

    except ExperimentTimeout:
        print(f"[prepare] TIMEOUT after {EXPERIMENT_TIMEOUT_SECS}s")
        score = -999.0
    except Exception:
        print(f"[prepare] EXCEPTION:\n{traceback.format_exc()}")
        score = -999.0

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Calmar ratio (val): {score:.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return score


if __name__ == "__main__":
    df = load_data()
    train_df, val_df = train_val_split(df)
    print(f"\nTotal bars (deduped): {len(df):,}")
    print(f"Train: {len(train_df):,} bars | Val: {len(val_df):,} bars")
    print("\nData looks good. Ready to run experiments.")
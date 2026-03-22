"""
prepare.py — FIXED. Do not modify.

Handles:
  - Loading NQ 1-min bar data from .pkl files
  - Skipping first 4 weeks of each contract (low-volume back-month period)
  - Deduplication by proper futures expiry order (H < M < U < Z, year-aware)
  - Train/validation/forward-test split
      Train   = H5 + M5 + U5  (after 4-week skip, front-month bars only)
      Val     = Z5             (Sep 2025 → Dec 2025, OOS)
      Forward = H6             (Dec 2025 → Mar 2026, true future)
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

# Split boundaries
# Train        = H5 + M5 + U5  (Nov 2024 → Aug 2025, front-month only, 4-week warmup skipped)
# Validation   = Z5             (Sep 2025 → Dec 2025, OOS)
# Forward test = H6             (Dec 2025 rollover → Mar 2026, true future — not used in training loop)
VALIDATION_START   = "2025-09-01"
FORWARD_TEST_START = "2025-12-20"

# Backtest parameters
POINT_VALUE          = 20       # NQ: $20 per point
COMMISSION_PER_TRADE = 2.50     # per side, in USD
SLIPPAGE_POINTS      = 0.25     # 1 tick slippage each way
MAX_POSITION         = 1        # max 1 contract long or short

# ─── CONTRACT UTILITIES ───────────────────────────────────────────────────────

_MONTH_CODE = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}

def _contract_expiry_key(stem: str) -> tuple:
    """
    Parse contract expiry from filename stem for sorting.
    e.g. 'NQH5_TRADES_...' → month_code='H', year_digit='5'
    Returns (year_int, month_int) so contracts sort in expiry order.

    Year digit is relative: 5 → 2025, 6 → 2026 etc.
    H5 (Mar 2025) < M5 (Jun 2025) < U5 (Sep 2025) < Z5 (Dec 2025) < H6 (Mar 2026)
    """
    # Contract code is always at position 2-3 in stem (NQ + code)
    code  = stem[2:4]          # e.g. 'H5', 'Z5', 'H6'
    month = _MONTH_CODE.get(code[0], 99)
    year  = int(code[1]) if code[1].isdigit() else 99
    return (year, month)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load all NQ 1-min pkl files from DATA_DIR.

    Per-contract processing:
      - Skip first 4 weeks (low-volume back-month overlap period)
      - Tag with expiry sort key

    Deduplication:
      - At each timestamp, keep the bar from the front-month contract
        (lowest expiry key = soonest expiry)

    Returns a clean, sorted, deduplicated DataFrame covering the full
    date range across all contracts.
    """
    import pickle

    files = sorted(DATA_DIR.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {DATA_DIR}")

    frames = []
    for f in files:
        with open(f, "rb") as fh:
            df = pickle.load(fh)

        df.columns    = [c.lower() for c in df.columns]
        df.index.name = "timestamp"
        df            = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # ── Skip first 4 weeks (low-volume back-month period) ──
        warmup_end = df["timestamp"].min() + pd.Timedelta(weeks=4)
        df = df[df["timestamp"] >= warmup_end].copy()

        # ── Tag with expiry sort key for deduplication ──
        ey, em = _contract_expiry_key(f.stem)
        df["_expiry_year"]  = ey
        df["_expiry_month"] = em
        frames.append(df)

    if not frames:
        raise ValueError("All contract files were empty after 4-week warmup skip.")

    combined = pd.concat(frames, ignore_index=True)

    # Sort by timestamp first, then by expiry (front month = lowest key first)
    combined = combined.sort_values(["timestamp", "_expiry_year", "_expiry_month"])

    # Keep front-month bar at each timestamp
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    combined = combined.drop(columns=["_expiry_year", "_expiry_month"])
    combined = combined.reset_index(drop=True)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing  = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return combined


def train_val_split(df: pd.DataFrame):
    """
    Split into (train_df, val_df) using VALIDATION_START boundary.
    Forward test data (H6) is excluded from both — use load_forward_test() separately.
    """
    t_val     = pd.Timestamp(VALIDATION_START)
    t_forward = pd.Timestamp(FORWARD_TEST_START)

    train = df[df["timestamp"] < t_val].reset_index(drop=True)
    val   = df[(df["timestamp"] >= t_val) & (df["timestamp"] < t_forward)].reset_index(drop=True)

    print(f"[prepare] Train: {len(train):,} bars  "
          f"({train['timestamp'].iloc[0]} → {train['timestamp'].iloc[-1]})")
    print(f"[prepare] Val:   {len(val):,} bars  "
          f"({val['timestamp'].iloc[0]} → {val['timestamp'].iloc[-1]})")
    return train, val


def load_forward_test() -> pd.DataFrame:
    """
    Return H6 forward test data (Dec 2025 → Mar 2026).
    Not used in the training loop — for manual champion validation only.
    """
    df     = load_data()
    t_fwd  = pd.Timestamp(FORWARD_TEST_START)
    fwd_df = df[df["timestamp"] >= t_fwd].reset_index(drop=True)
    print(f"[prepare] Forward: {len(fwd_df):,} bars  "
          f"({fwd_df['timestamp'].iloc[0]} → {fwd_df['timestamp'].iloc[-1]})")
    return fwd_df


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
    df["rsi_14"]  = _rsi(df["close"], 14)
    df["atr_14"]  = _atr(df, 14)
    return df.dropna().reset_index(drop=True)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signals: np.ndarray) -> dict:
    """
    Vectorised backtest.
    signals: array of len(df), values in {-1, 0, 1}
      +1 = long, -1 = short, 0 = flat
    """
    signals     = np.clip(np.round(signals).astype(int), -MAX_POSITION, MAX_POSITION)
    closes      = df["close"].values
    n           = len(closes)
    equity      = np.zeros(n)
    position    = 0
    entry_price = 0.0
    cash        = 0.0

    for i in range(1, n):
        sig = signals[i - 1]

        # Close if direction changes or goes flat
        if position != 0 and sig != position:
            pnl   = (closes[i] - entry_price) * position * POINT_VALUE
            cost  = COMMISSION_PER_TRADE + SLIPPAGE_POINTS * POINT_VALUE
            cash += pnl - cost
            position    = 0
            entry_price = 0.0

        # Open new position
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
    """Load agent, run on validation set (Z5), return Calmar ratio."""
    df_all      = load_data()
    _, val_df   = train_val_split(df_all)
    val_df_feat = add_basic_features(val_df)

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


# ─── FORWARD TEST EVALUATION ─────────────────────────────────────────────────

# Minimum Calmar on H6 forward test to consider a strategy generalizable.
# A strategy that scores well on Z5 but below this on H6 is considered overfit.
H6_MIN_CALMAR = 0.6

def evaluate_agent_forward(agent_module) -> float:
    """Run agent on H6 forward test set, return Calmar ratio."""
    fwd_df      = load_forward_test()
    fwd_df_feat = add_basic_features(fwd_df)

    signals = agent_module.get_signals(fwd_df_feat)

    if len(signals) != len(fwd_df_feat):
        raise ValueError(
            f"Agent returned {len(signals)} signals but forward set has {len(fwd_df_feat)} bars"
        )

    result = run_backtest(fwd_df_feat, signals)
    return calmar_ratio(result["equity"])


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_experiment() -> float:
    """
    Run one full experiment:
      1. Train agent on train set (if agent.train() exists)
      2. Evaluate on Z5 validation set  -> z5_calmar
      3. Evaluate on H6 forward test    -> h6_calmar
      4. Print a clear KEEP / REVERT verdict

    KEEP only if:
      - z5_calmar improved over current champion  (agent tracks this)
      - h6_calmar >= H6_MIN_CALMAR               (generalization gate)

    Returns z5_calmar for the agent to compare against its best.
    """
    import sys

    print(f"\n{'='*60}")
    print(f"Experiment started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Timeout: {EXPERIMENT_TIMEOUT_SECS // 60} minutes")
    print(f"H6 minimum to keep: {H6_MIN_CALMAR}")
    print(f"{'='*60}")

    start    = time.time()
    z5_score = -999.0
    h6_score = -999.0

    try:
        with time_limited_experiment():
            if "agent" in sys.modules:
                del sys.modules["agent"]
            agent = importlib.import_module("agent")

            # Train
            if hasattr(agent, "train"):
                df_all = load_data()
                train_df, _ = train_val_split(df_all)
                train_df_feat = add_basic_features(train_df)
                print("[prepare] Running agent.train() on training data...")
                agent.train(train_df_feat)

            # Z5 validation
            print("[prepare] Evaluating on validation set (Z5)...")
            z5_score = evaluate_agent(agent)

            # H6 forward test
            print("[prepare] Evaluating on forward test (H6)...")
            h6_score = evaluate_agent_forward(agent)

    except ExperimentTimeout:
        print(f"[prepare] TIMEOUT after {EXPERIMENT_TIMEOUT_SECS}s")
    except Exception:
        print(f"[prepare] EXCEPTION:\n{traceback.format_exc()}")

    elapsed    = time.time() - start
    passes_h6  = h6_score >= H6_MIN_CALMAR
    verdict    = "KEEP (if Z5 improved)" if passes_h6 else "REVERT -- H6 overfit"

    print(f"\n{'='*60}")
    print(f"Calmar Z5  (val):     {z5_score:.4f}")
    print(f"Calmar H6  (forward): {h6_score:.4f}  [min required: {H6_MIN_CALMAR}]")
    print(f"Verdict:              {verdict}")
    print(f"Elapsed:              {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return z5_score


if __name__ == "__main__":
    print("=== Data split summary ===")
    df = load_data()
    train_df, val_df = train_val_split(df)
    fwd_df = load_forward_test()
    print(f"\nTotal bars (deduped, 4-week warmup skipped): {len(df):,}")
    print(f"  Train (H5+M5+U5): {len(train_df):,} bars")
    print(f"  Val   (Z5):       {len(val_df):,} bars")
    print(f"  Forward (H6):     {len(fwd_df):,} bars")
    print("\nReady to run experiments.")
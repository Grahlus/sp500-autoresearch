#!/usr/bin/env python3
"""
ml_signal.py — ML-based momentum ranking for SP500 strategy.

Replaces naive percentile rank with XGBoost/RF/LightGBM predicted forward return.
Strict walk-forward: model trained on [train_s, train_e], predicts on test window.
No lookahead — features use only data available at prediction time.

Features per (date, ticker):
  mom_26w, mom_13w, mom_4w, mom_52w   — multi-period price momentum
  ma_ratio                             — price vs 100d MA
  vol_ratio                            — 20d vs 60d volume trend
  realized_vol                         — 20d annualized volatility
  rsi_14                               — RSI(14)
  above_ma                             — binary: price > 100d MA
  vix                                  — market vol regime (scalar)
  fg                                   — Fear & Greed (scalar)
  breadth                              — % stocks above 200d MA (scalar)

Target: forward 20-trading-day return.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    'mom_26w', 'mom_13w', 'mom_4w', 'mom_52w',
    'ma_ratio', 'vol_ratio', 'realized_vol', 'rsi_14',
    'above_ma', 'vix', 'fg', 'breadth',
]

SAMPLE_EVERY = 10      # sample training rows every N trading days
FWD_DAYS     = 20      # forward return horizon for training target
MIN_HISTORY  = 260     # minimum trading days needed before first feature


def precompute_features(close: pd.DataFrame, volume: pd.DataFrame,
                        vix_s: pd.Series, fg_s: pd.Series) -> dict:
    """
    Precompute all rolling features for entire dataset (vectorized).
    Returns dict of DataFrames/Series indexed by date.
    Compute once, index cheaply per date.
    """
    # ── Price momentum (skip=15 to match S10-005 signal) ─────────────────────
    f = {}
    f['mom_26w'] = close.shift(15) / close.shift(130) - 1     # ~6 months, skip 3w
    f['mom_13w'] = close.shift(15) / close.shift(65)  - 1     # ~3 months, skip 3w
    f['mom_4w']  = close.shift(15) / close.shift(20)  - 1     # ~1 month, skip 3w
    f['mom_52w'] = close           / close.shift(252)  - 1    # 52w, no skip

    # ── MA ratio and above-MA flag ────────────────────────────────────────────
    ma_100 = close.rolling(100, min_periods=50).mean()
    f['ma_ratio'] = close / ma_100.replace(0, np.nan) - 1
    f['above_ma'] = (close > ma_100).astype(float)

    # ── Volume ratio ──────────────────────────────────────────────────────────
    avg_vol_20 = volume.rolling(20,  min_periods=10).mean()
    avg_vol_60 = volume.rolling(60,  min_periods=30).mean()
    f['vol_ratio'] = avg_vol_20 / avg_vol_60.replace(0, np.nan)

    # ── Realized volatility (annualized) ─────────────────────────────────────
    f['realized_vol'] = (close.pct_change()
                              .rolling(20, min_periods=10)
                              .std() * np.sqrt(252))

    # ── RSI(14) ───────────────────────────────────────────────────────────────
    delta  = close.diff()
    gains  = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    losses = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs     = gains / losses.replace(0, np.nan)
    f['rsi_14'] = 100 - 100 / (1 + rs)

    # ── Market-level (scalar per date) ────────────────────────────────────────
    ma_200   = close.rolling(200, min_periods=100).mean()
    f['breadth'] = (close > ma_200).mean(axis=1)   # Series

    # Defensive: parquet may return 1-column DataFrame instead of Series
    _vix = vix_s if isinstance(vix_s, pd.Series) else vix_s.iloc[:, 0]
    _fg  = fg_s  if isinstance(fg_s,  pd.Series) else fg_s.iloc[:,  0]
    f['vix'] = _vix.reindex(close.index).ffill().fillna(20.0)
    f['fg']  = _fg.reindex(close.index).ffill().fillna(50.0)

    return f


def build_training_matrix(close: pd.DataFrame, features: dict,
                           train_start: str, train_end: str) -> tuple:
    """
    Build (X, y, feature_names) for the given training window.

    Samples every SAMPLE_EVERY trading days to reduce autocorrelation.
    Target = forward FWD_DAYS return for each stock.
    Drops NaN rows automatically.

    Returns:
        X: ndarray (n_samples, n_features)
        y: ndarray (n_samples,)
        n_dates: number of sample dates used
    """
    idx   = close.index
    ts    = idx[idx >= train_start][0]
    te    = idx[idx <= train_end][-1]

    # Sample dates within training window, with minimum history
    date_locs = [i for i, d in enumerate(idx)
                 if ts <= d <= te
                 and i >= MIN_HISTORY
                 and i % SAMPLE_EVERY == 0
                 and i + FWD_DAYS < len(idx)]

    if not date_locs:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), 0

    stock_cols = ['mom_26w', 'mom_13w', 'mom_4w', 'mom_52w',
                  'ma_ratio', 'vol_ratio', 'realized_vol', 'rsi_14', 'above_ma']

    all_X, all_y = [], []

    for d_loc in date_locs:
        fwd_loc = d_loc + FWD_DAYS

        # Forward return target
        fwd_ret = (close.iloc[fwd_loc] / close.iloc[d_loc] - 1)

        # Stock-level features: (n_tickers,) → DataFrame
        feat_df = pd.DataFrame({
            col: features[col].iloc[d_loc]
            for col in stock_cols
        })

        # Market-level scalars (broadcast)
        feat_df['vix']     = float(features['vix'].iloc[d_loc])
        feat_df['fg']      = float(features['fg'].iloc[d_loc])
        feat_df['breadth'] = float(features['breadth'].iloc[d_loc])

        feat_df['_target'] = fwd_ret

        # Drop NaN/Inf rows
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()
        if feat_df.empty:
            continue

        all_X.append(feat_df[FEATURE_COLS].values)
        all_y.append(feat_df['_target'].values)

    if not all_X:
        return np.empty((0, len(FEATURE_COLS))), np.empty(0), 0

    X = np.vstack(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    return X, y, len(date_locs)


def predict_rank(close: pd.DataFrame, features: dict,
                 model, date_loc: int, tickers_filter) -> pd.Series:
    """
    Predict forward returns for all tickers at `date_loc`, return as Series.
    Uses only data available at `date_loc` (no lookahead).

    Args:
        tickers_filter: index of filtered tickers to rank (those passing MA/vol)
    Returns:
        pd.Series of predicted rank (pct=True) indexed by ticker
    """
    stock_cols = ['mom_26w', 'mom_13w', 'mom_4w', 'mom_52w',
                  'ma_ratio', 'vol_ratio', 'realized_vol', 'rsi_14', 'above_ma']

    feat_df = pd.DataFrame({
        col: features[col].iloc[date_loc]
        for col in stock_cols
    }, index=close.columns)

    feat_df['vix']     = float(features['vix'].iloc[date_loc])
    feat_df['fg']      = float(features['fg'].iloc[date_loc])
    feat_df['breadth'] = float(features['breadth'].iloc[date_loc])

    # Only predict for filtered tickers
    feat_df = feat_df.loc[feat_df.index.intersection(tickers_filter)]
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    # For rows with NaN: fallback to mom_26w rank (same as baseline)
    missing = feat_df[feat_df.isnull().any(axis=1)].index
    valid   = feat_df.dropna()

    if valid.empty:
        # Complete fallback: return baseline momentum rank
        mom = features['mom_26w'].iloc[date_loc].loc[tickers_filter]
        return mom.rank(pct=True)

    X_pred = valid[FEATURE_COLS].values.astype(np.float32)
    preds  = model.predict(X_pred)
    pred_s = pd.Series(preds, index=valid.index)

    # Fallback for NaN tickers using mom_26w
    if len(missing) > 0:
        mom_fb = features['mom_26w'].iloc[date_loc].loc[missing]
        # Scale fallback to similar range as predictions
        fb_scaled = mom_fb.rank(pct=True) * pred_s.std() + pred_s.mean()
        pred_s = pd.concat([pred_s, fb_scaled])

    return pred_s.rank(pct=True)

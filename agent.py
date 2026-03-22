"""
agent.py — THIS FILE IS EDITED BY THE AGENT. Humans do not touch this.

Exp 069: GBM with 480-bar horizon, structured features: calendar + trend-state + vol-state.
Prior GBM attempts (030, 058) failed with next-bar/short-horizon targets and raw price lags.
This uses:
  - Features: dow, hour (sin/cos encoded), trend_state = sign(EMA3 - EMA480), vol_state = sign(vol60 - vol240)
  - Target: 480-bar-ahead return direction (binary: up/down)
  - Model: GBM n_estimators=30, max_depth=3 (lightweight)
Hypothesis: ML can learn the optimal combination of the known-good signals
(calendar + trend-state + volume-state) better than any single threshold rule.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

_model = None
_feature_cols = ["dow_cos", "dow_sin", "hour_cos", "hour_sin", "trend_state", "vol_state"]
HORIZON = 480


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dow = df["timestamp"].dt.dayofweek
    hour = df["timestamp"].dt.hour
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    vwap = df["vwap_proxy"]
    ema_fast = vwap.ewm(span=3, adjust=False).mean()
    ema_slow = vwap.ewm(span=480, adjust=False).mean()
    df["trend_state"] = np.sign(ema_fast - ema_slow)
    vol60 = df["volume"].rolling(60).mean()
    vol240 = df["volume"].rolling(240).mean()
    df["vol_state"] = np.sign(vol60 - vol240)
    return df


def train(df: pd.DataFrame) -> None:
    global _model
    df2 = _add_features(df)
    df2["target"] = (df2["close"].shift(-HORIZON) > df2["close"]).astype(int)
    df2 = df2.dropna(subset=_feature_cols + ["target"])
    X = df2[_feature_cols].values
    y = df2["target"].values
    _model = GradientBoostingClassifier(
        n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42
    )
    _model.fit(X, y)
    importances = dict(zip(_feature_cols, _model.feature_importances_.round(3)))
    print(f"[agent] GBM trained on {len(X):,} samples, horizon={HORIZON}")
    print(f"[agent] Feature importances: {importances}")


def get_signals(df: pd.DataFrame) -> np.ndarray:
    """Long when GBM predicts P(up in 480 bars) > 0.52."""
    df2 = _add_features(df)
    X = df2[_feature_cols].fillna(0).values
    proba = _model.predict_proba(X)[:, 1]
    signals = np.where(proba > 0.52, 1, 0)
    return signals.astype(int)

from __future__ import annotations

import pandas as pd

from .base import StrategyFamily
from .fear_greed_overlay import _fear_greed_series
from .momentum import generate_signals_with_config as generate_momentum_signals

METRIC = "sharpe"
HYPOTHESIS = (
    "momentum_fear_greed_overlay: Use Fear & Greed only as a sleeve-level "
    "exposure scaler on top of the existing momentum strategy. The underlying "
    "momentum stock selection and exits are unchanged."
)

MOMENTUM_KEYS = {
    "LOOKBACK_WEEKS",
    "SKIP_WEEKS",
    "REBAL_WEEKS",
    "TOP_PCT",
    "MA_WEEKS",
    "STOP_LOSS_PCT",
    "STOP_PARABOLIC",
    "STOP_TYPE",
    "INV_VOL_DAYS",
    "MIN_HOLD_DAYS",
    "FG_MIN",
    "EXIT_PCT_RANK",
    "RANK_EXIT_CONFIRM",
}


def _confirmed_signal(signal: pd.Series, days: int) -> pd.Series:
    if days <= 1:
        return signal.fillna(False)
    return (
        signal.fillna(False)
        .rolling(days, min_periods=days)
        .sum()
        .ge(days)
    )


def _scaling_series(data: dict, config: dict, index: pd.Index) -> pd.Series:
    fear_greed = _fear_greed_series(data).reindex(index).ffill()

    entry_threshold = float(config.get("entry_threshold", 25.0))
    exit_threshold = float(config.get("exit_threshold", 55.0))
    greed_threshold = float(config.get("greed_threshold", 75.0))
    fear_exposure = float(config.get("fear_exposure", 0.75))
    normal_exposure = float(config.get("normal_exposure", 1.00))
    greed_exposure = float(config.get("greed_exposure", 0.75))
    confirmation_days = int(config.get("confirmation_days", 1))

    enter_signal = _confirmed_signal(fear_greed <= entry_threshold, confirmation_days)
    exit_signal = _confirmed_signal(fear_greed >= exit_threshold, confirmation_days)

    in_fear_regime = False
    multipliers: list[float] = []
    for i, fg_value in enumerate(fear_greed):
        if pd.isna(fg_value):
            in_fear_regime = False
        elif not in_fear_regime and bool(enter_signal.iloc[i]):
            in_fear_regime = True
        elif in_fear_regime and bool(exit_signal.iloc[i]):
            in_fear_regime = False

        if in_fear_regime:
            multiplier = fear_exposure
        elif not pd.isna(fg_value) and float(fg_value) >= greed_threshold:
            multiplier = greed_exposure
        else:
            multiplier = normal_exposure
        multipliers.append(float(min(max(multiplier, 0.0), 1.0)))

    return pd.Series(multipliers, index=index)


def _momentum_config(config: dict) -> dict:
    return {key: value for key, value in (config or {}).items() if key in MOMENTUM_KEYS}


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    base_weights = generate_momentum_signals(data, _momentum_config(config))
    base_weights = base_weights.reindex(index=close.index, columns=close.columns, fill_value=0.0)
    scaler = _scaling_series(data, config or {}, base_weights.index)
    return base_weights.mul(scaler, axis=0)


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="momentum_fear_greed_overlay",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

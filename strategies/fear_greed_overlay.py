from __future__ import annotations

import pandas as pd

from .base import StrategyFamily

METRIC = "sharpe"
HYPOTHESIS = (
    "fear_greed_contrarian_overlay_v2: Fear & Greed is used only as a market "
    "exposure regime signal. Enter partial SPY exposure after fear/extreme-fear "
    "readings and exit to cash as sentiment normalizes. Edge source: sentiment "
    "regime overlay, not stock selection."
)


def _fear_greed_series(data: dict) -> pd.Series:
    fear_greed = data["fear_greed"]
    if isinstance(fear_greed, pd.DataFrame):
        if "fg_value" in fear_greed.columns:
            return fear_greed["fg_value"]
        return fear_greed.iloc[:, 0]
    return fear_greed


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    fear_greed = _fear_greed_series(data).reindex(close.index).ffill()

    entry_threshold = float(config.get("entry_threshold", 25.0))
    exit_threshold = float(config.get("exit_threshold", 55.0))
    exposure = float(config.get("exposure", 0.75))
    confirmation_days = int(config.get("confirmation_days", 1))
    market_symbol = str(config.get("market_symbol", "SPY"))

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if market_symbol not in weights.columns:
        return weights

    enter_signal = (fear_greed <= entry_threshold).fillna(False)
    exit_signal = (fear_greed >= exit_threshold).fillna(False)
    if confirmation_days > 1:
        enter_signal = enter_signal.rolling(
            confirmation_days,
            min_periods=confirmation_days,
        ).sum() >= confirmation_days
        exit_signal = exit_signal.rolling(
            confirmation_days,
            min_periods=confirmation_days,
        ).sum() >= confirmation_days

    in_market = False
    symbol_idx = weights.columns.get_loc(market_symbol)
    for i, fg_value in enumerate(fear_greed):
        if pd.isna(fg_value):
            in_market = False
        elif not in_market and bool(enter_signal.iloc[i]):
            in_market = True
        elif in_market and bool(exit_signal.iloc[i]):
            in_market = False

        if in_market:
            weights.iat[i, symbol_idx] = exposure

    return weights


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="fear_greed_contrarian_overlay",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

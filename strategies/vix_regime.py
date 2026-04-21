from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import StrategyFamily
from .sector_map import load_sector_map

METRIC = "sharpe"
HYPOTHESIS = (
    "vix_regime_sector_tilt_v1: Sector rotation driven by VIX level. "
    "Low VIX → cyclicals (Tech/Discretionary/Industrials). "
    "High VIX → defensives (Utilities/Staples/Health Care). "
    "Mid VIX → all sectors equal-weight. "
    "Edge source: volatility regime, not stock return rank."
)

_CYCLICAL_NARROW = frozenset({"Information Technology", "Consumer Discretionary", "Industrials"})
_CYCLICAL_BROAD = frozenset({
    "Information Technology", "Consumer Discretionary", "Industrials",
    "Financials", "Communication Services",
})
_DEFENSIVE = frozenset({"Utilities", "Consumer Staples", "Health Care"})


def _load_sector_map() -> pd.Series:
    return load_sector_map()


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    vix = data["vix"]

    low_vix = float(config.get("low_vix_threshold", 18.0))
    high_vix = float(config.get("high_vix_threshold", 28.0))
    vix_smooth = int(config.get("vix_smooth_days", 5))
    cyclical_mode = str(config.get("cyclical_sectors_mode", "narrow"))

    cyclical_set = _CYCLICAL_BROAD if cyclical_mode == "broad" else _CYCLICAL_NARROW
    sector_map = _load_sector_map()
    universe = close.columns

    cyclical_stocks = [t for t in universe if sector_map.get(t, "") in cyclical_set]
    defensive_stocks = [t for t in universe if sector_map.get(t, "") in _DEFENSIVE]
    all_sector_stocks = [t for t in universe if t in sector_map.index]

    # vix may be a single-column DataFrame or a Series — normalise to Series
    vix_series = vix.iloc[:, 0] if isinstance(vix, pd.DataFrame) else vix
    vix_s = vix_series.rolling(vix_smooth, min_periods=1).mean()
    low_regime = (vix_s < low_vix).fillna(False)
    high_regime = (vix_s > high_vix).fillna(False)
    mid_regime = ~low_regime & ~high_regime & vix_s.notna()

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    if cyclical_stocks:
        weights.loc[low_regime, cyclical_stocks] = 1.0 / len(cyclical_stocks)
    if defensive_stocks:
        weights.loc[high_regime, defensive_stocks] = 1.0 / len(defensive_stocks)
    if all_sector_stocks:
        weights.loc[mid_regime, all_sector_stocks] = 1.0 / len(all_sector_stocks)

    return weights


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="vix_regime_sector_tilt",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import StrategyFamily
from .sector_map import load_sector_map

METRIC = "sharpe"
HYPOTHESIS = (
    "sector_breadth_overlay_v1: Allocate exposure from sector internal breadth, "
    "not stock return ranking. Broad risk-on, selective-on, and defensive regimes "
    "set SPY exposure levels; otherwise cash."
)

_DEFENSIVE_SECTORS = frozenset({"Consumer Staples", "Health Care", "Utilities"})


def _load_sector_map() -> pd.Series:
    return load_sector_map()


def _sector_breadth(close: pd.DataFrame, sector_map: pd.Series, window: int) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=window // 2).mean()
    above_ma = (close > ma).where(close.notna())
    sector_values: dict[str, pd.Series] = {}
    for sector in sorted(set(sector_map.dropna())):
        cols = [ticker for ticker in sector_map[sector_map == sector].index if ticker in close.columns]
        if not cols:
            continue
        sector_values[sector] = above_ma[cols].mean(axis=1)
    return pd.DataFrame(sector_values, index=close.index)


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    sector_map = _load_sector_map()

    breadth_window = int(config.get("breadth_window", 100))
    sector_on_threshold = float(config.get("sector_on_threshold", 0.55))
    sector_off_threshold = float(config.get("sector_off_threshold", 0.45))
    broad_sector_fraction = float(config.get("broad_sector_fraction", 0.55))
    risk_on_exposure = float(config.get("risk_on_exposure", 0.75))
    selective_exposure = float(config.get("selective_exposure", 0.50))
    defensive_exposure = float(config.get("defensive_exposure", 0.35))
    min_selective_sectors = int(config.get("min_selective_sectors", 3))
    market_symbol = str(config.get("market_symbol", "SPY"))

    sector_breadth = _sector_breadth(close, sector_map, breadth_window)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if sector_breadth.empty or market_symbol not in weights.columns:
        return weights
    market_col = weights.columns.get_loc(market_symbol)

    for i, (_, breadth_row) in enumerate(sector_breadth.iterrows()):
        valid = breadth_row.dropna()
        if valid.empty:
            continue

        strong_sectors = list(valid[valid >= sector_on_threshold].index)
        weak_fraction = float((valid <= sector_off_threshold).mean())
        strong_fraction = len(strong_sectors) / len(valid)
        defensive_strong = [
            sector
            for sector in strong_sectors
            if sector in _DEFENSIVE_SECTORS
        ]

        if strong_fraction >= broad_sector_fraction:
            weights.iat[i, market_col] = risk_on_exposure
        elif len(strong_sectors) >= min_selective_sectors and weak_fraction < 0.50:
            weights.iat[i, market_col] = selective_exposure
        elif defensive_strong:
            weights.iat[i, market_col] = defensive_exposure

    return weights


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="sector_breadth_overlay",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

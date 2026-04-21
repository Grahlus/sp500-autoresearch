from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BASE_SECTOR_PATH = Path("data/sp500_sectors.parquet")
ENRICHED_SECTOR_PATH = Path("data/sp500_sectors_enriched.parquet")

GICS_SECTOR_ALIASES = {
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Technology": "Information Technology",
    "Utilities": "Utilities",
}

GICS_SECTORS = frozenset({
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
})

EXCLUDED_SYMBOLS = frozenset({
    "COPX",
    "CPER",
    "DBA",
    "DBC",
    "GDX",
    "GDXJ",
    "GLD",
    "IWM",
    "PICK",
    "QQQ",
    "SLV",
    "SPY",
    "URA",
    "USD",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
})


def normalize_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def normalize_sector_label(label: Any) -> str | None:
    if label is None:
        return None
    try:
        if pd.isna(label):
            return None
    except (TypeError, ValueError):
        pass
    text = str(label).strip()
    if not text:
        return None
    return GICS_SECTOR_ALIASES.get(text, text)


def load_sector_frame(*, enriched_first: bool = True) -> pd.DataFrame:
    path = ENRICHED_SECTOR_PATH if enriched_first and ENRICHED_SECTOR_PATH.exists() else BASE_SECTOR_PATH
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    normalized = frame.copy()
    normalized.index = normalized.index.map(normalize_ticker)
    normalized.index.name = "Symbol"
    if "GICS Sector" in normalized.columns:
        normalized["GICS Sector"] = normalized["GICS Sector"].map(normalize_sector_label)
    return normalized[~normalized.index.duplicated(keep="last")]


def load_sector_map(*, enriched_first: bool = True) -> pd.Series:
    frame = load_sector_frame(enriched_first=enriched_first)
    if frame.empty or "GICS Sector" not in frame.columns:
        return pd.Series(dtype="object", name="GICS Sector")
    return frame["GICS Sector"].copy()


def is_gics_sector(value: Any) -> bool:
    return normalize_sector_label(value) in GICS_SECTORS


def is_excluded_symbol(symbol: str) -> bool:
    return normalize_ticker(symbol) in EXCLUDED_SYMBOLS or normalize_ticker(symbol).startswith("XL")


def resolve_sector_label(symbol: str, sector_map: pd.Series) -> str | None:
    ticker = normalize_ticker(symbol)
    if ticker in sector_map.index:
        label = normalize_sector_label(sector_map.get(ticker))
        if label in GICS_SECTORS:
            return label
        return label
    if is_excluded_symbol(ticker):
        return "Excluded / ETF"
    return None


def resolve_sector_universe(
    universe: Iterable[str],
    *,
    sector_map: pd.Series | None = None,
) -> tuple[pd.Series, pd.Index, pd.Index, pd.Index]:
    sector_map = sector_map if sector_map is not None else load_sector_map()
    labels: dict[str, str | None] = {}
    eligible: list[str] = []
    excluded: list[str] = []
    unresolved: list[str] = []

    for symbol in universe:
        ticker = normalize_ticker(symbol)
        label = resolve_sector_label(ticker, sector_map)
        labels[ticker] = label
        if label in GICS_SECTORS:
            eligible.append(ticker)
        elif label == "Excluded / ETF":
            excluded.append(ticker)
        else:
            unresolved.append(ticker)

    series = pd.Series(labels, dtype="object", name="sector")
    series.index.name = "Symbol"
    return series, pd.Index(eligible), pd.Index(excluded), pd.Index(unresolved)

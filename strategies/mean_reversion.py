from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .base import StrategyFamily
from .sector_map import GICS_SECTORS, load_sector_map, resolve_sector_universe

METRIC = "sharpe"
HYPOTHESIS = (
    "mr_sector_neutral_v2: Lower-turnover sector-relative mean reversion with mapped-only "
    "universe, liquidity/volatility filters, cooldown, and minimum hold. Edge source: "
    "short-horizon overreaction reversal in tradable equities, not momentum."
)


def _eligible_sector_mask(sector_map: pd.Series, columns: pd.Index) -> pd.Series:
    labels = sector_map.reindex(columns)
    return labels.map(lambda value: value in GICS_SECTORS).fillna(False)


def _generate(data: dict, config: dict) -> pd.DataFrame:
    close = data["close"]
    open_ = data["open"]
    volume = data["volume"]
    vix = data["vix"]

    lookback_days = int(config.get("lookback_days", 20))
    rebal_days = int(config.get("rebal_days", 20))
    max_positions = int(config.get("max_positions", 6))
    max_sector_positions = int(config.get("max_sector_positions", 1))
    min_hold_days = int(config.get("min_hold_days", 20))
    cooldown_days = int(config.get("cooldown_days", 20))
    confirm_rebalances = int(config.get("confirm_rebalances", 2))
    min_dollar_volume = float(config.get("min_dollar_volume", 3_000_000.0))
    max_daily_volatility = float(config.get("max_daily_volatility", 0.04))
    max_abs_open_gap = float(config.get("max_abs_open_gap", 0.08))
    dislocation_z = float(config.get("dislocation_z", 1.5))
    raw_vix_gate = config.get("vix_gate", 40.0)
    vix_gate = float(raw_vix_gate) if raw_vix_gate is not None else None

    sector_map = load_sector_map()
    sector_label, eligible_universe, excluded_universe, unresolved_universe = resolve_sector_universe(close.columns, sector_map=sector_map)
    eligible_columns = [ticker for ticker in close.columns if ticker in set(eligible_universe)]

    # vix may be a single-column DataFrame or a Series — normalise to Series
    vix_series = vix.iloc[:, 0] if isinstance(vix, pd.DataFrame) else vix

    # N-day returns (no lookahead: return[t] = close[t]/close[t-N] - 1)
    returns = close.pct_change(lookback_days)

    # Sector-demeaned returns: remove sector mean from each stock's return.
    sector_demeaned = returns.copy()
    for sector in sorted(GICS_SECTORS):
        cols = [t for t in eligible_columns if sector_label.get(t) == sector]
        if len(cols) < 2:
            continue
        sector_mean = returns[cols].mean(axis=1)
        sector_demeaned[cols] = returns[cols].sub(sector_mean, axis=0)

    # Liquidity/quality filters: keep only tradable, mapped equities.
    dollar_vol = (close * volume).rolling(20, min_periods=5).mean()
    realized_vol = close.pct_change().rolling(20, min_periods=10).std()
    open_gap = open_.div(close.shift(1)).sub(1.0).abs().rolling(20, min_periods=10).mean()
    liquid = dollar_vol >= min_dollar_volume
    tradable = liquid & (realized_vol <= max_daily_volatility) & (open_gap <= max_abs_open_gap)

    # Pre-extract numpy arrays for fast per-day weight assignment
    n_dates = len(close.index)
    n_tickers = len(close.columns)
    ticker_list = list(close.columns)
    ticker_pos = {t: i for i, t in enumerate(ticker_list)}

    weight_arr = np.zeros((n_dates, n_tickers), dtype=np.float64)
    current_arr = np.zeros(n_tickers, dtype=np.float64)
    current_positions: dict[str, int] = {}
    cooldown_until: dict[str, int] = {}
    candidate_streak: dict[str, int] = {}

    eligible_mask = _eligible_sector_mask(sector_label, close.columns)
    unresolved_count = int(unresolved_universe.size)
    excluded_count = int(excluded_universe.size)
    if not eligible_mask.any():
        return pd.DataFrame(weight_arr, index=close.index, columns=close.columns)

    for i in range(max(lookback_days, 20), n_dates):
        is_rebal = ((i - max(lookback_days, 20)) % rebal_days == 0)

        if is_rebal:
            vix_val = float(vix_series.iloc[i])
            flat_regime = vix_gate is not None and not np.isnan(vix_val) and vix_val > vix_gate
            score = sector_demeaned.iloc[i]
            liq_mask = tradable.iloc[i] & eligible_mask
            eligible = score[liq_mask].dropna()

            current_positions = {
                ticker: entry_idx
                for ticker, entry_idx in current_positions.items()
                if (i - entry_idx) < min_hold_days
                or (ticker in eligible.index and not flat_regime)
            }

            candidate_pool = pd.Series(dtype="float64")
            if not flat_regime and len(eligible) >= max_positions:
                row_mean = float(eligible.mean())
                row_std = float(eligible.std(ddof=0))
                dislocation_cutoff = row_mean - (dislocation_z * row_std)
                candidate_pool = eligible[eligible <= dislocation_cutoff].sort_values()
                if candidate_pool.empty:
                    candidate_pool = eligible.sort_values().head(max_positions * 2)

                candidate_set = set(candidate_pool.head(max_positions * 3).index)
                new_streak: dict[str, int] = {}
                for ticker in candidate_set:
                    new_streak[ticker] = candidate_streak.get(ticker, 0) + 1
                candidate_streak = new_streak

                sector_counts = Counter()
                selected: list[str] = []
                for ticker in candidate_pool.index:
                    sector = sector_label.get(ticker)
                    if sector not in GICS_SECTORS:
                        continue
                    if ticker in cooldown_until and i < cooldown_until[ticker]:
                        continue
                    if ticker in current_positions:
                        continue
                    if candidate_streak.get(ticker, 0) < confirm_rebalances:
                        continue
                    if sector_counts[sector] >= max_sector_positions:
                        continue
                    selected.append(ticker)
                    sector_counts[sector] += 1
                    if len(selected) >= max_positions:
                        break

                new_positions: dict[str, int] = {}
                for ticker, entry_idx in current_positions.items():
                    if ticker in candidate_pool.index or (i - entry_idx) < min_hold_days:
                        new_positions[ticker] = entry_idx
                    else:
                        cooldown_until[ticker] = i + cooldown_days
                for ticker in selected:
                    new_positions[ticker] = i
                current_positions = new_positions
            else:
                candidate_streak = {}
                # VIX gate or shallow signal: let existing positions age out, do not add new ones.
                current_positions = {
                    ticker: entry_idx
                    for ticker, entry_idx in current_positions.items()
                    if (i - entry_idx) < min_hold_days
                }

            if current_positions:
                current_arr = np.zeros(n_tickers)
                w = 1.0 / len(current_positions)
                for ticker in current_positions:
                    if ticker in ticker_pos:
                        current_arr[ticker_pos[ticker]] = w
            else:
                current_arr = np.zeros(n_tickers)

        weight_arr[i] = current_arr

    return pd.DataFrame(weight_arr, index=close.index, columns=close.columns)


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return _generate(data, config)


def generate_signals(data: dict) -> pd.DataFrame:
    return _generate(data, {})


def load() -> StrategyFamily:
    return StrategyFamily(
        name="mean_reversion",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

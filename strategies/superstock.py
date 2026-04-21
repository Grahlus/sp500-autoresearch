import pandas as pd

from .base import StrategyFamily
from .superstock_entries import build_superstock_breakout_entries
from .superstock_exits import build_superstock_exit_signals, build_superstock_position_mask
from .superstock_screen import build_superstock_screen
from .superstock_weekly import build_superstock_weekly_features

METRIC = "sharpe"
HYPOTHESIS = "Superstock v1: screened breakout entries with support/parabolic/HV exits"
MAX_POSITIONS = 5


def _build_weight_matrix(
    close: pd.DataFrame,
    positions: pd.DataFrame,
    entries,
    max_positions: int,
) -> pd.DataFrame:
    rs_rank_26w = entries.diagnostics.get("rs_rank_26w")
    rs_rank_52w = entries.diagnostics.get("rs_rank_52w")
    if not isinstance(rs_rank_26w, pd.DataFrame) or not isinstance(rs_rank_52w, pd.DataFrame):
        raise ValueError("Superstock entries must expose rs_rank_26w and rs_rank_52w diagnostics.")

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    max_positions = max(int(max_positions), 1)

    for dt in close.index:
        active = positions.loc[dt]
        if not active.any():
            continue

        selected = active[active].index
        if len(selected) > max_positions:
            ranked = pd.DataFrame(
                {
                    "rs_26w": rs_rank_26w.loc[dt, selected],
                    "rs_52w": rs_rank_52w.loc[dt, selected],
                }
            ).sort_values(["rs_26w", "rs_52w"], ascending=False)
            selected = ranked.head(max_positions).index

        weights.loc[dt, selected] = 1.0 / len(selected)

    return weights


def build_superstock_pipeline(
    data: dict,
    max_positions: int = MAX_POSITIONS,
    config: dict | None = None,
) -> dict[str, object]:
    config = dict(config or {})
    close = data["close"]
    resolved_max_positions = int(config.get("max_positions", max_positions))

    cache = data.setdefault("_grid_cache", {})
    weekly = cache.get("superstock_weekly")
    if weekly is None:
        weekly = build_superstock_weekly_features(data)
        cache["superstock_weekly"] = weekly
    screen = build_superstock_screen(data, weekly=weekly, config=config)
    entries = build_superstock_breakout_entries(data, weekly=weekly, screen=screen, config=config)
    exits = build_superstock_exit_signals(data, weekly=weekly, entries=entries, config=config)
    positions = build_superstock_position_mask(entries, exits, execution_lag_days=1)
    weights = _build_weight_matrix(close, positions, entries, resolved_max_positions)

    return {
        "weekly": weekly,
        "screen": screen,
        "entries": entries,
        "exits": exits,
        "positions": positions,
        "weights": weights,
        "config": config,
    }


def build_superstock_weights(
    data: dict,
    max_positions: int = MAX_POSITIONS,
    config: dict | None = None,
) -> pd.DataFrame:
    return build_superstock_pipeline(data, max_positions=max_positions, config=config)["weights"]


def generate_signals(data: dict) -> pd.DataFrame:
    return build_superstock_weights(data)


def generate_signals_with_config(data: dict, config: dict) -> pd.DataFrame:
    return build_superstock_weights(data, config=config)


def load() -> StrategyFamily:
    return StrategyFamily(
        name="superstock",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

from contextlib import contextmanager

import agent
from agent import HYPOTHESIS, METRIC, generate_signals

from .base import StrategyFamily


@contextmanager
def _temporary_agent_config(config: dict):
    overrides = {
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
        "HYPOTHESIS",
        "METRIC",
    }
    original = {}
    try:
        for key, value in (config or {}).items():
            if key in overrides and hasattr(agent, key):
                original[key] = getattr(agent, key)
                setattr(agent, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(agent, key, value)


def generate_signals_with_config(data: dict, config: dict) -> "pd.DataFrame":
    with _temporary_agent_config(config):
        return agent.generate_signals(data)


def load() -> StrategyFamily:
    return StrategyFamily(
        name="momentum",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
        generate_signals_with_config=generate_signals_with_config,
    )

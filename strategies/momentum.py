from agent import HYPOTHESIS, METRIC, generate_signals

from .base import StrategyFamily


def load() -> StrategyFamily:
    return StrategyFamily(
        name="momentum",
        metric=METRIC,
        hypothesis=HYPOTHESIS,
        generate_signals=generate_signals,
    )

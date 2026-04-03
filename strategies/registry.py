from .base import StrategyFamily
from .momentum import load as load_momentum
from .ml_ranker import load as load_ml_ranker
from .rl_bandit import load as load_rl_bandit
from .superstock import load as load_superstock


def get_strategy_family(name: str = "momentum") -> StrategyFamily:
    normalized = (name or "momentum").strip().lower()
    if normalized == "momentum":
        return load_momentum()
    if normalized == "superstock":
        return load_superstock()
    if normalized == "ml_ranker":
        return load_ml_ranker()
    if normalized == "rl_bandit":
        return load_rl_bandit()
    available = ", ".join(list_strategy_families())
    raise ValueError(f"Unknown strategy family '{name}'. Available: {available}")


def list_strategy_families() -> list[str]:
    return ["ml_ranker", "momentum", "rl_bandit", "superstock"]

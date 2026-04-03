from .base import StrategyFamily
from .momentum import load as load_momentum


def get_strategy_family(name: str = "momentum") -> StrategyFamily:
    normalized = (name or "momentum").strip().lower()
    if normalized == "momentum":
        return load_momentum()
    available = ", ".join(list_strategy_families())
    raise ValueError(f"Unknown strategy family '{name}'. Available: {available}")


def list_strategy_families() -> list[str]:
    return ["momentum"]

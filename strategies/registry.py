from .base import StrategyFamily
from .amihud_illiquidity import load as load_amihud_illiquidity
from .breadth_timing import load as load_breadth_timing
from .fear_greed_contrarian import load as load_fear_greed_contrarian
from .fear_greed_overlay import load as load_fear_greed_overlay
from .mean_reversion import load as load_mean_reversion
from .ml_ranker import load as load_ml_ranker
from .momentum_fear_greed_overlay import load as load_momentum_fear_greed_overlay
from .momentum import load as load_momentum
from .rl_bandit import load as load_rl_bandit
from .sector_breadth_overlay import load as load_sector_breadth_overlay
from .superstock import load as load_superstock
from .volatility_compression import load as load_volatility_compression
from .vix_regime import load as load_vix_regime


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
    if normalized == "breadth_timing_overlay":
        return load_breadth_timing()
    if normalized == "vix_regime_sector_tilt":
        return load_vix_regime()
    if normalized == "mean_reversion":
        return load_mean_reversion()
    if normalized == "momentum_fear_greed_overlay":
        return load_momentum_fear_greed_overlay()
    if normalized == "fear_greed_contrarian":
        return load_fear_greed_contrarian()
    if normalized == "fear_greed_contrarian_overlay":
        return load_fear_greed_overlay()
    if normalized == "amihud_illiquidity_premium":
        return load_amihud_illiquidity()
    if normalized == "sector_breadth_overlay":
        return load_sector_breadth_overlay()
    if normalized == "volatility_compression_expansion":
        return load_volatility_compression()
    available = ", ".join(list_strategy_families())
    raise ValueError(f"Unknown strategy family '{name}'. Available: {available}")


def list_strategy_families() -> list[str]:
    return [
        "breadth_timing_overlay",
        "amihud_illiquidity_premium",
        "fear_greed_contrarian",
        "fear_greed_contrarian_overlay",
        "mean_reversion",
        "ml_ranker",
        "momentum",
        "momentum_fear_greed_overlay",
        "rl_bandit",
        "sector_breadth_overlay",
        "superstock",
        "volatility_compression_expansion",
        "vix_regime_sector_tilt",
    ]

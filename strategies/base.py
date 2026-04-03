from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class StrategyFamily:
    name: str
    metric: str
    hypothesis: str
    generate_signals: Callable[[dict], pd.DataFrame]

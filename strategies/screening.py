from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScreeningResult:
    eligible: pd.DataFrame
    rule_masks: dict[str, pd.DataFrame]
    diagnostics: dict[str, pd.DataFrame | pd.Series | str | None]


def combine_masks(rule_masks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not rule_masks:
        raise ValueError("At least one rule mask is required.")

    combined = None
    for mask in rule_masks.values():
        normalized = mask.fillna(False).astype(bool)
        combined = normalized if combined is None else (combined & normalized)
    return combined

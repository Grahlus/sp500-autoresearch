from __future__ import annotations

import itertools
import random
from typing import Any


SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "momentum": {
        "LOOKBACK_WEEKS": {"name": "LOOKBACK_WEEKS", "type": "int", "default": 26, "choices": [20, 26, 39], "log_scale": False, "nullable": False},
        "SKIP_WEEKS": {"name": "SKIP_WEEKS", "type": "int", "default": 3, "choices": [2, 3, 4], "log_scale": False, "nullable": False},
        "REBAL_WEEKS": {"name": "REBAL_WEEKS", "type": "int", "default": 4, "choices": [2, 4], "log_scale": False, "nullable": False},
        "TOP_PCT": {"name": "TOP_PCT", "type": "float", "default": 0.025, "choices": [0.015, 0.025, 0.03], "log_scale": False, "nullable": False},
        "MA_WEEKS": {"name": "MA_WEEKS", "type": "int", "default": 20, "choices": [16, 20, 24], "log_scale": False, "nullable": False},
        "STOP_TYPE": {"name": "STOP_TYPE", "type": "categorical", "default": "adaptive", "choices": ["adaptive", "fixed", "none"], "log_scale": False, "nullable": False},
        "STOP_LOSS_PCT": {"name": "STOP_LOSS_PCT", "type": "float", "default": 0.20, "choices": [0.15, 0.20, 0.25, 0.30], "log_scale": False, "nullable": True},
        "STOP_PARABOLIC": {"name": "STOP_PARABOLIC", "type": "float", "default": 0.30, "choices": [0.25, 0.30, 0.35, 0.40], "log_scale": False, "nullable": True},
        "INV_VOL_DAYS": {"name": "INV_VOL_DAYS", "type": "int", "default": 15, "choices": [15], "log_scale": False, "nullable": False},
        "MIN_HOLD_DAYS": {"name": "MIN_HOLD_DAYS", "type": "int", "default": 5, "choices": [5, 10, 15], "log_scale": False, "nullable": False},
        "FG_MIN": {"name": "FG_MIN", "type": "float", "default": 10.0, "choices": [10.0, 22.0, 30.0], "log_scale": False, "nullable": False},
        "EXIT_PCT_RANK": {"name": "EXIT_PCT_RANK", "type": "float", "default": 0.97, "choices": [None, 0.94, 0.95, 0.97], "log_scale": False, "nullable": True},
        "RANK_EXIT_CONFIRM": {"name": "RANK_EXIT_CONFIRM", "type": "int", "default": None, "choices": [None, 1, 2], "log_scale": False, "nullable": True},
    },
    "superstock": {
        "max_positions": {"name": "max_positions", "type": "int", "default": 5, "choices": [3, 5, 8], "log_scale": False, "nullable": False},
        "price_min": {"name": "price_min", "type": "float", "default": 5.0, "choices": [3.0, 5.0], "log_scale": False, "nullable": False},
        "price_max": {"name": "price_max", "type": "float", "default": 15.0, "choices": [15.0, 20.0], "log_scale": False, "nullable": False},
        "min_dollar_volume": {"name": "min_dollar_volume", "type": "float", "default": 1_000_000.0, "choices": [500_000.0, 1_000_000.0, 2_000_000.0], "log_scale": False, "nullable": False},
        "above_52w_low_mult": {"name": "above_52w_low_mult", "type": "float", "default": 1.25, "choices": [1.25], "log_scale": False, "nullable": False},
        "near_52w_high_mult": {"name": "near_52w_high_mult", "type": "float", "default": 0.75, "choices": [0.75], "log_scale": False, "nullable": False},
        "rs_rank_26w_min": {"name": "rs_rank_26w_min", "type": "float", "default": 0.70, "choices": [0.60, 0.70, 0.80], "log_scale": False, "nullable": False},
        "rs_rank_52w_min": {"name": "rs_rank_52w_min", "type": "float", "default": 0.70, "choices": [0.60, 0.70, 0.80], "log_scale": False, "nullable": False},
        "base_depth_max": {"name": "base_depth_max", "type": "float", "default": 0.60, "choices": [0.40, 0.60, 0.80], "log_scale": False, "nullable": False},
        "weekly_range_median_max": {"name": "weekly_range_median_max", "type": "float", "default": 0.18, "choices": [0.15, 0.18, 0.22], "log_scale": False, "nullable": False},
        "weekly_volatility_max": {"name": "weekly_volatility_max", "type": "float", "default": 0.12, "choices": [0.10, 0.12, 0.15], "log_scale": False, "nullable": False},
        "volume_dryup_max": {"name": "volume_dryup_max", "type": "float", "default": 0.80, "choices": [0.80, 0.90, 1.00], "log_scale": False, "nullable": False},
        "vix_hard_cap": {"name": "vix_hard_cap", "type": "float", "default": 35.0, "choices": [30.0, 35.0, 40.0], "log_scale": False, "nullable": False},
        "vix_ma_multiplier": {"name": "vix_ma_multiplier", "type": "float", "default": 1.25, "choices": [1.10, 1.25, 1.50], "log_scale": False, "nullable": False},
        "breakout_extension_max": {"name": "breakout_extension_max", "type": "float", "default": 0.10, "choices": [0.08, 0.10, 0.15], "log_scale": False, "nullable": False},
        "daily_volume_expansion_mult": {"name": "daily_volume_expansion_mult", "type": "float", "default": 1.50, "choices": [1.25, 1.50, 2.00], "log_scale": False, "nullable": False},
        "daily_dollar_volume_expansion_mult": {"name": "daily_dollar_volume_expansion_mult", "type": "float", "default": 1.50, "choices": [1.25, 1.50, 2.00], "log_scale": False, "nullable": False},
        "parabolic_from_pivot_min": {"name": "parabolic_from_pivot_min", "type": "float", "default": 0.25, "choices": [0.20, 0.25, 0.30], "log_scale": False, "nullable": False},
        "parabolic_above_10w_min": {"name": "parabolic_above_10w_min", "type": "float", "default": 0.20, "choices": [0.15, 0.20, 0.25], "log_scale": False, "nullable": False},
        "late_stage_range_mult": {"name": "late_stage_range_mult", "type": "float", "default": 1.75, "choices": [1.50, 1.75, 2.00], "log_scale": False, "nullable": False},
        "late_stage_volume_mult": {"name": "late_stage_volume_mult", "type": "float", "default": 2.00, "choices": [1.50, 2.00, 2.50], "log_scale": False, "nullable": False},
    },
}


def list_searchable_families() -> list[str]:
    return sorted(SEARCH_SPACES)


def get_family_search_space(family: str) -> dict[str, dict]:
    normalized = family.strip().lower()
    if normalized not in SEARCH_SPACES:
        raise ValueError(f"Unknown experiment family '{family}'.")
    return SEARCH_SPACES[normalized]


def get_family_default_config(family: str) -> dict[str, Any]:
    space = get_family_search_space(family)
    return {name: spec["default"] for name, spec in space.items()}


def _coerce_value(spec: dict[str, Any], value: Any) -> Any:
    if value is None:
        if spec.get("nullable", False):
            return None
        raise ValueError(f"{spec['name']} does not allow null.")

    value_type = spec["type"]
    if value_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be an int.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ValueError(f"{spec['name']} must be an int.")
    if value_type == "float":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be a float.")
        try:
            return round(float(value), 6)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec['name']} must be a float.") from exc
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{spec['name']} must be a bool.")
    if value_type == "categorical":
        return value
    raise ValueError(f"Unsupported type '{value_type}' for {spec['name']}.")


def validate_experiment_config(family: str, config: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        normalize_experiment_config(family, config)
    except ValueError as exc:
        return False, str(exc)
    return True, None


def normalize_experiment_config(family: str, config: dict[str, Any]) -> dict[str, Any]:
    family = family.strip().lower()
    space = get_family_search_space(family)
    config = dict(config or {})

    unknown = sorted(set(config) - set(space))
    if unknown:
        raise ValueError(f"Unknown config key(s) for {family}: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    for name, spec in space.items():
        raw_value = config[name] if name in config else spec["default"]
        value = _coerce_value(spec, raw_value)

        choices = spec.get("choices")
        skip_choice_check = family == "superstock" and name in {"price_min", "price_max"}
        if choices is not None and not skip_choice_check and value not in choices:
            raise ValueError(f"{name} must be one of {choices}.")

        min_value = spec.get("min")
        max_value = spec.get("max")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}.")

        normalized[name] = value

    if family == "momentum":
        stop_type = normalized["STOP_TYPE"]
        if stop_type == "none":
            normalized["STOP_LOSS_PCT"] = None
            normalized["STOP_PARABOLIC"] = None
        elif stop_type == "fixed":
            normalized["STOP_PARABOLIC"] = None
        if normalized["EXIT_PCT_RANK"] is None:
            normalized["RANK_EXIT_CONFIRM"] = None

    if family == "superstock":
        price_min = float(normalized["price_min"])
        price_max = float(normalized["price_max"])
        normalized["price_min"] = min(price_min, price_max)
        normalized["price_max"] = max(price_min, price_max)

    return dict(sorted(normalized.items()))


def enumerate_grid_candidates(family: str, limit: int | None = None) -> list[dict[str, Any]]:
    space = get_family_search_space(family)
    keys = sorted(space)
    candidate_lists = [space[key]["choices"] for key in keys]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    for values in itertools.product(*candidate_lists):
        candidate = normalize_experiment_config(family, dict(zip(keys, values, strict=False)))
        token = tuple(sorted(candidate.items()))
        if token in seen:
            continue
        seen.add(token)
        candidates.append(candidate)
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def sample_random_candidates(family: str, n: int, seed: int) -> list[dict[str, Any]]:
    space = get_family_search_space(family)
    rng = random.Random(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    max_attempts = max(100, n * 20)
    attempts = 0

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        raw = {name: rng.choice(spec["choices"]) for name, spec in space.items()}
        candidate = normalize_experiment_config(family, raw)
        token = tuple(sorted(candidate.items()))
        if token in seen:
            continue
        seen.add(token)
        candidates.append(candidate)

    if len(candidates) < n:
        for candidate in enumerate_grid_candidates(family):
            token = tuple(sorted(candidate.items()))
            if token in seen:
                continue
            seen.add(token)
            candidates.append(candidate)
            if len(candidates) >= n:
                break
    return candidates


def normalize_experiment_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    return normalize_experiment_config(family, params)

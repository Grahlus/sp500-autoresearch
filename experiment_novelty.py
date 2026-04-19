from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from experiment_spaces import get_family_search_space, normalize_experiment_config


@dataclass(frozen=True)
class CandidateNovelty:
    exact_duplicate: bool
    near_duplicate: bool
    near_duplicate_of: str | None
    duplicate_risk: str | None
    novelty_score: float
    selection_score: float
    objective_proxy: float
    dead_zone_risk: float
    dead_zone_flags: list[str]


def _ordered_keys(family: str) -> list[str]:
    return list(get_family_search_space(family).keys())


def discrete_signature(family: str, config: dict[str, Any]) -> tuple[Any, ...]:
    space = get_family_search_space(family)
    normalized = normalize_experiment_config(family, config)
    signature: list[Any] = []
    for key in _ordered_keys(family):
        spec = space[key]
        value = normalized.get(key)
        choices = list(spec.get("choices") or [])
        if value is None:
            signature.append(None)
            continue
        if value in choices:
            signature.append(choices.index(value))
            continue
        if spec["type"] == "float":
            signature.append(round(float(value), 3))
        else:
            signature.append(value)
    return tuple(signature)


def signature_distance(family: str, left: dict[str, Any], right: dict[str, Any]) -> int:
    left_sig = discrete_signature(family, left)
    right_sig = discrete_signature(family, right)
    return sum(1 for a, b in zip(left_sig, right_sig, strict=False) if a != b)


def coarse_signature_key(family: str, config: dict[str, Any]) -> str:
    return json.dumps(discrete_signature(family, config), sort_keys=True, separators=(",", ":"))


def is_exact_duplicate(
    family: str,
    config: dict[str, Any],
    explored_hashes: set[str],
    *,
    config_hash: str | None = None,
) -> bool:
    if config_hash is None:
        from experiment_store import compute_config_hash

        config_hash = compute_config_hash(family, normalize_experiment_config(family, config))
    return config_hash in explored_hashes


def is_near_duplicate(
    family: str,
    config: dict[str, Any],
    explored_signatures: dict[str, str] | set[str],
    *,
    max_distance: int = 1,
) -> tuple[bool, str | None]:
    candidate = json.loads(json.dumps(discrete_signature(family, config), sort_keys=True, separators=(",", ":")))
    if isinstance(explored_signatures, set):
        candidate_key = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        if candidate_key in explored_signatures:
            return True, candidate_key
        return False, None

    best_match: str | None = None
    best_distance: int | None = None
    for config_hash, signature_text in explored_signatures.items():
        try:
            parsed = json.loads(signature_text)
        except Exception:
            continue
        distance = sum(1 for a, b in zip(candidate, parsed, strict=False) if a != b)
        if distance <= max_distance and (best_distance is None or distance < best_distance):
            best_distance = distance
            best_match = config_hash
    return (best_match is not None, best_match)


def _objective_from_history(
    history: list[dict[str, Any]],
    family: str,
    config: dict[str, Any],
    *,
    config_hash: str | None = None,
) -> float:
    if not history:
        return 0.0
    exact_matches = [row for row in history if config_hash is not None and row.get("config_hash") == config_hash]
    if exact_matches:
        scores = [float(row.get("objective_score") or 0.0) for row in exact_matches]
        return float(sum(scores) / len(scores))
    candidate_sig = discrete_signature(family, config)
    neighbors: list[tuple[int, float]] = []
    for row in history:
        row_config = row.get("config")
        if not isinstance(row_config, dict):
            continue
        distance = signature_distance(family, config, row_config)
        score = float(row.get("objective_score") or 0.0)
        neighbors.append((distance, score))
    if not neighbors:
        return 0.0
    neighbors.sort(key=lambda item: item[0])
    weighted = 0.0
    weight_total = 0.0
    for distance, score in neighbors[:5]:
        weight = 1.0 / float(distance + 1)
        weighted += score * weight
        weight_total += weight
    return weighted / weight_total if weight_total else 0.0


def score_candidate(
    family: str,
    config: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    explored_hashes: set[str],
    explored_signatures: dict[str, str] | set[str],
    dead_zone_hashes: set[str] | None = None,
    dead_zone_values: dict[str, set[Any]] | None = None,
    dead_zone_signatures: set[str] | None = None,
    source_type: str = "template_expansion",
    template_id: str | None = None,
    exploration_mode: str = "template_expansion",
    novelty_floor: float = 0.0,
    near_duplicate_distance: int = 1,
) -> CandidateNovelty | None:
    from experiment_store import compute_config_hash

    normalized = normalize_experiment_config(family, config)
    config_hash = compute_config_hash(family, normalized)
    if config_hash in explored_hashes or (dead_zone_hashes and config_hash in dead_zone_hashes):
        return None

    exact = is_exact_duplicate(family, normalized, explored_hashes, config_hash=config_hash)
    near, near_of = is_near_duplicate(family, normalized, explored_signatures, max_distance=near_duplicate_distance)
    duplicate_risk: str | None = None
    if exact:
        duplicate_risk = "exact"
    elif near:
        duplicate_risk = "near"

    objective_proxy = _objective_from_history(history, family, normalized, config_hash=config_hash)
    novelty_score = 0.0
    if source_type == "local_refinement":
        novelty_score += 0.20
    elif source_type == "idea_seed":
        novelty_score += 0.28
    elif source_type == "cross_family_hybrid":
        novelty_score += 0.45
    elif source_type == "template_expansion":
        novelty_score += 0.30
    elif source_type == "model_based":
        novelty_score += 0.55
    elif source_type == "policy_learning":
        novelty_score += 0.60
    elif source_type == "external_seed":
        novelty_score += 0.55
    elif source_type == "saturation_escape":
        novelty_score += 0.35
    if exploration_mode == "local_refinement":
        novelty_score += 0.10
    elif exploration_mode == "idea_seed":
        novelty_score += 0.15
    elif exploration_mode == "broader_exploration":
        novelty_score += 0.20
    elif exploration_mode == "saturation_escape":
        novelty_score += 0.25
    if near:
        novelty_score -= 0.40
    if exact:
        novelty_score = 0.0
    novelty_score = max(0.0, min(1.0, novelty_score))
    if novelty_score < novelty_floor:
        return None

    dead_zone_flags: list[str] = []
    dead_zone_risk = 0.0
    if dead_zone_values:
        for param, bad_values in dead_zone_values.items():
            if str(normalized.get(param)) in {str(value) for value in bad_values}:
                dead_zone_flags.append(f"{param}={normalized.get(param)}")
        if dead_zone_flags:
            dead_zone_risk = min(1.0, 0.25 * len(dead_zone_flags))

    if dead_zone_signatures and json.dumps(discrete_signature(family, normalized), sort_keys=True, separators=(",", ":")) in dead_zone_signatures:
        dead_zone_risk = 1.0

    selection_score = (0.55 * objective_proxy) + (0.45 * novelty_score) - (0.35 * dead_zone_risk)
    if source_type in {"idea_seed", "cross_family_hybrid", "external_seed", "model_based", "policy_learning"}:
        selection_score += 0.05
    if near:
        selection_score -= 0.20

    return CandidateNovelty(
        exact_duplicate=exact,
        near_duplicate=near,
        near_duplicate_of=near_of,
        duplicate_risk=duplicate_risk,
        novelty_score=round(novelty_score, 6),
        selection_score=round(selection_score, 6),
        objective_proxy=round(objective_proxy, 6),
        dead_zone_risk=round(dead_zone_risk, 6),
        dead_zone_flags=dead_zone_flags,
    )

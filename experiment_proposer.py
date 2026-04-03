from __future__ import annotations

from datetime import datetime

import pandas as pd

from experiment_spaces import enumerate_grid_candidates, normalize_experiment_params, sample_random_candidates
from experiment_store import compute_config_hash
from experiment_types import ExperimentSpec


OBJECTIVE_NAME = "wf_v1_score"


def _make_spec(family: str, params: dict, method: str, batch_id: str) -> ExperimentSpec:
    normalized = normalize_experiment_params(family, params)
    return ExperimentSpec(
        family=family,
        params=normalized,
        search_method=method,
        objective_name=OBJECTIVE_NAME,
        batch_id=batch_id,
        config_hash=compute_config_hash(family, normalized),
    )


def propose_initial_batch(
    family: str,
    method: str,
    n: int,
    seed: int,
    batch_id: str | None = None,
) -> list[ExperimentSpec]:
    batch_id = batch_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if method == "grid":
        candidates = enumerate_grid_candidates(family, limit=n)
    else:
        candidates = sample_random_candidates(family, n=n, seed=seed)
    return [_make_spec(family, params, method, batch_id) for params in candidates]


def propose_next_batch(
    family: str,
    prior_results: pd.DataFrame,
    n: int,
    seed: int,
    batch_id: str | None = None,
) -> list[ExperimentSpec]:
    batch_id = batch_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if prior_results.empty:
        return propose_initial_batch(family, method="random", n=n, seed=seed, batch_id=batch_id)

    explored = set(prior_results["config_hash"].astype(str))
    candidates = sample_random_candidates(family, n=max(n * 3, n), seed=seed)
    specs: list[ExperimentSpec] = []
    for params in candidates:
        spec = _make_spec(family, params, "random", batch_id)
        if spec.config_hash in explored:
            continue
        specs.append(spec)
        if len(specs) >= n:
            break
    return specs

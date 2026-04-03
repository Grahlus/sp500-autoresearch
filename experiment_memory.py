from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MEMORY_FILE = "memory.json"


def _memory_path(base_dir: str) -> Path:
    return Path(base_dir) / MEMORY_FILE


def load_research_memory(base_dir: str = "experiments") -> dict[str, Any]:
    path = _memory_path(base_dir)
    if not path.exists():
        return {
            "version": 1,
            "updated_at": None,
            "families": {},
        }
    return json.loads(path.read_text())


def save_research_memory(memory: dict[str, Any], base_dir: str = "experiments") -> None:
    path = _memory_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(memory)
    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def update_family_memory(
    memory: dict[str, Any],
    family: str,
    *,
    exact_hashes: set[str],
    coarse_signatures: set[str],
    dead_zone_values: dict[str, set[Any]],
    dead_zone_signatures: set[str],
    poor_region_signatures: set[str] | None = None,
    template_counts: dict[str, int],
    best_objective_score: float | None,
    best_config_hash: str | None,
    stagnation_batches: int,
) -> dict[str, Any]:
    families = memory.setdefault("families", {})
    families[family] = {
        "exact_hashes": sorted(exact_hashes),
        "coarse_signatures": sorted(coarse_signatures),
        "dead_zone_values": {key: sorted(str(value) for value in values) for key, values in dead_zone_values.items()},
        "dead_zone_signatures": sorted(dead_zone_signatures),
        "poor_region_signatures": sorted(poor_region_signatures or set()),
        "template_counts": dict(sorted(template_counts.items())),
        "best_objective_score": best_objective_score,
        "best_config_hash": best_config_hash,
        "stagnation_batches": int(stagnation_batches),
    }
    return memory

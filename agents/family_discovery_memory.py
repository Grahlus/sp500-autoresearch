"""Rolling memory for family discovery — prevents repetition across runs.

State file: queues/family_candidates/discovery_memory.json

Tracks per-run history:
  - recently_proposed_names    — snake_case family names from all recent runs
  - recently_proposed_mechanisms — edge_source strings (first 80 chars)
  - recently_used_themes       — search theme slugs
  - recently_used_queries      — full search query strings
  - recently_cited_sources     — web source URLs
  - recently_promoted          — names that reached controlled_probe
  - batches                    — per-run records (rolling window)

The memory is loaded BEFORE discovery to:
  (a) inject "do not re-propose" names into the generation prompt
  (b) score candidates for repeat_penalty
  (c) enforce search-theme rotation

The memory is updated AFTER every successful discovery run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.schemas import _atomic_write_json, utc_now_iso

_MEMORY_FILE = "queues/family_candidates/discovery_memory.json"
_NAME_WINDOW = 60       # max recent family names kept
_MECHANISM_WINDOW = 40  # max recent mechanism strings
_THEME_WINDOW = 20      # max recent search theme slugs
_QUERY_WINDOW = 30      # max recent search query strings
_SOURCE_WINDOW = 30     # max recent web source URLs
_PROMOTED_WINDOW = 20   # max recently promoted names
_BATCH_WINDOW = 8       # max recent batch records

# Repeat penalty levels
_PENALTY_EXACT_NAME = 0.90
_PENALTY_HIGH_TOKEN_OVERLAP = 0.65
_PENALTY_MECHANISM_OVERLAP = 0.40


def _memory_path(workspace_root: str) -> Path:
    return Path(workspace_root) / _MEMORY_FILE


def load_memory(workspace_root: str = ".") -> dict[str, Any]:
    """Load memory from disk. Returns empty state if not found."""
    p = _memory_path(workspace_root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {
        "recently_proposed_names": [],
        "recently_proposed_mechanisms": [],
        "recently_used_themes": [],
        "recently_used_queries": [],
        "recently_cited_sources": [],
        "recently_promoted": [],
        "batches": [],
        "total_runs": 0,
    }


def save_memory(workspace_root: str, memory: dict[str, Any]) -> None:
    """Atomically persist memory to disk."""
    p = _memory_path(workspace_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(p, memory)


def update_memory_after_run(
    workspace_root: str = ".",
    *,
    batch_id: str,
    family_names: list[str],
    mechanisms: list[str] | None = None,
    search_themes: list[str] | None = None,
    search_queries: list[str] | None = None,
    web_sources: list[str] | None = None,
    promoted: list[str] | None = None,
) -> None:
    """Update rolling memory after a completed discovery run. Prepend-and-truncate."""
    mem = load_memory(workspace_root)

    def _prepend(key: str, items: list[str], window: int) -> None:
        if not items:
            return
        combined = [x for x in items if x] + [x for x in mem.get(key, []) if x]
        # deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for x in combined:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                deduped.append(x)
        mem[key] = deduped[:window]

    _prepend("recently_proposed_names", family_names, _NAME_WINDOW)
    _prepend("recently_proposed_mechanisms", [m[:80] for m in (mechanisms or [])], _MECHANISM_WINDOW)
    _prepend("recently_used_themes", search_themes or [], _THEME_WINDOW)
    _prepend("recently_used_queries", search_queries or [], _QUERY_WINDOW)
    _prepend("recently_cited_sources", web_sources or [], _SOURCE_WINDOW)
    _prepend("recently_promoted", promoted or [], _PROMOTED_WINDOW)

    batch_record = {
        "batch_id": batch_id,
        "timestamp": utc_now_iso(),
        "family_names": family_names[:20],
        "search_themes": (search_themes or [])[:10],
        "promoted": (promoted or [])[:10],
        "n_web_sources": len(web_sources or []),
    }
    mem["batches"] = ([batch_record] + mem.get("batches", []))[:_BATCH_WINDOW]
    mem["total_runs"] = mem.get("total_runs", 0) + 1

    save_memory(workspace_root, mem)


def get_avoidance_context(workspace_root: str = ".") -> dict[str, Any]:
    """Return names/themes that the next run should actively avoid.

    Used to build the generation prompt's "do not re-propose" section.
    """
    mem = load_memory(workspace_root)
    return {
        "names_to_avoid": mem.get("recently_proposed_names", [])[:25],
        "themes_to_avoid": mem.get("recently_used_themes", [])[:12],
        "promoted_to_avoid": mem.get("recently_promoted", [])[:10],
        "total_past_runs": mem.get("total_runs", 0),
    }


def compute_repeat_score(
    family_name: str,
    edge_source: str,
    workspace_root: str = ".",
) -> tuple[float, str]:
    """Return (repeat_penalty 0..1, reason_string).

    A score of 1.0 means certain repeat; 0.0 means genuinely fresh.
    """
    mem = load_memory(workspace_root)
    recent_names = [n.lower() for n in mem.get("recently_proposed_names", [])]
    recent_mechs = [m.lower() for m in mem.get("recently_proposed_mechanisms", [])]

    name_lower = family_name.lower()
    edge_lower = edge_source.lower()[:80]

    # Exact name match
    if name_lower in recent_names:
        return _PENALTY_EXACT_NAME, f"exact_name_match:{family_name}"

    # High token overlap in name
    name_tokens = set(name_lower.replace("_", " ").split())
    best_name_overlap = 0.0
    for rn in recent_names[:20]:
        rn_tokens = set(rn.replace("_", " ").split())
        if not name_tokens or not rn_tokens:
            continue
        overlap = len(name_tokens & rn_tokens) / max(len(name_tokens), len(rn_tokens))
        best_name_overlap = max(best_name_overlap, overlap)

    if best_name_overlap >= 0.65:
        return _PENALTY_HIGH_TOKEN_OVERLAP, f"name_token_overlap:{best_name_overlap:.2f}"

    # Edge source token overlap
    edge_tokens = set(edge_lower.split())
    best_edge_overlap = 0.0
    for rm in recent_mechs[:15]:
        rm_tokens = set(rm.split())
        if not edge_tokens or not rm_tokens:
            continue
        overlap = len(edge_tokens & rm_tokens) / max(len(edge_tokens), len(rm_tokens))
        best_edge_overlap = max(best_edge_overlap, overlap)

    if best_edge_overlap >= 0.55:
        return _PENALTY_MECHANISM_OVERLAP, f"mechanism_overlap:{best_edge_overlap:.2f}"

    # Partial name overlap (softer)
    if best_name_overlap >= 0.40:
        return round(best_name_overlap * 0.5, 3), f"partial_name_overlap:{best_name_overlap:.2f}"

    return 0.0, "fresh"

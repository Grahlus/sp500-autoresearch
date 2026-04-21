"""Persistence and query layer for family candidate records.

Family candidates flow through these statuses:
  new_family_candidate  →  controlled_probe  →  (active/retired_for_now/redesign_candidate)

The store reads from queues/family_candidates/*.json and provides helpers
that the autonomous runner, family discovery agent, and experiment_spaces
module can use to query and update candidate state.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any

from agents.schemas import (
    FAMILY_CANDIDATE_ACTIVE_STATUSES,
    FAMILY_CANDIDATE_STATUSES,
    FamilyCandidateRecord,
    _atomic_write_json,
    load_active_family_candidates,
    load_family_candidates,
    load_json_records,
    save_family_candidate,
    update_family_candidate_status,
    utc_now_iso,
)

_PROBE_SEED_STALE_HOURS = 72
_PROBE_PROPOSAL_STALE_HOURS = 48
_PROBE_MAX_NON_VIABLE = 3


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_candidate_by_id(candidate_id: str, workspace_root: str = ".") -> dict[str, Any] | None:
    """Return a single candidate record by ID, or None if not found."""
    path = Path(workspace_root) / "queues" / "family_candidates" / f"{candidate_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def get_candidate_by_family_name(family_name: str, workspace_root: str = ".") -> list[dict[str, Any]]:
    """Return all candidate records for a given family_name (may have multiple from different batches)."""
    all_candidates = load_family_candidates(workspace_root)
    return [c for c in all_candidates if c.get("family_name") == family_name]


def get_controlled_probes(workspace_root: str = ".") -> list[dict[str, Any]]:
    """Return all candidates currently in controlled_probe status."""
    all_candidates = load_family_candidates(workspace_root)
    return [c for c in all_candidates if c.get("status") == "controlled_probe"]


def get_new_family_candidates(workspace_root: str = ".") -> list[dict[str, Any]]:
    """Return all candidates in new_family_candidate status."""
    all_candidates = load_family_candidates(workspace_root)
    return [c for c in all_candidates if c.get("status") == "new_family_candidate"]


def top_candidates_by_score(
    workspace_root: str = ".",
    *,
    n: int = 10,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Return top N candidates sorted by ranking_score descending."""
    all_candidates = load_family_candidates(workspace_root)
    if status_filter:
        all_candidates = [c for c in all_candidates if c.get("status") == status_filter]
    # Deduplicate by family_name, keeping the one with the highest ranking_score
    best_by_name: dict[str, dict[str, Any]] = {}
    for c in all_candidates:
        name = c.get("family_name", "")
        existing = best_by_name.get(name)
        if existing is None or _safe_float(c.get("ranking_score"), 0.0) > _safe_float(existing.get("ranking_score"), 0.0):
            best_by_name[name] = c
    ranked = sorted(best_by_name.values(), key=lambda c: _safe_float(c.get("ranking_score"), 0.0), reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

def promote_to_controlled_probe(
    candidate_id: str,
    workspace_root: str = ".",
    *,
    reason: str = "manual_promotion",
) -> bool:
    """Promote a candidate from new_family_candidate to controlled_probe."""
    ok = update_family_candidate_status(
        candidate_id,
        "controlled_probe",
        workspace_root,
        extra_updates={"promoted_at": utc_now_iso(), "promotion_reason": reason},
    )
    if ok:
        print(f"[family_store] promoted {candidate_id} → controlled_probe (reason={reason})", flush=True)
    return ok


def retire_candidate(
    candidate_id: str,
    workspace_root: str = ".",
    *,
    reason: str = "manual_retirement",
) -> bool:
    """Mark a candidate as retired_for_now."""
    ok = update_family_candidate_status(
        candidate_id,
        "retired_for_now",
        workspace_root,
        extra_updates={"retired_at": utc_now_iso(), "retirement_reason": reason},
    )
    if ok:
        print(f"[family_store] retired {candidate_id} (reason={reason})", flush=True)
    return ok


def mark_for_redesign(
    candidate_id: str,
    workspace_root: str = ".",
    *,
    reason: str = "manual_redesign",
) -> bool:
    """Mark a candidate as redesign_candidate."""
    ok = update_family_candidate_status(
        candidate_id,
        "redesign_candidate",
        workspace_root,
        extra_updates={"redesign_flagged_at": utc_now_iso(), "redesign_reason": reason},
    )
    if ok:
        print(f"[family_store] marked {candidate_id} for redesign (reason={reason})", flush=True)
    return ok


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _hours_since(ts_str: str | None) -> float | None:
    then = _parse_ts(ts_str)
    if then is None:
        return None
    now = datetime.now(UTC)
    return (now - then).total_seconds() / 3600.0


def recycle_stale_probe_slots(workspace_root: str = ".") -> list[dict[str, Any]]:
    """Release controlled_probe slots from candidates that are stale.

    Slot-release reasons:
      promoted_not_seeded   — promoted but no IdeaRecord seeded within PROBE_SEED_STALE_HOURS
      seeded_no_proposal    — seeded but no proposal generated within PROBE_PROPOSAL_STALE_HOURS
      repeated_non_viable   — proposal executed but non-viable >= PROBE_MAX_NON_VIABLE times
      explicitly_retired    — status is retired_for_now or redesign_candidate

    Returns a list of release descriptors for reporting.
    """
    releases: list[dict[str, Any]] = []
    candidates_dir = Path(workspace_root) / "queues" / "family_candidates"

    ideas_dir = Path(workspace_root) / "queues" / "ideas"
    proposals_dir = Path(workspace_root) / "queues" / "proposals"

    ideas_by_family: dict[str, list[dict]] = {}
    if ideas_dir.exists():
        for p in ideas_dir.glob("*.json"):
            try:
                idea = json.loads(p.read_text())
                fam = idea.get("family") or idea.get("idea_family")
                if fam:
                    ideas_by_family.setdefault(fam, []).append(idea)
            except Exception:
                continue

    proposals_by_family: dict[str, list[dict]] = {}
    if proposals_dir.exists():
        for p in proposals_dir.glob("*.json"):
            try:
                prop = json.loads(p.read_text())
                for fam in prop.get("strategy_families", []):
                    proposals_by_family.setdefault(fam, []).append(prop)
            except Exception:
                continue

    for p in candidates_dir.glob("*.json"):
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue

        if c.get("status") != "controlled_probe":
            continue

        candidate_id = c.get("candidate_id", p.stem)
        family_name = c.get("family_name", "")
        release_reason: str | None = None
        release_detail: str = ""

        if family_name in ("retired_for_now", "redesign_candidate"):
            release_reason = "explicitly_retired"
            release_detail = f"status={c.get('status')}"

        promoted_at = c.get("promoted_at")
        seeded_ideas = ideas_by_family.get(family_name, [])
        family_proposals = proposals_by_family.get(family_name, [])

        if release_reason is None and promoted_at and not seeded_ideas:
            hrs = _hours_since(promoted_at)
            if hrs is not None and hrs > _PROBE_SEED_STALE_HOURS:
                release_reason = "promoted_not_seeded"
                release_detail = f"promoted_at={promoted_at}, hrs={hrs:.1f}h"

        if release_reason is None and seeded_ideas:
            latest_idea = max(seeded_ideas, key=lambda i: i.get("timestamp_utc") or "")
            idea_created = latest_idea.get("timestamp_utc")
            if idea_created and not family_proposals:
                hrs = _hours_since(idea_created)
                if hrs is not None and hrs > _PROBE_PROPOSAL_STALE_HOURS:
                    release_reason = "seeded_no_proposal"
                    release_detail = f"idea_created={idea_created}, hrs={hrs:.1f}h"

        if release_reason is None and family_proposals:
            n_viable = sum(1 for p in family_proposals if p.get("status") == "executed" and p.get("viable") is True)
            n_non_viable = sum(1 for p in family_proposals if p.get("status") == "executed" and p.get("viable") is False)
            if n_non_viable >= _PROBE_MAX_NON_VIABLE and n_viable == 0:
                release_reason = "repeated_non_viable"
                release_detail = f"non_viable={n_non_viable}, viable={n_viable}"

        if release_reason is not None:
            update_family_candidate_status(
                candidate_id,
                "retired_for_now",
                workspace_root,
                extra_updates={
                    "recycled_at": utc_now_iso(),
                    "slot_release_reason": release_reason,
                    "slot_release_detail": release_detail,
                },
            )
            releases.append({
                "candidate_id": candidate_id,
                "family_name": family_name,
                "reason": release_reason,
                "detail": release_detail,
            })
            print(f"[family_store] recycled probe slot {candidate_id} ({family_name}): {release_reason} — {release_detail}", flush=True)

    return releases


def get_probe_slot_usage(workspace_root: str = ".") -> dict[str, Any]:
    """Return current probe slot usage summary."""
    probes = get_controlled_probes(workspace_root)
    now = datetime.now(UTC)
    slot_details = []
    for c in probes:
        promoted_at = c.get("promoted_at")
        hrs_old = _hours_since(promoted_at) if promoted_at else None
        slot_details.append({
            "candidate_id": c.get("candidate_id"),
            "family_name": c.get("family_name"),
            "status": c.get("status"),
            "promoted_at": promoted_at,
            "age_hours": round(hrs_old, 1) if hrs_old is not None else None,
        })
    return {
        "active_slots": len(probes),
        "slots": slot_details,
    }


def get_dynamic_probe_cap(workspace_root: str = ".") -> int:
    """Return the effective probe cap based on system health.

    Rules:
      constrained  — active probes >= 4 and backlog < 6  → cap 4
      healthy      — active probes <= 2 and backlog >= 6  → cap 6
      very_healthy — active probes <= 1 and backlog >= 12  → cap 8 (optional, not used by default)
      default      — anything else                          → cap 6
    """
    probes = get_controlled_probes(workspace_root)
    new_candidates = get_new_family_candidates(workspace_root)
    active = len(probes)
    backlog = len(new_candidates)

    if active >= 4 and backlog < 6:
        return 4
    if active <= 2 and backlog >= 6:
        return 6
    return 6


# ---------------------------------------------------------------------------
# Compact summary for reporting
# ---------------------------------------------------------------------------

def family_candidate_summary(workspace_root: str = ".") -> dict[str, Any]:
    """Return a compact summary of the family candidate queue state."""
    all_candidates = load_family_candidates(workspace_root)
    if not all_candidates:
        return {"total": 0, "by_status": {}, "top_candidates": []}

    by_status: dict[str, int] = {}
    for c in all_candidates:
        status = c.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

    top = top_candidates_by_score(workspace_root, n=5)
    latest_batch = max(
        (c.get("discovery_batch_id") for c in all_candidates if c.get("discovery_batch_id")),
        default=None,
    )
    return {
        "total": len(all_candidates),
        "by_status": by_status,
        "latest_batch_id": latest_batch,
        "top_candidates": [
            {
                "family_name": c.get("family_name"),
                "status": c.get("status"),
                "ranking_score": c.get("ranking_score"),
                "orthogonality_score": c.get("orthogonality_score"),
                "edge_source": (c.get("edge_source") or "")[:80],
            }
            for c in top
        ],
    }


# ---------------------------------------------------------------------------
# Integration hook: inject controlled_probe candidates into experiment_spaces
# ---------------------------------------------------------------------------

def get_controlled_probe_status_entries(workspace_root: str = ".") -> dict[str, dict[str, Any]]:
    """Return a FAMILY_SEARCH_STATUS-compatible dict for all controlled_probe candidates.

    These can be merged into the static registry so the experiment pipeline
    becomes aware of them without modifying experiment_spaces.py directly.
    """
    probes = get_controlled_probes(workspace_root)
    entries: dict[str, dict[str, Any]] = {}
    for c in probes:
        name = c.get("family_name", "")
        if not name:
            continue
        entries[name] = {
            "status": "controlled_candidate",
            "canonical_status": "controlled_candidate",
            "lane": "bounded",
            "budget_scale": 0.30,
            "searchable": True,
            "reason": f"Family-discovery controlled probe: {(c.get('edge_source') or '')[:80]}",
            "family_candidate_id": c.get("candidate_id"),
            "orthogonality_score": c.get("orthogonality_score"),
            "ranking_score": c.get("ranking_score"),
        }
    return entries


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

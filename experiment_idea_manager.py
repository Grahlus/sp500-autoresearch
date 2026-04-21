"""
Idea lifecycle manager: cluster deduplication, queue caps, and backlog reporting.

This module is read/write for idea metadata only. It does not touch any backtest
results, strategy logic, or the official experiments store.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lifecycle status definitions
# ---------------------------------------------------------------------------

IDEA_ACTIVE_STATUSES: frozenset[str] = frozenset({"new", "hot", "shortlisted"})
IDEA_TERMINAL_STATUSES: frozenset[str] = frozenset({"archived", "retired", "merged"})
IDEA_EXECUTED_STATUS = "executed"

# All known statuses in the lifecycle
IDEA_ALL_STATUSES = (
    "new",           # freshly generated, eligible for scheduling
    "hot",           # explicitly promoted to active scheduling pool
    "shortlisted",   # selected into a proposal this cycle
    "executed",      # experiment has run; not re-scheduled by default
    "merged",        # superseded by a higher-priority cluster representative
    "stale",         # not selected after enough cycles; lower priority
    "archived",      # retired from active pool; provenance kept on disk
    "retired",       # explicitly excluded (e.g. from repeated failures)
)

# ---------------------------------------------------------------------------
# Queue caps
# ---------------------------------------------------------------------------

MAX_QUEUE_TOTAL: int = 60
MAX_QUEUE_PER_FAMILY: int = 20
MAX_QUEUE_PER_CLUSTER: int = 3

# How many hours before an un-shortlisted idea is considered stale
STALE_AGE_HOURS: float = 48.0

# Structural/novel ideas are protected from archiving
_PROTECTED_SOURCES = frozenset({"web_search", "structural_extension"})


# ---------------------------------------------------------------------------
# Cluster key
# ---------------------------------------------------------------------------

def compute_cluster_key(idea: dict[str, Any]) -> str:
    """
    Returns a string key that groups near-duplicate ideas into a cluster.
    Template ideas are grouped by (family, template_id).
    Structural/novel ideas are grouped by (family, template_similarity_class, source).
    Everything else falls back to (family, idea_kind, idea_source).
    """
    family = str(idea.get("family") or "").strip().lower()
    template_id = str(idea.get("template_id") or idea.get("suggested_template_id") or "").strip().lower()
    idea_source = str(idea.get("idea_source") or idea.get("source") or "").strip().lower()
    idea_kind = str(idea.get("idea_kind") or "").strip().lower()
    sim_class = str(idea.get("template_similarity_class") or "").strip().lower()
    is_structural = bool(idea.get("is_structurally_novel")) or idea_source in _PROTECTED_SOURCES

    if template_id and not is_structural:
        return f"{family}|template|{template_id}"
    if is_structural and sim_class:
        return f"{family}|structural|{sim_class}|{idea_source}"
    if idea_kind:
        return f"{family}|kind|{idea_kind}|{idea_source}"
    return f"{family}|other|{idea_source}"


def cluster_ideas(ideas: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group ideas by cluster key. Returns {cluster_key: [idea, ...]}."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for idea in ideas:
        key = compute_cluster_key(idea)
        clusters.setdefault(key, []).append(idea)
    return clusters


# ---------------------------------------------------------------------------
# Active idea loading
# ---------------------------------------------------------------------------

def _ideas_dir(workspace_root: str) -> Path:
    return Path(workspace_root) / "queues" / "ideas"


def load_all_idea_records_raw(workspace_root: str) -> list[dict[str, Any]]:
    """Load all idea JSON files from disk, sorted by timestamp ascending."""
    directory = _ideas_dir(workspace_root)
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            continue
    records.sort(key=lambda r: str(r.get("timestamp_utc") or r.get("idea_id") or ""))
    return records


def load_active_idea_records(workspace_root: str) -> list[dict[str, Any]]:
    """Load idea records whose status is active (not archived/retired/merged/executed)."""
    return [
        r for r in load_all_idea_records_raw(workspace_root)
        if r.get("status", "new") in IDEA_ACTIVE_STATUSES
    ]


# ---------------------------------------------------------------------------
# Status update
# ---------------------------------------------------------------------------

def _atomic_patch_status(path: Path, new_status: str, extra: dict[str, Any] | None = None) -> bool:
    """Patch only the status (and optional extra fields) in an idea JSON file atomically."""
    try:
        payload = json.loads(path.read_text())
        payload["status"] = new_status
        if extra:
            payload.update(extra)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            tmp = Path(handle.name)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def update_idea_status(
    idea_id: str,
    new_status: str,
    workspace_root: str,
    *,
    extra: dict[str, Any] | None = None,
) -> bool:
    """
    Update the status field of an idea record file in-place.
    Returns True on success, False if the file was not found.
    """
    directory = _ideas_dir(workspace_root)
    path = directory / f"{idea_id}.json"
    if not path.exists():
        return False
    return _atomic_patch_status(path, new_status, extra)


# ---------------------------------------------------------------------------
# Hot queue selection (cluster-aware, capped)
# ---------------------------------------------------------------------------

def _sort_key_for_idea(idea: dict[str, Any]) -> tuple:
    """Lower is better."""
    is_protected = (
        bool(idea.get("is_structurally_novel"))
        or bool(idea.get("is_out_of_box"))
        or bool(idea.get("is_uncommon_idea"))
        or str(idea.get("idea_source") or idea.get("source") or "") in _PROTECTED_SOURCES
    )
    return (
        0 if is_protected else 1,
        -float(idea.get("priority") or 0.0),
        -float(idea.get("novelty_score") or 0.0),
        str(idea.get("timestamp_utc") or ""),
    )


def select_hot_representatives(
    ideas: list[dict[str, Any]],
    *,
    max_total: int = MAX_QUEUE_TOTAL,
    max_per_family: int = MAX_QUEUE_PER_FAMILY,
    max_per_cluster: int = MAX_QUEUE_PER_CLUSTER,
) -> list[dict[str, Any]]:
    """
    Select the hot subset of ideas that the scheduler should consider this cycle.

    Rules:
    - Cluster deduplication: at most max_per_cluster representatives per cluster.
    - Family cap: at most max_per_family ideas per family.
    - Total cap: at most max_total ideas overall.
    - Structural/novel ideas are promoted and protected within their cluster.

    Returns a list sorted by priority descending.
    """
    ordered = sorted(ideas, key=_sort_key_for_idea)

    selected: list[dict[str, Any]] = []
    cluster_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()

    for idea in ordered:
        if len(selected) >= max_total:
            break
        family = str(idea.get("family") or "")
        if family_counts[family] >= max_per_family:
            continue
        cluster = compute_cluster_key(idea)
        if cluster_counts[cluster] >= max_per_cluster:
            continue
        selected.append(idea)
        cluster_counts[cluster] += 1
        family_counts[family] += 1

    selected.sort(
        key=lambda r: (
            -float(r.get("priority") or 0.0),
            -float(r.get("novelty_score") or 0.0),
        )
    )
    return selected


# ---------------------------------------------------------------------------
# Queue gate: should we save a new idea?
# ---------------------------------------------------------------------------

def gate_new_idea(
    new_idea: dict[str, Any],
    workspace_root: str,
    *,
    max_total: int = MAX_QUEUE_TOTAL,
    max_per_family: int = MAX_QUEUE_PER_FAMILY,
    max_per_cluster: int = MAX_QUEUE_PER_CLUSTER,
) -> tuple[bool, str]:
    """
    Decide whether to save a new idea to the queue.

    Returns (should_save, reason_string).

    If the per-cluster cap would be exceeded, the oldest non-protected cluster
    member is archived first, then the new idea is allowed.
    If max_total or max_per_family caps would be exceeded, the oldest non-protected
    ideas are archived to make room.
    """
    active = load_all_idea_records_raw(workspace_root)
    active_non_terminal = [r for r in active if r.get("status", "new") not in IDEA_TERMINAL_STATUSES and r.get("status") != IDEA_EXECUTED_STATUS]

    family = str(new_idea.get("family") or "")
    cluster = compute_cluster_key(new_idea)
    is_protected = (
        bool(new_idea.get("is_structurally_novel"))
        or bool(new_idea.get("is_out_of_box"))
        or str(new_idea.get("idea_source") or new_idea.get("source") or "") in _PROTECTED_SOURCES
    )

    # Count existing cluster members
    cluster_members = [r for r in active_non_terminal if compute_cluster_key(r) == cluster]
    if len(cluster_members) >= max_per_cluster:
        if is_protected:
            # Archive the oldest non-protected cluster member
            evict_candidates = sorted(
                [r for r in cluster_members if not bool(r.get("is_structurally_novel")) and not bool(r.get("is_out_of_box"))],
                key=lambda r: str(r.get("timestamp_utc") or ""),
            )
            if evict_candidates:
                oldest = evict_candidates[0]
                update_idea_status(str(oldest.get("idea_id") or ""), "archived", workspace_root, extra={"archive_reason": "cluster_cap_evict_for_protected"})
                active_non_terminal = [r for r in active_non_terminal if r.get("idea_id") != oldest.get("idea_id")]
            else:
                return False, f"cluster_cap_exceeded cluster={cluster}"
        else:
            return False, f"cluster_cap_exceeded cluster={cluster}"

    # Family cap
    family_ideas = [r for r in active_non_terminal if str(r.get("family") or "") == family]
    if len(family_ideas) >= max_per_family:
        if is_protected:
            evict_candidates = sorted(
                [r for r in family_ideas if not bool(r.get("is_structurally_novel")) and not bool(r.get("is_out_of_box"))],
                key=lambda r: str(r.get("timestamp_utc") or ""),
            )
            if evict_candidates:
                oldest = evict_candidates[0]
                update_idea_status(str(oldest.get("idea_id") or ""), "archived", workspace_root, extra={"archive_reason": "family_cap_evict_for_protected"})
                active_non_terminal = [r for r in active_non_terminal if r.get("idea_id") != oldest.get("idea_id")]
            else:
                return False, f"family_cap_exceeded family={family}"
        else:
            return False, f"family_cap_exceeded family={family}"

    # Total cap
    if len(active_non_terminal) >= max_total:
        evict_candidates = sorted(
            [r for r in active_non_terminal if not bool(r.get("is_structurally_novel")) and not bool(r.get("is_out_of_box")) and str(r.get("family") or "") == family],
            key=lambda r: str(r.get("timestamp_utc") or ""),
        )
        if not evict_candidates:
            evict_candidates = sorted(
                [r for r in active_non_terminal if not bool(r.get("is_structurally_novel")) and not bool(r.get("is_out_of_box"))],
                key=lambda r: str(r.get("timestamp_utc") or ""),
            )
        if evict_candidates:
            oldest = evict_candidates[0]
            update_idea_status(str(oldest.get("idea_id") or ""), "archived", workspace_root, extra={"archive_reason": "total_cap_evict"})
        else:
            return False, "total_cap_exceeded_no_evict_candidates"

    return True, "ok"


# ---------------------------------------------------------------------------
# Lifecycle maintenance
# ---------------------------------------------------------------------------

def run_lifecycle_maintenance(
    workspace_root: str,
    *,
    max_total: int = MAX_QUEUE_TOTAL,
    max_per_family: int = MAX_QUEUE_PER_FAMILY,
    max_per_cluster: int = MAX_QUEUE_PER_CLUSTER,
    stale_age_hours: float = STALE_AGE_HOURS,
) -> dict[str, Any]:
    """
    Full maintenance pass:
    1. Archive ideas older than stale_age_hours that were never shortlisted/executed.
    2. Enforce cluster caps by archiving oldest non-protected duplicates.
    3. Enforce family caps by archiving oldest non-protected ideas.
    4. Enforce total cap by archiving oldest non-protected ideas.

    Returns a summary of actions taken.
    """
    directory = _ideas_dir(workspace_root)
    if not directory.exists():
        return {"archived": 0, "actions": []}

    actions: list[str] = []
    now_ts = datetime.now(UTC).timestamp()
    stale_cutoff_ts = now_ts - (stale_age_hours * 3600.0)

    all_records = load_all_idea_records_raw(workspace_root)
    active = [r for r in all_records if r.get("status", "new") not in IDEA_TERMINAL_STATUSES and r.get("status") != IDEA_EXECUTED_STATUS]

    def _is_protected(r: dict[str, Any]) -> bool:
        return bool(r.get("is_structurally_novel")) or bool(r.get("is_out_of_box"))

    def _ts(r: dict[str, Any]) -> float:
        try:
            return datetime.fromisoformat(str(r.get("timestamp_utc") or "")).timestamp()
        except Exception:
            return 0.0

    def _archive(r: dict[str, Any], reason: str) -> None:
        idea_id = str(r.get("idea_id") or "")
        if idea_id and update_idea_status(idea_id, "archived", workspace_root, extra={"archive_reason": reason}):
            actions.append(f"archived {idea_id} reason={reason}")

    # Step 1: stale age demotion for non-protected, non-shortlisted ideas
    for r in active[:]:
        if _is_protected(r):
            continue
        if str(r.get("status") or "new") in {"shortlisted", "hot"}:
            continue
        if _ts(r) < stale_cutoff_ts:
            _archive(r, "stale_age")

    # Reload after step 1
    active = [r for r in load_all_idea_records_raw(workspace_root) if r.get("status", "new") not in IDEA_TERMINAL_STATUSES and r.get("status") != IDEA_EXECUTED_STATUS]

    # Step 2: cluster cap
    clusters = cluster_ideas(active)
    for cluster_key, members in clusters.items():
        if len(members) <= max_per_cluster:
            continue
        non_protected = [r for r in members if not _is_protected(r)]
        non_protected.sort(key=_ts)
        evict_count = len(members) - max_per_cluster
        for r in non_protected[:evict_count]:
            _archive(r, f"cluster_cap cluster={cluster_key}")

    # Reload
    active = [r for r in load_all_idea_records_raw(workspace_root) if r.get("status", "new") not in IDEA_TERMINAL_STATUSES and r.get("status") != IDEA_EXECUTED_STATUS]

    # Step 3: family cap
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in active:
        fam = str(r.get("family") or "")
        by_family.setdefault(fam, []).append(r)
    for fam, members in by_family.items():
        if len(members) <= max_per_family:
            continue
        non_protected = sorted([r for r in members if not _is_protected(r)], key=_ts)
        evict_count = len(members) - max_per_family
        for r in non_protected[:evict_count]:
            _archive(r, f"family_cap family={fam}")

    # Reload
    active = [r for r in load_all_idea_records_raw(workspace_root) if r.get("status", "new") not in IDEA_TERMINAL_STATUSES and r.get("status") != IDEA_EXECUTED_STATUS]

    # Step 4: total cap
    if len(active) > max_total:
        non_protected = sorted([r for r in active if not _is_protected(r)], key=_ts)
        evict_count = len(active) - max_total
        for r in non_protected[:evict_count]:
            _archive(r, "total_cap")

    return {"archived": len(actions), "actions": actions}


# ---------------------------------------------------------------------------
# Backlog report
# ---------------------------------------------------------------------------

def backfill_executed_ideas(
    workspace_root: str,
    experiments_index_path: str | None = None,
) -> dict[str, Any]:
    """
    Cross-reference the idea queue against the experiments index and mark ideas
    whose idea_id appears as a source_idea_id in any experiment as `executed`.

    This is safe to run repeatedly (idempotent): ideas already in terminal or
    executed state are skipped.

    Returns a summary of how many ideas were updated.
    """
    import csv

    if experiments_index_path is None:
        directory = _ideas_dir(workspace_root)
        # Walk up to find experiments/
        experiments_dir = directory.parent.parent / "experiments"
        experiments_index_path = str(experiments_dir / "index.csv")

    idx_path = Path(experiments_index_path)
    if not idx_path.exists():
        return {"backfilled": 0, "already_executed": 0, "not_found": 0}

    # Collect all idea_ids referenced in experiments
    executed_idea_ids: set[str] = set()
    try:
        with open(idx_path, newline="") as f:
            for row in csv.DictReader(f):
                sid = row.get("source_idea_ids", "")
                if not sid or sid in ("", "nan", "None", "[]"):
                    continue
                try:
                    import json as _j
                    parsed = _j.loads(sid.replace("'", '"'))
                    for iid in (parsed if isinstance(parsed, list) else []):
                        iid_str = str(iid).strip()
                        if iid_str.startswith("idea_"):
                            executed_idea_ids.add(iid_str)
                except Exception:
                    pass
    except Exception:
        return {"backfilled": 0, "already_executed": 0, "not_found": 0, "error": "index_read_failed"}

    all_records = load_all_idea_records_raw(workspace_root)
    queue_by_id = {str(r.get("idea_id") or ""): r for r in all_records}

    backfilled = 0
    already_executed = 0
    not_found = 0

    for idea_id in executed_idea_ids:
        if idea_id not in queue_by_id:
            not_found += 1
            continue
        record = queue_by_id[idea_id]
        current_status = str(record.get("status") or "new")
        if current_status == IDEA_EXECUTED_STATUS:
            already_executed += 1
            continue
        if current_status in IDEA_TERMINAL_STATUSES:
            already_executed += 1
            continue
        if update_idea_status(idea_id, IDEA_EXECUTED_STATUS, workspace_root, extra={"backfilled_from_index": True}):
            backfilled += 1

    return {
        "backfilled": backfilled,
        "already_executed": already_executed,
        "not_found": not_found,
        "total_referenced": len(executed_idea_ids),
    }


def backlog_report(workspace_root: str) -> dict[str, Any]:
    """
    Build a human-readable backlog report showing queue health.
    """
    all_records = load_all_idea_records_raw(workspace_root)
    total = len(all_records)

    by_status: Counter[str] = Counter()
    by_family: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    by_cluster: Counter[str] = Counter()
    protected_count = 0
    executed_count = 0
    never_shortlisted_count = 0

    now_ts = datetime.now(UTC).timestamp()

    for r in all_records:
        status = str(r.get("status") or "new")
        by_status[status] += 1
        fam = str(r.get("family") or "unknown")
        by_family[fam] += 1
        src = str(r.get("idea_source") or r.get("source") or "unknown")
        by_source[src] += 1
        kind = str(r.get("idea_kind") or "unknown")
        by_kind[kind] += 1
        # Only count active (non-terminal) ideas in cluster sizes
        if status not in IDEA_TERMINAL_STATUSES and status != IDEA_EXECUTED_STATUS:
            cluster = compute_cluster_key(r)
            by_cluster[cluster] += 1
        if bool(r.get("is_structurally_novel")) or bool(r.get("is_out_of_box")):
            protected_count += 1
        if status == IDEA_EXECUTED_STATUS:
            executed_count += 1
        if status not in {"shortlisted", "executed", "merged", "archived", "retired"}:
            never_shortlisted_count += 1

    active = [r for r in all_records if r.get("status", "new") in IDEA_ACTIVE_STATUSES]
    active_count = len(active)

    terminal = [r for r in all_records if r.get("status", "new") in IDEA_TERMINAL_STATUSES]
    terminal_count = len(terminal)

    # Top clusters by size
    top_clusters = by_cluster.most_common(10)

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "total_ideas": total,
        "active_ideas": active_count,
        "terminal_ideas": terminal_count,
        "executed_ideas": executed_count,
        "protected_ideas": protected_count,
        "never_shortlisted": never_shortlisted_count,
        "by_status": dict(by_status),
        "by_family": dict(by_family),
        "by_source": dict(by_source),
        "by_kind": dict(by_kind),
        "top_clusters_by_size": top_clusters,
        "caps": {
            "max_total": MAX_QUEUE_TOTAL,
            "max_per_family": MAX_QUEUE_PER_FAMILY,
            "max_per_cluster": MAX_QUEUE_PER_CLUSTER,
        },
    }


# ---------------------------------------------------------------------------
# Throughput helper
# ---------------------------------------------------------------------------

def throughput_report(workspace_root: str) -> dict[str, Any]:
    """
    Summarise idea-to-proposal-to-execution throughput from idea queue metadata.
    """
    all_records = load_all_idea_records_raw(workspace_root)
    shortlisted = sum(1 for r in all_records if r.get("status") == "shortlisted")
    executed = sum(1 for r in all_records if r.get("status") == IDEA_EXECUTED_STATUS)
    active = sum(1 for r in all_records if r.get("status", "new") in IDEA_ACTIVE_STATUSES)
    total = len(all_records)

    shortlist_rate = shortlisted / total if total > 0 else 0.0
    execution_rate = executed / total if total > 0 else 0.0

    return {
        "total": total,
        "active_in_queue": active,
        "shortlisted": shortlisted,
        "executed": executed,
        "shortlist_rate": round(shortlist_rate, 4),
        "execution_rate": round(execution_rate, 4),
    }

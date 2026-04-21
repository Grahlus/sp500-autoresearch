"""Scheduling and state tracking for periodic family discovery runs.

Fast scheduler tick: cheap queue/probe management (promote, seed, recycle) — runs every cycle.
Slow scheduler tick: LLM family discovery — runs on interval/cycle threshold/stagnation.

State file: queues/family_candidates/discovery_state.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from agents.schemas import _atomic_write_json, utc_now_iso

_STATE_FILE = "queues/family_candidates/discovery_state.json"
_DEFAULT_INTERVAL_HOURS = 24.0
_DEFAULT_MIN_QUEUE = 3
_DEFAULT_EVERY_N_CYCLES = 8
_DEFAULT_FAST_TICK_EVERY_N_CYCLES = 1
_DEFAULT_FAST_TICK_INTERVAL_MINUTES = 10.0
_DEFAULT_STALE_CYCLES = 5
_DEFAULT_STALE_MINUTES = 60.0


def _state_path(workspace_root: str) -> Path:
    return Path(workspace_root) / _STATE_FILE


def _load_state(workspace_root: str) -> dict[str, Any]:
    p = _state_path(workspace_root)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {
        "last_run_ts": 0.0,
        "last_run_at": None,
        "run_count": 0,
        "cycles_since_last_run": 0,
        "last_batch_id": None,
        "last_n_candidates": 0,
        "last_n_promoted": 0,
        "stagnation_trigger_count": 0,
        "last_fast_tick_ts": 0.0,
        "last_fast_tick_at": None,
        "fast_tick_cycle_count": 0,
        "cycles_since_last_fast_tick": 0,
        "fast_tick_promotions": 0,
        "fast_tick_slot_releases": 0,
        "consecutive_failures": 0,
        "last_attempt_ts": 0.0,
        "last_attempt_at": None,
        "total_attempts": 0,
        "total_failures": 0,
    }


def _save_state(workspace_root: str, state: dict[str, Any]) -> None:
    p = _state_path(workspace_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(p, state)


def should_run_discovery(
    workspace_root: str = ".",
    *,
    enabled: bool = True,
    interval_hours: float = _DEFAULT_INTERVAL_HOURS,
    every_n_cycles: int = _DEFAULT_EVERY_N_CYCLES,
    min_queue: int = _DEFAULT_MIN_QUEUE,
    trigger_on_stagnation: bool = True,
    stagnation_batches: int = 0,
    current_cycle: int = 0,
) -> tuple[bool, str]:
    """Return (should_run, reason_str).

    Triggers when ANY of the following is true:
    - interval_hours elapsed since last run
    - every_n_cycles cycles have passed
    - active probe queue is below min_queue threshold
    - stagnation detected and trigger_on_stagnation=True
    """
    if not enabled:
        return False, "disabled"

    state = _load_state(workspace_root)
    now = time.time()
    elapsed_hours = (now - state["last_run_ts"]) / 3600.0

    if state["last_run_ts"] == 0.0 and state.get("last_attempt_ts", 0.0) == 0.0:
        return True, "never_run"

    if elapsed_hours >= interval_hours:
        return True, f"interval_elapsed_{elapsed_hours:.1f}h"

    # Apply backoff when discovery has been failing: multiply cycle threshold by 2^min(failures,3)
    consecutive_failures = int(state.get("consecutive_failures", 0))
    effective_every_n = every_n_cycles
    if consecutive_failures >= 3:
        effective_every_n = every_n_cycles * min(8, 2 ** (consecutive_failures - 2))

    cycles_since = state.get("cycles_since_last_run", 0)
    if effective_every_n > 0 and cycles_since >= effective_every_n:
        return True, f"cycle_trigger_{cycles_since}_cycles_backoff={effective_every_n}"

    try:
        from family_candidate_store import get_controlled_probes, get_new_family_candidates
        n_active = len(get_controlled_probes(workspace_root)) + len(get_new_family_candidates(workspace_root))
        if n_active < min_queue:
            return True, f"queue_below_min_{n_active}<{min_queue}"
    except Exception:
        pass

    if trigger_on_stagnation and stagnation_batches >= 2:
        return True, f"stagnation_{stagnation_batches}_batches"

    return False, f"not_due_yet_{elapsed_hours:.1f}h_elapsed"


def should_run_fast_tick(
    workspace_root: str = ".",
    *,
    enabled: bool = True,
    every_n_cycles: int = _DEFAULT_FAST_TICK_EVERY_N_CYCLES,
    interval_minutes: float = _DEFAULT_FAST_TICK_INTERVAL_MINUTES,
) -> tuple[bool, str]:
    """Return (should_run, reason_str) for the fast scheduler tick.

    Fast tick runs when BOTH:
    - every_n_cycles have passed (cycle gate)
    - interval_minutes have elapsed (time gate, prevents spam)
    """
    if not enabled:
        return False, "disabled"

    state = _load_state(workspace_root)
    now = time.time()
    last_fast_ts = state.get("last_fast_tick_ts", 0.0)
    elapsed_minutes = (now - last_fast_ts) / 60.0

    cycles_since = state.get("cycles_since_last_fast_tick", 0)

    if last_fast_ts == 0.0:
        return True, "never_run"

    if elapsed_minutes >= interval_minutes and cycles_since >= every_n_cycles:
        return True, f"due_{elapsed_minutes:.1f}min_{cycles_since}_cycles"

    return False, f"not_due_{elapsed_minutes:.1f}min_elapsed_{cycles_since}_cycles_since"


def run_fast_scheduler_tick(
    workspace_root: str = ".",
    *,
    auto_promote_top: int = 3,
    probe_budget: int = 3,
    max_active_probes: int = 6,
    stale_cycles: int = _DEFAULT_STALE_CYCLES,
    stale_minutes: float = _DEFAULT_STALE_MINUTES,
) -> dict[str, Any]:
    """Run the fast scheduler tick: cheap queue/probe management.

    Handles: stale slot recycling, automatic promotion of waiting candidates,
    probe idea seeding — all without LLM or web research.

    Returns a report dict with tick results.
    """
    from family_candidate_store import (
        get_controlled_probes,
        get_new_family_candidates,
        get_dynamic_probe_cap,
        recycle_stale_probe_slots,
        update_family_candidate_status,
        utc_now_iso as _utc_now,
    )
    from agents.schemas import utc_now_iso

    state = _load_state(workspace_root)
    now = time.time()

    tick_report = {
        "ts": utc_now_iso(),
        "timestamp_unix": now,
        "promotions": [],
        "slot_releases": [],
        "slot_usage_before": len(get_controlled_probes(workspace_root)),
        "slot_usage_after": 0,
        "waiting_before": len(get_new_family_candidates(workspace_root)),
        "waiting_after": 0,
        "recycled_slots": 0,
        "new_promotions": 0,
        "fast_tick_cycle": state.get("fast_tick_cycle_count", 0),
        "status": "ok",
    }

    releases = recycle_stale_probe_slots(workspace_root)
    tick_report["slot_releases"] = releases
    tick_report["recycled_slots"] = len(releases)

    effective_cap = get_dynamic_probe_cap(workspace_root)
    current_probes = get_controlled_probes(workspace_root)
    free_slots = max(0, effective_cap - len(current_probes))

    waiting = get_new_family_candidates(workspace_root)
    tick_report["waiting_before"] = len(waiting)

    newly_promoted: list[dict] = []
    if free_slots > 0 and waiting:
        to_promote = sorted(
            waiting,
            key=lambda c: (c.get("ranking_score") or 0.0, c.get("candidate_id") or ""),
            reverse=True,
        )[: min(free_slots, auto_promote_top, len(waiting))]

        for rec in to_promote:
            candidate_id = rec.get("candidate_id")
            family_name = rec.get("family_name")
            update_family_candidate_status(
                candidate_id,
                "controlled_probe",
                workspace_root,
                extra_updates={"promoted_at": utc_now_iso(), "promotion_reason": "fast_scheduler_tick"},
            )
            tick_report["promotions"].append({
                "candidate_id": candidate_id,
                "family_name": family_name,
                "ranking_score": rec.get("ranking_score"),
            })
            tick_report["new_promotions"] += 1
            newly_promoted.append(rec)

    tick_report["slot_usage_after"] = len(get_controlled_probes(workspace_root))
    tick_report["waiting_after"] = len(get_new_family_candidates(workspace_root))

    # Seed ideas for promoted-but-unseeded controlled_probe candidates.
    # This covers two cases: newly promoted above + candidates that were promoted
    # in prior runs (slow or fast) but missed idea seeding due to probe_budget limits.
    tick_report["ideas_seeded"] = 0
    try:
        from agents.family_discovery_agent import _seed_idea_from_candidate
        from agents.schemas import save_idea_record, FamilyCandidateRecord, update_family_candidate_status as _update_status
        import json as _json
        from pathlib import Path as _Path

        ideas_dir = _Path(workspace_root) / "queues" / "ideas"
        # Build set of family names already seeded
        seeded_families: set[str] = set()
        if ideas_dir.exists():
            for p in ideas_dir.glob("idea_family_discovery_*.json"):
                try:
                    d = _json.loads(p.read_text())
                    disc_fam = (d.get("metadata") or {}).get("discovery_family_name")
                    if disc_fam:
                        seeded_families.add(disc_fam)
                except Exception:
                    pass

        all_probes = get_controlled_probes(workspace_root)
        unseeded = [
            c for c in all_probes
            if c.get("family_name") and c.get("family_name") not in seeded_families
        ]
        seeds_this_tick = 0
        for c in unseeded[:probe_budget]:
            try:
                # Build a FamilyCandidateRecord only from known fields, supplying
                # safe defaults for fields the JSON may be missing.
                known_fields = FamilyCandidateRecord.__dataclass_fields__
                kwargs = {k: c[k] for k in known_fields if k in c}
                for req in ("why_it_should_exist", "why_not_momentum", "why_not_superstock",
                            "required_data", "implementation_complexity", "expected_holding_horizon",
                            "novelty_reason", "timestamp_utc", "edge_source", "first_test_family_template"):
                    if req not in kwargs:
                        kwargs[req] = {} if req == "first_test_family_template" else ""
                rec = FamilyCandidateRecord(**kwargs)
                idea = _seed_idea_from_candidate(rec, workspace_root)
                if idea is not None:
                    save_idea_record(idea, workspace_root=workspace_root)
                    _update_status(
                        c.get("candidate_id", ""),
                        "controlled_probe",
                        workspace_root,
                        extra_updates={"probe_idea_id": idea.idea_id},
                    )
                    seeds_this_tick += 1
                    seeded_families.add(c.get("family_name", ""))
                    print(
                        f"[fast_scheduler] seeded idea {idea.idea_id} for "
                        f"unseeded probe {c.get('family_name')}",
                        flush=True,
                    )
            except Exception as _exc:
                print(f"[fast_scheduler] seeding failed for {c.get('family_name')}: {_exc}", flush=True)
        tick_report["ideas_seeded"] = seeds_this_tick
    except Exception as _exc:
        tick_report["ideas_seeded_error"] = str(_exc)

    state["last_fast_tick_ts"] = now
    state["last_fast_tick_at"] = utc_now_iso()
    state["fast_tick_cycle_count"] = state.get("fast_tick_cycle_count", 0) + 1
    state["cycles_since_last_fast_tick"] = 0
    state["fast_tick_promotions"] = state.get("fast_tick_promotions", 0) + tick_report["new_promotions"]
    state["fast_tick_slot_releases"] = state.get("fast_tick_slot_releases", 0) + tick_report["recycled_slots"]
    _save_state(workspace_root, state)

    return tick_report


def record_discovery_run(
    workspace_root: str = ".",
    *,
    batch_id: str | None = None,
    n_candidates: int = 0,
    n_promoted: int = 0,
    trigger_reason: str = "manual",
) -> None:
    """Update state after a completed (successful) discovery run."""
    state = _load_state(workspace_root)
    now = time.time()
    state.update({
        "last_run_ts": now,
        "last_run_at": utc_now_iso(),
        "run_count": state.get("run_count", 0) + 1,
        "cycles_since_last_run": 0,
        "last_batch_id": batch_id,
        "last_n_candidates": n_candidates,
        "last_n_promoted": n_promoted,
        "last_trigger_reason": trigger_reason,
        "last_attempt_ts": now,
        "last_attempt_at": utc_now_iso(),
        "total_attempts": state.get("total_attempts", 0) + 1,
        "consecutive_failures": 0,
    })
    _save_state(workspace_root, state)


def record_discovery_attempt(
    workspace_root: str = ".",
    *,
    error: str | None = None,
    trigger_reason: str = "auto",
) -> None:
    """Record a failed discovery attempt; resets the cycle counter to prevent
    every-cycle retries. Consecutive failures increase backoff in should_run_discovery."""
    state = _load_state(workspace_root)
    now = time.time()
    consecutive = state.get("consecutive_failures", 0) + 1
    state.update({
        "last_attempt_ts": now,
        "last_attempt_at": utc_now_iso(),
        "cycles_since_last_run": 0,
        "total_attempts": state.get("total_attempts", 0) + 1,
        "total_failures": state.get("total_failures", 0) + 1,
        "consecutive_failures": consecutive,
        "last_failure_reason": error or "unknown",
        "last_trigger_reason": trigger_reason,
    })
    _save_state(workspace_root, state)


def increment_cycle_counter(workspace_root: str = ".") -> int:
    """Increment both fast and slow cycle counters."""
    state = _load_state(workspace_root)
    state["cycles_since_last_run"] = state.get("cycles_since_last_run", 0) + 1
    state["cycles_since_last_fast_tick"] = state.get("cycles_since_last_fast_tick", 0) + 1
    _save_state(workspace_root, state)
    return state["cycles_since_last_run"]


def get_scheduler_status(workspace_root: str = ".") -> dict[str, Any]:
    """Return combined fast+slow scheduler state for health reporting."""
    state = _load_state(workspace_root)
    now = time.time()

    last_run_ts = state.get("last_run_ts", 0.0)
    elapsed_hours = (now - last_run_ts) / 3600.0 if last_run_ts else None

    last_fast_ts = state.get("last_fast_tick_ts", 0.0)
    elapsed_fast_minutes = (now - last_fast_ts) / 60.0 if last_fast_ts else None

    try:
        from family_candidate_store import get_controlled_probes, get_new_family_candidates, get_dynamic_probe_cap
        active_probes = len(get_controlled_probes(workspace_root))
        waiting = len(get_new_family_candidates(workspace_root))
        dynamic_cap = get_dynamic_probe_cap(workspace_root)
    except Exception:
        active_probes = waiting = dynamic_cap = None

    return {
        "slow": {
            "last_run_at": state.get("last_run_at"),
            "run_count": state.get("run_count", 0),
            "cycles_since_last_run": state.get("cycles_since_last_run", 0),
            "elapsed_hours": round(elapsed_hours, 2) if elapsed_hours is not None else None,
            "last_n_candidates": state.get("last_n_candidates", 0),
            "last_n_promoted": state.get("last_n_promoted", 0),
            "last_trigger_reason": state.get("last_trigger_reason"),
            "last_batch_id": state.get("last_batch_id"),
        },
        "fast": {
            "last_fast_tick_at": state.get("last_fast_tick_at"),
            "elapsed_minutes": round(elapsed_fast_minutes, 1) if elapsed_fast_minutes is not None else None,
            "fast_tick_cycle_count": state.get("fast_tick_cycle_count", 0),
            "cycles_since_last_fast_tick": state.get("cycles_since_last_fast_tick", 0),
            "total_fast_promotions": state.get("fast_tick_promotions", 0),
            "total_fast_slot_releases": state.get("fast_tick_slot_releases", 0),
        },
        "probe_slots": {
            "active": active_probes,
            "waiting": waiting,
            "cap": dynamic_cap,
            "free_slots": max(0, (dynamic_cap or 6) - (active_probes or 0)),
        },
        "discovery_due": should_run_discovery(workspace_root)[0],
        "fast_tick_due": should_run_fast_tick(workspace_root)[0],
    }

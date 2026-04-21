"""Health reporting for the family discovery pipeline.

Aggregates scheduler state, queue stats, probe coverage in experiment index,
conversion funnel, and prints a compact status report.
"""
from __future__ import annotations

from typing import Any

from family_candidate_store import family_candidate_summary, get_controlled_probes
from family_discovery_scheduler import get_discovery_status


def build_health_report(workspace_root: str = ".") -> dict[str, Any]:
    """Aggregate all discovery pipeline state into one dict."""
    sched = get_discovery_status(workspace_root)
    queue = family_candidate_summary(workspace_root)

    # Count probe ideas in queues/ideas/
    probe_ideas = _count_probe_ideas(workspace_root)

    # Count how many controlled-probe configs appeared in index.csv
    probe_experiments = _count_probe_experiments(workspace_root)

    # Full conversion funnel
    funnel = _build_funnel_summary(workspace_root)

    return {
        "scheduler": sched,
        "queue": queue,
        "probe_ideas_queued": probe_ideas,
        "probe_experiments_run": probe_experiments,
        "funnel": funnel,
    }


def print_health_report(workspace_root: str = ".") -> None:
    r = build_health_report(workspace_root)
    sched = r["scheduler"]
    queue = r["queue"]

    print("=" * 56)
    print("FAMILY DISCOVERY HEALTH")
    print("=" * 56)

    # Scheduler state
    last = sched.get("last_run_at") or "never"
    n_runs = sched.get("run_count", 0)
    elapsed = sched.get("elapsed_hours")
    elapsed_str = f"{elapsed:.1f}h ago" if elapsed is not None else "n/a"
    print(f"  last_run        : {last}  ({elapsed_str})")
    print(f"  total_runs      : {n_runs}")
    print(f"  cycles_since    : {sched.get('cycles_since_last_run', 0)}")
    print(f"  trigger_reason  : {sched.get('last_trigger_reason', 'n/a')}")

    # Queue stats
    total = queue.get("total", 0)
    by_status = queue.get("by_status", {})
    print(f"\n  candidates_total: {total}")
    for status, cnt in sorted(by_status.items()):
        print(f"    {status:<30}: {cnt}")

    # Probe idea / experiment counts
    print(f"\n  probe_ideas_queued  : {r['probe_ideas_queued']}")
    print(f"  probe_experiments_run: {r['probe_experiments_run']}")

    # Top candidates
    top = queue.get("top_candidates", [])
    if top:
        print(f"\n  top_candidates (by ranking_score):")
        for c in top[:5]:
            score = c.get("ranking_score")
            score_str = f"{score:.3f}" if score is not None else "n/a"
            orth = c.get("orthogonality_score")
            orth_str = f"{orth:.2f}" if orth is not None else "n/a"
            print(f"    [{score_str}] {c.get('family_name', '?'):<32} orth={orth_str}  status={c.get('status')}")

    # Conversion funnel summary
    funnel = r.get("funnel") or {}
    stages = funnel.get("stages") or []
    if stages:
        print(f"\n  conversion funnel:")
        for s in stages:
            print(f"    {s['stage']:<26} {s['count']:>5}  ({s['conversion']})")
        bn = funnel.get("bottleneck")
        if bn:
            print(f"  [!] bottleneck: {bn}")
        else:
            direct = funnel.get("direct_synthesis_count", 0)
            viable = funnel.get("viable", 0)
            beat = funnel.get("baseline_beating", 0)
            print(f"  direct_synthesis={direct}  viable={viable}  baseline_beating={beat}")

    print("=" * 56)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_probe_ideas(workspace_root: str) -> int:
    from pathlib import Path
    import json
    ideas_dir = Path(workspace_root) / "queues" / "ideas"
    if not ideas_dir.exists():
        return 0
    count = 0
    for f in ideas_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("idea_source") == "family_discovery":
                count += 1
        except Exception:
            pass
    return count


def _build_funnel_summary(workspace_root: str) -> dict[str, Any]:
    """Return a compact funnel summary for the health report."""
    try:
        from family_discovery_funnel import build_funnel
        full = build_funnel(workspace_root)
        compact_stages = [
            {"stage": s["stage"], "count": s["count"], "conversion": s["conversion"]}
            for s in (full.get("funnel_stages") or [])
        ]
        return {
            "stages": compact_stages,
            "bottleneck": full.get("bottleneck"),
            "direct_synthesis_count": full.get("direct_synthesis_count", 0),
            "viable": full.get("total_viable", 0),
            "baseline_beating": full.get("total_baseline_beating", 0),
            "top_experiments": full.get("top_attributed_experiments") or [],
        }
    except Exception as exc:
        return {"error": str(exc)}


def _count_probe_experiments(workspace_root: str) -> int:
    from pathlib import Path
    import csv
    index_path = Path(workspace_root) / "experiments" / "index.csv"
    if not index_path.exists():
        return 0
    count = 0
    try:
        with open(index_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                notes = row.get("notes", "") or ""
                if "family_discovery" in notes or "family_probe" in notes:
                    count += 1
    except Exception:
        pass
    return count

"""
Adaptive workload policy for the autonomous research loop.

Reads health signals (memory, disk, recent fill rates, error counts)
and returns recommended batch_size and max_workers for the next cycle.

This module is consulted by autonomous_runner.py and research_loop.sh.
It never changes strategy logic, backtests, or guardrails.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Defaults (conservative starting point)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE: int = 36
DEFAULT_MAX_WORKERS: int = 8
DEFAULT_MIN_BATCH_SIZE: int = 6
DEFAULT_MIN_WORKERS: int = 2
DEFAULT_MAX_BATCH_SIZE: int = 48
DEFAULT_MAX_WORKERS_CAP: int = 8

# Memory guardrails
MEMORY_BACKOFF_AVAILABLE_GB: float = 8.0   # back off if available RAM drops below
MEMORY_CRITICAL_AVAILABLE_GB: float = 5.0  # hard floor: reduce workers to 4

# Disk guardrails
DISK_BACKOFF_FREE_GB: float = 10.0   # back off if free disk drops below
DISK_CRITICAL_FREE_GB: float = 6.0   # hard floor: minimal batch

# Fill rate: if experiments executed / requested < this, flag as underutilised
FILL_RATE_LOW_THRESHOLD: float = 0.50

# Recent error threshold: if >50% of last N cycles errored, back off
ERROR_RATE_BACKOFF_THRESHOLD: float = 0.50

# Backlog thresholds
BACKLOG_HIGH_THRESHOLD: int = 40   # active queue > this → idea generation is ahead, keep executions high
BACKLOG_LOW_THRESHOLD: int = 5     # active queue < this → slow down idea intake

# Workers/batch caps
MAX_WORKERS_CAP: int = 8
MIN_WORKERS: int = 2
MAX_BATCH_SIZE_CAP: int = 48
MIN_BATCH_SIZE: int = 6


# ---------------------------------------------------------------------------
# Health signals
# ---------------------------------------------------------------------------

def _read_proc_meminfo() -> dict[str, float]:
    """Read /proc/meminfo and return key values in GB."""
    result: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    kb = float(parts[1])
                    result[key] = kb / 1024 / 1024  # → GB
    except Exception:
        pass
    return result


def _disk_free_gb(path: str) -> float:
    """Return free disk space in GB for the given path."""
    try:
        import shutil
        return shutil.disk_usage(path).free / 1024 / 1024 / 1024
    except Exception:
        return 999.0


def _recent_batch_stats(base_dir: str, n_batches: int = 10) -> dict[str, Any]:
    """Load the last n_batches batch summaries and compute fill rate and error count."""
    batches_path = Path(base_dir) / "batches"
    if not batches_path.exists():
        return {
            "fill_rate": 1.0,
            "error_rate": 0.0,
            "avg_executed": 0.0,
            "executed_per_minute": 0.0,
            "batch_runtime_seconds": 0.0,
            "n_checked": 0,
        }

    batch_dirs = sorted(batches_path.glob("*/"), reverse=True)[:n_batches]
    total_executed = 0
    total_requested = 0
    total_runtime_seconds = 0.0
    total_executed_per_minute = 0.0
    n_checked = 0
    for bd in batch_dirs:
        summary = bd / "summary.json"
        if not summary.exists():
            continue
        try:
            s = json.loads(summary.read_text())
            executed = int(s.get("total_executed") or 0)
            requested = int(s.get("total_sampled") or executed or 1)
            total_executed += executed
            total_requested += max(requested, executed)
            total_runtime_seconds += float(s.get("batch_runtime_seconds") or 0.0)
            total_executed_per_minute += float(s.get("executed_per_minute") or 0.0)
            n_checked += 1
        except Exception:
            continue

    fill_rate = total_executed / total_requested if total_requested > 0 else 1.0
    avg_executed = total_executed / n_checked if n_checked > 0 else 0
    return {
        "fill_rate": fill_rate,
        "avg_executed": avg_executed,
        "executed_per_minute": total_executed_per_minute / n_checked if n_checked > 0 else 0.0,
        "batch_runtime_seconds": total_runtime_seconds / n_checked if n_checked > 0 else 0.0,
        "n_checked": n_checked,
    }


def _recent_error_rate(workspace_root: str, window: int = 10) -> tuple[float, int]:
    """
    Estimate recent error rate from the research_status.txt file.
    Returns (error_rate, recent_rc_failures).
    """
    status_path = Path(workspace_root) / "run" / "research_status.txt"
    if not status_path.exists():
        return 0.0, 0
    try:
        content = status_path.read_text()
        for line in content.splitlines():
            if line.startswith("last_rc="):
                rc = line.split("=", 1)[1].strip()
                recent_failures = 1 if rc not in ("0", "") else 0
                return float(recent_failures), recent_failures
    except Exception:
        pass
    return 0.0, 0


def _active_region_pressure(workspace_root: str) -> tuple[int, float]:
    try:
        from experiment_promoted_regions import load_promoted_regions

        regions = load_promoted_regions(workspace_root=workspace_root)
        active = [
            region for region in regions
            if region.get("ready_for_follow_up")
            and str((region.get("region_health") or {}).get("region_health_state") or "") in {"strengthening", "stable"}
        ]
        pressure = min(1.0, len(active) / 4.0)
        return len(active), pressure
    except Exception:
        return 0, 0.0


def _active_idea_count(workspace_root: str) -> int:
    """Count active (non-terminal) ideas in the queue."""
    try:
        from experiment_idea_manager import load_active_idea_records
        return len(load_active_idea_records(workspace_root))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Workload state dataclass
# ---------------------------------------------------------------------------

@dataclass
class WorkloadState:
    timestamp_utc: str
    workload_state: str              # healthy / elevated_memory / elevated_disk / degraded
    recommended_batch_size: int
    recommended_max_workers: int
    throughput_reason: str
    backlog_pressure_state: str      # low / normal / high
    available_memory_gb: float
    free_disk_gb: float
    recent_fill_rate: float
    recent_avg_executed: float
    recent_executed_per_minute: float
    recent_rc_failures: int
    active_idea_count: int
    active_region_count: int
    memory_pressure_state: str
    disk_pressure_state: str
    signals: dict[str, Any]


def assess_workload(
    *,
    workspace_root: str = ".",
    base_dir: str = "experiments",
    target_batch_size: int = DEFAULT_BATCH_SIZE,
    target_max_workers: int = DEFAULT_MAX_WORKERS,
) -> WorkloadState:
    """
    Evaluate health signals and return the recommended workload settings.

    Call this at the start of each cycle; the result drives batch_size and
    max_workers for that cycle.
    """
    now = datetime.now(UTC).isoformat()
    mem = _read_proc_meminfo()
    available_gb = mem.get("MemAvailable", 999.0)
    free_disk = _disk_free_gb(workspace_root)
    batch_stats = _recent_batch_stats(base_dir)
    fill_rate = batch_stats["fill_rate"]
    avg_executed = batch_stats["avg_executed"]
    executed_per_minute = float(batch_stats.get("executed_per_minute") or 0.0)
    batch_runtime_seconds = float(batch_stats.get("batch_runtime_seconds") or 0.0)
    error_rate, recent_rc_failures = _recent_error_rate(workspace_root)
    active_ideas = _active_idea_count(workspace_root)
    active_regions, active_region_pressure = _active_region_pressure(workspace_root)

    signals = {
        "available_memory_gb": round(available_gb, 2),
        "free_disk_gb": round(free_disk, 2),
        "recent_fill_rate": round(fill_rate, 3),
        "recent_avg_executed": round(avg_executed, 1),
        "recent_executed_per_minute": round(executed_per_minute, 3),
        "recent_batch_runtime_seconds": round(batch_runtime_seconds, 1),
        "recent_error_rate": round(error_rate, 3),
        "recent_rc_failures": recent_rc_failures,
        "active_idea_count": active_ideas,
        "active_region_count": active_regions,
        "active_region_pressure": round(active_region_pressure, 3),
    }

    # Determine workload state
    workload_state = "healthy"
    reasons: list[str] = []
    memory_pressure_state = "normal"
    disk_pressure_state = "normal"

    # Memory pressure
    if available_gb < MEMORY_CRITICAL_AVAILABLE_GB:
        workload_state = "degraded"
        memory_pressure_state = "critical"
        reasons.append(f"critical_memory available={available_gb:.1f}GB")
    elif available_gb < MEMORY_BACKOFF_AVAILABLE_GB:
        workload_state = "elevated_memory"
        memory_pressure_state = "elevated"
        reasons.append(f"elevated_memory available={available_gb:.1f}GB")

    # Disk pressure
    if free_disk < DISK_CRITICAL_FREE_GB:
        workload_state = "degraded"
        disk_pressure_state = "critical"
        reasons.append(f"critical_disk free={free_disk:.1f}GB")
    elif free_disk < DISK_BACKOFF_FREE_GB:
        disk_pressure_state = "elevated"
        if workload_state == "healthy":
            workload_state = "elevated_disk"
        reasons.append(f"elevated_disk free={free_disk:.1f}GB")

    # Error rate
    if error_rate >= ERROR_RATE_BACKOFF_THRESHOLD:
        workload_state = "degraded"
        reasons.append(f"high_error_rate={error_rate:.2f}")

    # Backlog pressure
    backlog_score = active_ideas + (active_regions * 2)
    if backlog_score > BACKLOG_HIGH_THRESHOLD:
        backlog_state = "high"
        reasons.append(f"backlog_high ideas={active_ideas} regions={active_regions}")
    elif backlog_score < BACKLOG_LOW_THRESHOLD:
        backlog_state = "low"
    else:
        backlog_state = "normal"

    # Compute recommended settings
    rec_batch = target_batch_size
    rec_workers = target_max_workers
    health_is_good = workload_state == "healthy" and memory_pressure_state == "normal" and disk_pressure_state == "normal" and error_rate < ERROR_RATE_BACKOFF_THRESHOLD

    if workload_state == "degraded":
        rec_batch = max(DEFAULT_MIN_BATCH_SIZE, min(target_batch_size, 12))
        rec_workers = max(DEFAULT_MIN_WORKERS, min(target_max_workers, 4))
        reasons.append("backoff=severe")
    elif workload_state in {"elevated_memory", "elevated_disk"}:
        rec_batch = max(DEFAULT_MIN_BATCH_SIZE, min(target_batch_size, int(round(target_batch_size * 0.80))))
        rec_workers = max(DEFAULT_MIN_WORKERS, min(target_max_workers, 6))
        reasons.append("backoff=moderate")
    else:
        rec_batch = min(DEFAULT_MAX_BATCH_SIZE, target_batch_size)
        rec_workers = min(DEFAULT_MAX_WORKERS_CAP, target_max_workers)
        if health_is_good and backlog_state == "high" and fill_rate >= 0.80 and executed_per_minute >= 6.0:
            rec_batch = min(DEFAULT_MAX_BATCH_SIZE, max(rec_batch, int(round(target_batch_size * 1.10))))
            rec_workers = min(DEFAULT_MAX_WORKERS_CAP, max(rec_workers, target_max_workers + 1))
            reasons.append(
                f"throughput_boost backlog_high fill_rate={fill_rate:.2f} executed_per_minute={executed_per_minute:.2f}"
            )
        elif health_is_good and backlog_state == "normal" and fill_rate >= 0.75 and avg_executed >= rec_batch * 0.75:
            rec_batch = min(DEFAULT_MAX_BATCH_SIZE, max(rec_batch, int(round(target_batch_size * 1.05))))
            reasons.append(f"throughput_steady fill_rate={fill_rate:.2f}")
        elif fill_rate < FILL_RATE_LOW_THRESHOLD:
            rec_workers = max(DEFAULT_MIN_WORKERS, min(rec_workers, 6))
            reasons.append(f"fill_rate_low={fill_rate:.2f}_capping_workers")
        if active_region_pressure >= 0.5 and backlog_state != "low":
            rec_batch = min(DEFAULT_MAX_BATCH_SIZE, max(rec_batch, int(round(target_batch_size * 1.05))))
            reasons.append(f"active_region_pressure={active_region_pressure:.2f}")

    throughput_reason = "; ".join(reasons) if reasons else "healthy_nominal"

    return WorkloadState(
        timestamp_utc=now,
        workload_state=workload_state,
        recommended_batch_size=rec_batch,
        recommended_max_workers=rec_workers,
        throughput_reason=throughput_reason,
        backlog_pressure_state=backlog_state,
        available_memory_gb=round(available_gb, 2),
        free_disk_gb=round(free_disk, 2),
        recent_fill_rate=round(fill_rate, 3),
        recent_avg_executed=round(avg_executed, 1),
        recent_executed_per_minute=round(executed_per_minute, 3),
        recent_rc_failures=recent_rc_failures,
        active_idea_count=active_ideas,
        active_region_count=active_regions,
        memory_pressure_state=memory_pressure_state,
        disk_pressure_state=disk_pressure_state,
        signals=signals,
    )


def save_workload_state(state: WorkloadState, workspace_root: str = ".") -> Path:
    """Persist the workload state to run/workload_policy.json for inspection."""
    path = Path(workspace_root) / "run" / "workload_policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))
    return path


def load_workload_state(workspace_root: str = ".") -> dict[str, Any] | None:
    """Load the last saved workload state, or None if not found."""
    path = Path(workspace_root) / "run" / "workload_policy.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

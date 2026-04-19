from __future__ import annotations

import argparse
import json
import os
import shlex
import resource
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryBudgetDecision:
    batch_size: int
    max_workers: int
    reason: str
    memory_pressure_level: str
    mem_total_kb: int | None
    mem_available_kb: int | None
    last_rc: int | None
    last_state: str | None
    requested_batch_size: int
    requested_max_workers: int
    process_rss_kb: int | None = None
    process_peak_rss_kb: int | None = None
    memory_budget_reason: str | None = None


def _parse_status_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    status_path = Path(path)
    if not status_path.exists():
        return {}
    payload: dict[str, str] = {}
    for raw_line in status_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def _parse_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    meminfo: dict[str, int] = {}
    meminfo_path = Path(path)
    if not meminfo_path.exists():
        return meminfo
    for raw_line in meminfo_path.read_text().splitlines():
        if ":" not in raw_line:
            continue
        key, rest = raw_line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            meminfo[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return meminfo


def _parse_process_status(path: str = "/proc/self/status") -> dict[str, int]:
    status_path = Path(path)
    if not status_path.exists():
        return {}
    payload: dict[str, int] = {}
    for raw_line in status_path.read_text().splitlines():
        if ":" not in raw_line:
            continue
        key, rest = raw_line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            payload[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return payload


def _parse_child_pids(pid: int | None = None) -> list[int]:
    if pid is None:
        pid = os.getpid()
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    if not children_path.exists():
        return []
    raw = children_path.read_text().strip()
    if not raw:
        return []
    child_pids: list[int] = []
    for token in raw.split():
        try:
            child_pids.append(int(token))
        except ValueError:
            continue
    return child_pids


def _process_tree_rss_kb(pid: int | None = None) -> int | None:
    if pid is None:
        pid = os.getpid()
    seen: set[int] = set()
    stack = [pid]
    total = 0
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        status_path = Path(f"/proc/{current}/status")
        if not status_path.exists():
            continue
        rss_kb = None
        for raw_line in status_path.read_text().splitlines():
            if raw_line.startswith("VmRSS:"):
                parts = raw_line.split()
                if len(parts) >= 2:
                    try:
                        rss_kb = int(parts[1])
                    except ValueError:
                        rss_kb = None
                break
        if rss_kb is not None:
            total += rss_kb
        stack.extend(_parse_child_pids(current))
    return total if total > 0 else None


def current_process_memory_kb() -> dict[str, int | None]:
    status = _parse_process_status()
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss_kb = int(getattr(usage, "ru_maxrss", 0) or 0)
    except Exception:  # pragma: no cover - defensive
        peak_rss_kb = None
    return {
        "rss_kb": status.get("VmRSS"),
        "peak_rss_kb": peak_rss_kb,
        "vmsize_kb": status.get("VmSize"),
        "shared_kb": status.get("VmRSS", 0) - status.get("RssAnon", 0) if status.get("VmRSS") is not None and status.get("RssAnon") is not None else None,
        "tree_rss_kb": _process_tree_rss_kb(),
    }


def choose_memory_budget(
    *,
    requested_batch_size: int,
    requested_max_workers: int,
    status: dict[str, str] | None = None,
    meminfo: dict[str, int] | None = None,
    cycle_mode: str | None = None,
    confirmation_required: bool = False,
    holdout_check_required: bool = False,
    large_search_mode: bool = False,
) -> MemoryBudgetDecision:
    status = status or {}
    meminfo = meminfo or {}

    last_rc_raw = status.get("last_rc")
    last_state = str(status.get("state") or "").strip().lower() or None
    try:
        last_rc = int(last_rc_raw) if last_rc_raw is not None else None
    except ValueError:
        last_rc = None

    mem_total_kb = meminfo.get("MemTotal")
    mem_available_kb = meminfo.get("MemAvailable")
    mem_ratio = (mem_available_kb / mem_total_kb) if mem_total_kb and mem_available_kb else None

    batch_size = int(requested_batch_size)
    max_workers = int(requested_max_workers)
    reason_parts: list[str] = []
    pressure_level = "normal"

    if cycle_mode in {"confirmation", "holdout", "diagnostics"} or confirmation_required or holdout_check_required:
        batch_size = min(batch_size, 8)
        max_workers = min(max_workers, 1)
        reason_parts.append("confirmation_or_holdout_small_batch")
        pressure_level = "restricted"
    elif large_search_mode:
        batch_size = min(batch_size, 24)
        max_workers = min(max_workers, 4)
        reason_parts.append("large_search_worker_cap")

    if last_rc == 137 or last_state == "error":
        batch_size = min(batch_size, 12)
        max_workers = min(max_workers, 1)
        reason_parts.append("oom_backoff")
        pressure_level = "critical"
    elif mem_ratio is not None:
        if mem_ratio < 0.25:
            batch_size = min(batch_size, 12)
            max_workers = min(max_workers, 1)
            reason_parts.append("critical_memory_pressure")
            pressure_level = "critical"
        elif mem_ratio < 0.35:
            batch_size = min(batch_size, 16)
            max_workers = min(max_workers, 1)
            reason_parts.append("high_memory_pressure")
            pressure_level = "high"
        elif mem_ratio < 0.45:
            batch_size = min(batch_size, 24)
            max_workers = min(max_workers, 2)
            reason_parts.append("moderate_memory_pressure")
            pressure_level = "elevated"
        elif mem_ratio < 0.60:
            batch_size = min(batch_size, 24)
            max_workers = min(max_workers, 3)
            reason_parts.append("conservative_memory_pressure")
            pressure_level = "moderate"

    if requested_max_workers > 1 and batch_size < 12:
        max_workers = min(max_workers, 1)

    batch_size = max(1, batch_size)
    max_workers = max(1, max_workers)
    if not reason_parts:
        reason_parts.append("memory_budget_normal")

    process_memory = current_process_memory_kb()

    return MemoryBudgetDecision(
        batch_size=batch_size,
        max_workers=max_workers,
        reason=";".join(reason_parts),
        memory_pressure_level=pressure_level,
        mem_total_kb=mem_total_kb,
        mem_available_kb=mem_available_kb,
        last_rc=last_rc,
        last_state=last_state,
        requested_batch_size=int(requested_batch_size),
        requested_max_workers=int(requested_max_workers),
        process_rss_kb=process_memory.get("rss_kb"),
        process_peak_rss_kb=process_memory.get("peak_rss_kb"),
        memory_budget_reason=";".join(reason_parts),
    )


def emit_shell(decision: MemoryBudgetDecision) -> str:
    lines = {
        "BATCH_SIZE": str(decision.batch_size),
        "MAX_WORKERS": str(decision.max_workers),
        "MEMORY_BUDGET_REASON": decision.reason,
        "MEMORY_PRESSURE_LEVEL": decision.memory_pressure_level,
        "MEMORY_TOTAL_KB": "" if decision.mem_total_kb is None else str(decision.mem_total_kb),
        "MEMORY_AVAILABLE_KB": "" if decision.mem_available_kb is None else str(decision.mem_available_kb),
        "MEMORY_LAST_RC": "" if decision.last_rc is None else str(decision.last_rc),
        "MEMORY_LAST_STATE": "" if decision.last_state is None else str(decision.last_state),
        "MEMORY_REQUESTED_BATCH_SIZE": str(decision.requested_batch_size),
        "MEMORY_REQUESTED_MAX_WORKERS": str(decision.requested_max_workers),
        "MEMORY_RSS_KB": "" if decision.process_rss_kb is None else str(decision.process_rss_kb),
        "MEMORY_PEAK_RSS_KB": "" if decision.process_peak_rss_kb is None else str(decision.process_peak_rss_kb),
    }
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in lines.items())


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose memory-safe unattended research limits")
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--meminfo-file", default="/proc/meminfo")
    parser.add_argument("--base-batch-size", type=int, default=24)
    parser.add_argument("--base-max-workers", type=int, default=2)
    parser.add_argument("--cycle-mode", default=None)
    parser.add_argument("--confirmation-required", action="store_true")
    parser.add_argument("--holdout-check-required", action="store_true")
    parser.add_argument("--large-search-mode", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    decision = choose_memory_budget(
        requested_batch_size=args.base_batch_size,
        requested_max_workers=args.base_max_workers,
        status=_parse_status_file(args.status_file),
        meminfo=_parse_meminfo(args.meminfo_file),
        cycle_mode=args.cycle_mode,
        confirmation_required=args.confirmation_required,
        holdout_check_required=args.holdout_check_required,
        large_search_mode=args.large_search_mode,
    )

    if args.json:
        print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    else:
        print(emit_shell(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

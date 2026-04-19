#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from experiment_hot_index import (
    cleanup_validated_batch_artifacts,
    cleanup_validated_proposal_json_exports,
    validate_hot_batch_metadata,
)


DEFAULT_LOG_MAX_BYTES = 50_000_000
DEFAULT_LOG_RETAIN_COUNT = 4
DEFAULT_BATCH_SUMMARY_THRESHOLD_BYTES = 25_000_000
DEFAULT_KEEP_RECENT_PROPOSALS = 200
DEFAULT_KEEP_RECENT_BATCHES = 24
MIN_LOG_RETAIN_COUNT = 1


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _archive_paths(path: Path, retain_count: int) -> list[Path]:
    return [
        path.with_name(f"{path.name}.{index}")
        for index in range(1, retain_count + 1)
    ]


def _archive_index(path: Path) -> int | None:
    suffix = path.suffix.lstrip(".")
    return int(suffix) if suffix.isdigit() else None


def _copy_tail(source: Path, target: Path, max_bytes: int) -> None:
    size = source.stat().st_size
    offset = max(0, size - max_bytes)
    with source.open("rb") as src, target.open("wb") as dst:
        if offset > 0:
            src.seek(offset)
            src.readline()
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def _trim_log_file(
    path: Path,
    *,
    max_bytes: int,
    dry_run: bool,
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "trimmed": False,
            "before_bytes": 0,
            "after_bytes": 0,
        }

    before = path.stat().st_size
    if before <= max_bytes:
        return {
            "path": str(path),
            "exists": True,
            "trimmed": False,
            "before_bytes": before,
            "after_bytes": before,
        }

    if not dry_run:
        tmp_path = path.with_name(f"{path.name}.tail.tmp")
        tmp_path.unlink(missing_ok=True)
        _copy_tail(path, tmp_path, max_bytes)
        os.replace(tmp_path, path)

    return {
        "path": str(path),
        "exists": True,
        "trimmed": True,
        "before_bytes": before,
        "after_bytes": min(before, max_bytes),
    }


def _archive_candidates(path: Path) -> list[Path]:
    candidates = []
    for candidate in path.parent.glob(f"{path.name}.*"):
        if _archive_index(candidate) is not None:
            candidates.append(candidate)
    return candidates


def _rotate_log_file(
    path: Path,
    *,
    max_bytes: int,
    retain_count: int,
    dry_run: bool,
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "rotated": False,
            "before_bytes": 0,
            "after_bytes": 0,
            "archive_paths": [],
        }
    before = path.stat().st_size
    if before <= max_bytes:
        return {
            "path": str(path),
            "exists": True,
            "rotated": False,
            "before_bytes": before,
            "after_bytes": before,
            "archive_paths": [],
    }

    retain_count = max(MIN_LOG_RETAIN_COUNT, int(retain_count))
    archive_paths = _archive_paths(path, retain_count)
    if not dry_run:
        archives = []
        for candidate in _archive_candidates(path):
            index = _archive_index(candidate)
            if index is not None:
                archives.append((index, candidate))
        for index, candidate in sorted(archives, reverse=True):
            if index >= retain_count:
                candidate.unlink(missing_ok=True)
                continue
            dst = path.with_name(f"{path.name}.{index + 1}")
            os.replace(candidate, dst)

        tmp_archive = path.with_name(f"{path.name}.1.tmp")
        tmp_archive.unlink(missing_ok=True)
        _copy_tail(path, tmp_archive, max_bytes)
        os.replace(tmp_archive, path.with_name(f"{path.name}.1"))
        with path.open("r+b") as handle:
            handle.truncate(0)
    return {
        "path": str(path),
        "exists": True,
        "rotated": True,
        "before_bytes": before,
        "after_bytes": 0,
        "archive_paths": [str(item) for item in archive_paths],
    }


def rotate_logs(
    *,
    log_dir: Path,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    retain_count: int = DEFAULT_LOG_RETAIN_COUNT,
    dry_run: bool = False,
) -> dict[str, Any]:
    paths = [
        log_dir / "autonomous_research.log",
        log_dir / "cron_watchdog.log",
        log_dir / "tmux_watchdog.log",
    ]
    effective_retain_count = max(MIN_LOG_RETAIN_COUNT, int(retain_count))
    trim_reports = [
        _trim_log_file(candidate, max_bytes=max_bytes, dry_run=dry_run)
        for path in paths
        for candidate in _archive_candidates(path)
    ]
    reports = [
        _rotate_log_file(
            path,
            max_bytes=max_bytes,
            retain_count=effective_retain_count,
            dry_run=dry_run,
        )
        for path in paths
    ]
    return {
        "dry_run": dry_run,
        "policy": {
            "max_bytes": max_bytes,
            "requested_retain_count": retain_count,
            "effective_retain_count": effective_retain_count,
            "active_log": str(log_dir / "autonomous_research.log"),
            "retained_logs": [
                str(log_dir / f"autonomous_research.log.{index}")
                for index in range(1, effective_retain_count + 1)
            ],
        },
        "retained_count": effective_retain_count,
        "trimmed_count": sum(1 for item in trim_reports if item.get("trimmed")),
        "trimmed_bytes": sum(
            max(0, int(item.get("before_bytes") or 0) - int(item.get("after_bytes") or 0))
            for item in trim_reports
            if item.get("trimmed")
        ),
        "trimmed_logs": [item for item in trim_reports if item.get("trimmed")],
        "rotated_count": sum(1 for item in reports if item.get("rotated")),
        "archived_bytes": sum(
            max(0, int(item.get("before_bytes") or 0))
            for item in reports
            if item.get("rotated")
        ),
        "logs": reports,
    }


def run_maintenance(
    *,
    base_dir: Path,
    log_dir: Path,
    keep_recent_proposals: int = DEFAULT_KEEP_RECENT_PROPOSALS,
    keep_recent_batches: int = DEFAULT_KEEP_RECENT_BATCHES,
    min_summary_bytes: int = DEFAULT_BATCH_SUMMARY_THRESHOLD_BYTES,
    log_max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    log_retain_count: int = DEFAULT_LOG_RETAIN_COUNT,
    dry_run: bool = False,
    validate_batches: bool = False,
) -> dict[str, Any]:
    before_free = _free_bytes(base_dir.parent if base_dir.is_absolute() else Path.cwd())
    logs = rotate_logs(log_dir=log_dir, max_bytes=log_max_bytes, retain_count=log_retain_count, dry_run=dry_run)
    proposals = cleanup_validated_proposal_json_exports(
        str(base_dir),
        keep_recent=keep_recent_proposals,
        dry_run=dry_run,
    )
    batches = cleanup_validated_batch_artifacts(
        str(base_dir),
        keep_recent=keep_recent_batches,
        min_summary_bytes=min_summary_bytes,
        dry_run=dry_run,
        mode="thin",
    )
    batch_validation = validate_hot_batch_metadata(str(base_dir)) if validate_batches else None
    after_free = _free_bytes(base_dir.parent if base_dir.is_absolute() else Path.cwd())
    return {
        "dry_run": dry_run,
        "before_free_bytes": before_free,
        "after_free_bytes": after_free,
        "free_delta_bytes": after_free - before_free,
        "logs": logs,
        "proposals": proposals,
        "batches": batches,
        "batch_validation": batch_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run validated experiment-store retention maintenance.")
    parser.add_argument("--base-dir", default="experiments")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--keep-recent-proposals", type=int, default=DEFAULT_KEEP_RECENT_PROPOSALS)
    parser.add_argument("--keep-recent-batches", type=int, default=DEFAULT_KEEP_RECENT_BATCHES)
    parser.add_argument("--min-summary-bytes", type=int, default=DEFAULT_BATCH_SUMMARY_THRESHOLD_BYTES)
    parser.add_argument("--log-max-bytes", type=int, default=DEFAULT_LOG_MAX_BYTES)
    parser.add_argument("--log-retain-count", type=int, default=DEFAULT_LOG_RETAIN_COUNT)
    parser.add_argument("--validate-batches", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_maintenance(
        base_dir=Path(args.base_dir),
        log_dir=Path(args.log_dir),
        keep_recent_proposals=args.keep_recent_proposals,
        keep_recent_batches=args.keep_recent_batches,
        min_summary_bytes=args.min_summary_bytes,
        log_max_bytes=args.log_max_bytes,
        log_retain_count=args.log_retain_count,
        dry_run=args.dry_run,
        validate_batches=args.validate_batches,
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IdeaRecord:
    idea_id: str
    family: str | None
    strategy_type: str
    hypothesis: str
    source: str
    priority: float
    estimated_cost: str
    timestamp_utc: str
    novelty_score: float | None = None
    rationale: str | None = None
    estimated_runtime_cost: str | None = None
    runtime_cost_reason: str | None = None
    suggested_template_id: str | None = None
    suggested_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    idea_source: str | None = None
    source_idea_ids: list[str] = field(default_factory=list)
    paper_title: str | None = None
    web_search_used: bool = False
    idea_provider: str | None = None
    idea_model: str | None = None
    template_id: str | None = None
    idea_kind: str | None = None
    novelty_reason: str | None = None
    is_structurally_novel: bool = False
    is_out_of_box: bool = False
    structural_distance: float | None = None
    template_similarity_class: str | None = None
    uncommon_idea_reason: str | None = None
    is_uncommon_idea: bool = False
    is_new_idea: bool = False
    is_branch_repeat: bool = False
    status: str = "new"


@dataclass(frozen=True)
class ProposalRecord:
    proposal_id: str
    strategy_families: list[str]
    source_idea_ids: list[str]
    candidate_specs: list[dict[str, Any]]
    exploration_fraction: float
    exploitation_fraction: float
    family_budget: dict[str, int]
    timestamp_utc: str
    objective_name: str = "wf_v1_score"
    baseline_name: str | None = None
    planning_rationale: dict[str, Any] | None = None
    analysis_provenance: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    status: str = "pending_execution"


@dataclass(frozen=True)
class AnalysisReport:
    report_id: str
    batch_ids: list[str]
    summary: dict[str, Any]
    next_focus: list[dict[str, Any]]
    timestamp_utc: str
    metrics_summary: dict[str, Any] | None = None
    status: str = "complete"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_helper_dirs(workspace_root: str = ".") -> dict[str, Path]:
    root = Path(workspace_root)
    paths = {
        "agents_dir": root / "agents",
        "ideas_dir": root / "queues" / "ideas",
        "proposals_dir": root / "queues" / "proposals",
        "reports_dir": root / "reports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_json_value(payload)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=True, allow_nan=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            return _normalize_json_value(value.item())
        except (TypeError, ValueError):
            return value
    return value


def save_idea_record(record: IdeaRecord, workspace_root: str = ".") -> Path:
    paths = ensure_helper_dirs(workspace_root)
    path = paths["ideas_dir"] / f"{record.idea_id}.json"
    _atomic_write_json(path, asdict(record))
    return path


def save_proposal_record(record: ProposalRecord, workspace_root: str = ".") -> Path:
    paths = ensure_helper_dirs(workspace_root)
    path = paths["proposals_dir"] / f"{record.proposal_id}.json"
    _atomic_write_json(path, asdict(record))
    return path


def save_analysis_report(report: AnalysisReport, workspace_root: str = ".") -> Path:
    paths = ensure_helper_dirs(workspace_root)
    path = paths["reports_dir"] / f"{report.report_id}.json"
    _atomic_write_json(path, asdict(report))
    return path


def load_json_records(directory: str | Path) -> list[dict[str, Any]]:
    path = Path(directory)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for item in sorted(path.glob("*.json")):
        records.append(json.loads(item.read_text()))
    return records


def load_idea_records(workspace_root: str = ".") -> list[dict[str, Any]]:
    paths = ensure_helper_dirs(workspace_root)
    return load_json_records(paths["ideas_dir"])


def load_proposal_records(workspace_root: str = ".") -> list[dict[str, Any]]:
    paths = ensure_helper_dirs(workspace_root)
    return load_json_records(paths["proposals_dir"])


def load_latest_pending_proposal_record(workspace_root: str = ".") -> dict[str, Any] | None:
    proposals = [
        record
        for record in load_proposal_records(workspace_root)
        if str(record.get("status") or "pending_execution") == "pending_execution"
    ]
    if not proposals:
        return None
    proposals.sort(key=lambda payload: str(payload.get("timestamp_utc") or payload.get("proposal_id") or ""))
    return proposals[-1]


def update_proposal_record_status(
    proposal_id: str,
    *,
    status: str,
    workspace_root: str = ".",
    extra_updates: dict[str, Any] | None = None,
) -> Path | None:
    paths = ensure_helper_dirs(workspace_root)
    path = paths["proposals_dir"] / f"{proposal_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    payload["status"] = status
    if extra_updates:
        payload.update(extra_updates)
    _atomic_write_json(path, payload)
    return path


def load_latest_analysis_report(workspace_root: str = ".") -> dict[str, Any] | None:
    paths = ensure_helper_dirs(workspace_root)
    reports = load_json_records(paths["reports_dir"])
    if not reports:
        return None
    reports.sort(key=lambda payload: str(payload.get("timestamp_utc") or payload.get("report_id") or ""))
    return reports[-1]

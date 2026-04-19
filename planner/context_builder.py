"""Build a compact planning context dict for the Layer 2 planner.

Reads live experiment state from scorecards, best results, template tracking,
and memory, then assembles a JSON-serialisable summary small enough to fit
comfortably in a single planner prompt (target < 6k tokens).

All reads are best-effort: a missing or broken data source logs a warning and
contributes an empty section rather than raising.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE_DIR = "experiments"
_ALLOWED_FAMILIES = ("momentum", "superstock", "ml_ranker", "rl_bandit")
_TOP_N_RESULTS = 10
_RECENT_BATCHES = 5
_MAX_QUEUED_IDEAS = 10


def _safe(label: str, fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        logger.warning("context_builder: %s failed\n%s", label, traceback.format_exc())
        return default


def _scorecard_summary(base_dir: str) -> dict[str, Any]:
    from experiment_scorecards import build_family_scorecards, scorecards_to_records

    scorecards = build_family_scorecards(
        families=list(_ALLOWED_FAMILIES),
        base_dir=base_dir,
        recent_window=30,
    )
    if not scorecards:
        return {}
    records = scorecards_to_records(scorecards)
    # Keep only the fields that matter for planning decisions
    _KEEP = {
        "total_experiments", "viable_rate", "win_rate_vs_baseline",
        "robustness_score", "overfit_risk", "best_objective_score",
        "recent_objective_trend", "recent_viable_trend", "dead_zone_density",
        "duplicate_saturation", "validation_confidence", "validation_coverage",
        "lineage_type", "lineage_depth", "descendant_count",
        "recommended_budget_mix",
    }
    return {
        family: {k: v for k, v in rec.items() if k in _KEEP}
        for family, rec in records.items()
    }


def _best_results_summary(base_dir: str) -> list[dict[str, Any]]:
    from experiment_best_results import top_results_overall

    df = top_results_overall(limit=_TOP_N_RESULTS, base_dir=base_dir)
    if df.empty:
        return []
    _COLS = [
        "experiment_id", "strategy_family", "strategy_type", "config_hash",
        "objective_score", "sharpe", "viable", "confirmation_state",
        "holdout_check_status", "template_id", "lineage_depth",
    ]
    keep = [c for c in _COLS if c in df.columns]
    return df[keep].head(_TOP_N_RESULTS).to_dict(orient="records")


def _template_tracking_summary(base_dir: str) -> dict[str, Any]:
    from experiment_store import load_results_index
    from experiment_template_tracking import build_full_template_tracking_report

    index = load_results_index(base_dir)
    if index.empty:
        return {}
    report = build_full_template_tracking_report(index, family="momentum")
    # Trim to essentials only
    out: dict[str, Any] = {
        "template_handoff_state": report.get("template_handoff_state"),
    }
    for section in ("template_entry", "template_yield"):
        raw = report.get(section, {})
        if isinstance(raw, dict):
            out[section] = {
                tid: {k: v for k, v in rec.items()
                      if k in {"state", "experiment_count", "viable_count",
                               "win_count", "yield_score", "viable_rate",
                               "win_rate", "mean_score", "trend"}}
                for tid, rec in raw.items()
            }
    floor = report.get("structural_floor_active", {})
    out["structural_floor_active"] = floor.get("any_active", False)
    out["floor_templates"] = floor.get("active_templates", [])
    return out


def _recent_batches_summary(base_dir: str) -> list[dict[str, Any]]:
    from experiment_best_results import latest_non_empty_batch
    from experiment_store import load_results_index

    # Grab the last few batch dirs by mtime
    batches_dir = Path(base_dir) / "batches"
    if not batches_dir.exists():
        return []
    dirs = sorted(batches_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    summaries: list[dict[str, Any]] = []
    for batch_dir in dirs[:_RECENT_BATCHES]:
        summary_path = batch_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            summaries.append({
                "batch_id": data.get("batch_id"),
                "family": data.get("family"),
                "n_experiments": data.get("n_experiments"),
                "n_viable": data.get("n_viable"),
                "best_score": data.get("best_objective_score"),
                "timestamp_utc": data.get("timestamp_utc"),
                "cycle_mode": data.get("cycle_mode"),
            })
        except Exception:
            continue
    return summaries


def _queued_ideas_summary(workspace_root: str = ".") -> list[dict[str, Any]]:
    ideas_dir = Path(workspace_root) / "queues" / "ideas"
    if not ideas_dir.exists():
        return []
    _KEEP = {
        "idea_id",
        "family",
        "strategy_type",
        "hypothesis",
        "source",
        "priority",
        "novelty_score",
        "status",
        "timestamp_utc",
        "suggested_template_id",
        "idea_source",
        "source_idea_ids",
        "paper_title",
        "web_search_used",
        "idea_provider",
        "idea_model",
        "is_structurally_novel",
        "is_out_of_box",
        "structural_distance",
        "template_similarity_class",
        "uncommon_idea_reason",
        "is_uncommon_idea",
        "metadata",
    }
    summaries: list[dict[str, Any]] = []
    paths = sorted(ideas_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:_MAX_QUEUED_IDEAS]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Only include new/unconsumed ideas
            if str(data.get("status") or "new") not in {"new", "pending"}:
                continue
            # Keep compact metadata subset
            meta = data.get("metadata") or {}
            compact_meta = {k: meta[k] for k in ("idea_kind", "novelty_reason", "is_new_idea", "is_structurally_novel", "is_out_of_box", "is_uncommon_idea")
                            if k in meta}
            entry = {k: data[k] for k in _KEEP if k in data}
            if compact_meta:
                entry["metadata"] = compact_meta
            summaries.append(entry)
        except Exception:
            continue
    return summaries


def _web_research_status_summary(workspace_root: str = ".") -> dict[str, Any]:
    status_path = Path(workspace_root) / "queues" / "web_research" / "web_research_status.json"
    if not status_path.exists():
        return {
            "status_file_present": False,
            "backoff_state": "unknown",
            "session_limit_hit": False,
            "queued_web_idea_count": 0,
        }
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status_file_present": True,
            "status_file_valid": False,
            "backoff_state": "unknown",
            "session_limit_hit": False,
            "queued_web_idea_count": 0,
        }
    return {
        "status_file_present": True,
        "status_file_valid": True,
        "timestamp_utc": data.get("timestamp_utc"),
        "backoff_state": data.get("backoff_state"),
        "backoff_reason": data.get("backoff_reason"),
        "session_limit_hit": bool(data.get("session_limit_hit", False)),
        "last_attempt_at": data.get("last_attempt_at"),
        "last_success_at": data.get("last_success_at"),
        "last_failure_at": data.get("last_failure_at"),
        "next_retry_at": data.get("next_retry_at"),
        "queued_idea_count": data.get("queued_idea_count", 0),
        "queued_web_idea_count": data.get("queued_web_idea_count", 0),
        "web_search_available": bool(data.get("web_search_available", False)),
        "last_topic_slug": data.get("last_topic_slug"),
        "papers_found": data.get("papers_found", 0),
    }


def _memory_snapshot(base_dir: str) -> dict[str, Any]:
    from experiment_memory import load_research_memory

    mem = load_research_memory(base_dir)
    # Extract the lightweight keys useful for planning; drop large histories
    _KEEP_KEYS = {
        "stagnation_counter", "last_improvement_timestamp",
        "last_improvement_family", "cycle_mode", "exploration_budget",
        "confirmation_pending_count", "holdout_pending_count",
        "global_experiment_count",
    }
    return {k: mem[k] for k in _KEEP_KEYS if k in mem}


def build_planning_context(base_dir: str = _BASE_DIR, workspace_root: str = ".") -> dict[str, Any]:
    """Assemble a compact planning context from live experiment state.

    Returns a JSON-serialisable dict with keys:
      - timestamp_utc
      - scorecards          (per-family scorecard summaries)
      - best_results        (top-N overall results)
      - template_tracking   (momentum template handoff + yield state)
      - recent_batches      (last N batch summaries)
      - memory              (lightweight memory snapshot)
      - queued_ideas        (recent unconsumed idea-agent outputs)
      - web_research_status (web-search backoff / queue visibility)
    """
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "scorecards": _safe("scorecards", _scorecard_summary, base_dir, default={}),
        "best_results": _safe("best_results", _best_results_summary, base_dir, default=[]),
        "template_tracking": _safe("template_tracking", _template_tracking_summary, base_dir, default={}),
        "recent_batches": _safe("recent_batches", _recent_batches_summary, base_dir, default=[]),
        "memory": _safe("memory", _memory_snapshot, base_dir, default={}),
        "queued_ideas": _safe("queued_ideas", _queued_ideas_summary, workspace_root, default=[]),
        "web_research_status": _safe("web_research_status", _web_research_status_summary, workspace_root, default={}),
    }

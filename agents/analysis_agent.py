#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from agents.schemas import AnalysisReport, ensure_helper_dirs, save_analysis_report, utc_now_iso
from baseline_registry import load_baseline_registry
from experiment_best_results import latest_non_empty_batch, top_results_overall, top_results_per_family
from experiment_scorecards import build_family_scorecards, scorecards_to_records
from experiment_store import load_results_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only helper that writes analysis reports")
    parser.add_argument("--workspace-root", default=".", help="Repo root for helper queues/reports")
    parser.add_argument("--experiments-dir", default="experiments", help="Official experiment store to read")
    parser.add_argument("--overall", type=int, default=10, help="Top overall rows to summarize")
    parser.add_argument("--per-family", type=int, default=5, help="Top per-family rows to summarize")
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    selected = frame.head(limit) if limit is not None else frame
    return _json_safe(selected.to_dict("records"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def _best_row(index: pd.DataFrame, mask: pd.Series) -> dict[str, Any] | None:
    if index.empty or mask.empty:
        return None
    frame = index[mask].copy()
    if frame.empty:
        return None
    frame["objective_score"] = pd.to_numeric(frame.get("objective_score"), errors="coerce")
    frame["sharpe"] = pd.to_numeric(frame.get("sharpe"), errors="coerce")
    frame["calmar"] = pd.to_numeric(frame.get("calmar"), errors="coerce")
    frame = frame[frame["objective_score"].notna()].copy()
    if frame.empty:
        return None
    frame = frame.sort_values(
        by=["objective_score", "sharpe", "calmar", "timestamp_utc", "experiment_id"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return _records(frame, limit=1)[0]


def _best_viable_result(index: pd.DataFrame) -> dict[str, Any] | None:
    if index.empty:
        return None
    return _best_row(index, index["viable"].map(_truthy) & index["status"].astype(str).eq("success"))


def _best_baseline_beating_result(index: pd.DataFrame) -> dict[str, Any] | None:
    if index.empty:
        return None
    baseline_mask = index["beats_baseline_objective"].map(_truthy) | index["beats_baseline_guardrails"].map(_truthy)
    return _best_row(index, baseline_mask & index["status"].astype(str).eq("success"))


def _family_trends(index: pd.DataFrame, *, recent_window: int = 50) -> dict[str, dict[str, Any]]:
    if index.empty:
        return {}
    trends: dict[str, dict[str, Any]] = {}
    for family, frame in index.groupby(index["strategy_family"].astype(str), sort=True):
        ordered = frame.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort").tail(max(1, recent_window)).copy()
        objective = pd.to_numeric(ordered.get("objective_score"), errors="coerce").dropna()
        if len(objective) >= 4:
            midpoint = len(objective) // 2
            objective_trend = float(objective.iloc[midpoint:].mean() - objective.iloc[:midpoint].mean())
        else:
            objective_trend = 0.0
        viable_series = ordered.get("viable", pd.Series(dtype=object)).map(lambda value: 1.0 if _truthy(value) else 0.0)
        if len(viable_series) >= 4:
            midpoint = len(viable_series) // 2
            viable_trend = float(viable_series.iloc[midpoint:].mean() - viable_series.iloc[:midpoint].mean())
        else:
            viable_trend = 0.0
        status_counts = Counter(ordered["status"].fillna("").astype(str))
        trends[family] = {
            "recent_count": int(len(ordered)),
            "recent_objective_mean": None if objective.empty else round(float(objective.mean()), 6),
            "recent_objective_trend": round(objective_trend, 6),
            "recent_viable_rate": round(float(viable_series.mean()), 6) if len(viable_series) else 0.0,
            "recent_viable_trend": round(viable_trend, 6),
            "recent_status_counts": dict(status_counts),
        }
    return _json_safe(trends)


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item is not None]
    return [str(parsed)]


def _lineage_summary(index: pd.DataFrame, *, latest_batch: dict[str, Any] | None) -> dict[str, Any]:
    if index.empty:
        return {
            "result_count": 0,
            "result_ids_sample": [],
            "source_proposal_ids": [],
            "source_idea_ids": [],
            "latest_batch_id": latest_batch.get("batch_id") if latest_batch else None,
            "latest_batch_executed_count": latest_batch.get("executed_count") if latest_batch else 0,
        }

    source_proposal_ids = sorted(
        {
            str(value)
            for value in index.get("source_proposal_id", pd.Series(dtype=object)).dropna().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        }
    )
    source_idea_ids: set[str] = set()
    for value in index.get("source_idea_ids", pd.Series(dtype=object)).tolist():
        source_idea_ids.update(_parse_json_list(value))
    result_ids = [
        str(value)
        for value in index.get("experiment_id", pd.Series(dtype=object)).dropna().astype(str).head(50).tolist()
        if value
    ]
    return {
        "result_count": int(len(index)),
        "result_ids_sample": result_ids,
        "source_proposal_ids": source_proposal_ids,
        "source_idea_ids": sorted(source_idea_ids),
        "latest_batch_id": latest_batch.get("batch_id") if latest_batch else None,
        "latest_batch_executed_count": latest_batch.get("executed_count") if latest_batch else 0,
    }


def _score_summary(scorecard_records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for family, scorecard in sorted(scorecard_records.items()):
        summary[family] = {
            "total_experiments": scorecard.get("total_experiments"),
            "viable_rate": scorecard.get("viable_rate"),
            "win_rate_vs_baseline": scorecard.get("win_rate_vs_baseline"),
            "best_objective_score": scorecard.get("best_objective_score"),
            "recent_objective_trend": scorecard.get("recent_objective_trend"),
            "dead_zone_density": scorecard.get("dead_zone_density"),
            "duplicate_saturation": scorecard.get("duplicate_saturation"),
            "stagnation_experiments": scorecard.get("stagnation_experiments"),
            "recovery_signal": scorecard.get("recovery_signal"),
            "search_priority": scorecard.get("search_priority"),
            "confidence": scorecard.get("confidence"),
            "exploration_budget_recommendation": scorecard.get("exploration_budget_recommendation"),
            "exploitation_budget_recommendation": scorecard.get("exploitation_budget_recommendation"),
        }
    return summary


def _focus_from_scorecard(scorecard: dict[str, Any]) -> str:
    total = int(scorecard.get("total_experiments") or 0)
    viable_rate = float(scorecard.get("viable_rate") or 0.0)
    dead_zone_density = float(scorecard.get("dead_zone_density") or 0.0)
    duplicate_saturation = float(scorecard.get("duplicate_saturation") or 0.0)
    stagnation = int(scorecard.get("stagnation_experiments") or 0)
    recovery_signal = bool(scorecard.get("recovery_signal"))
    if recovery_signal:
        return "recover"
    if total < 10:
        return "explore"
    if viable_rate >= 0.10 and dead_zone_density < 0.50 and duplicate_saturation < 0.60:
        return "refine"
    if stagnation >= 50 or dead_zone_density >= 0.70 or duplicate_saturation >= 0.75:
        return "deprioritize"
    return "explore"


def _next_focus_from_scorecards(
    *,
    scorecard_records: dict[str, dict[str, Any]],
    top_family: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    next_focus: list[dict[str, Any]] = []
    for family, scorecard in sorted(scorecard_records.items()):
        focus = _focus_from_scorecard(scorecard)
        top_rows = _records(top_family.get(family, pd.DataFrame()), limit=1)
        best = top_rows[0] if top_rows else {}
        reason_parts: list[str] = []
        if focus == "refine":
            reason_parts.append("viable evidence supports focused refinement")
        elif focus == "recover":
            reason_parts.append("recent trend improved after weak prior evidence")
        elif focus == "deprioritize":
            reason_parts.append("dead-zone, duplicate, or stagnation pressure is high")
        else:
            reason_parts.append("collect more evidence before narrowing")
        if scorecard.get("win_rate_vs_baseline") is not None:
            reason_parts.append(f"baseline win rate={scorecard.get('win_rate_vs_baseline')}")
        next_focus.append(
            {
                "family": family,
                "focus": focus,
                "best_experiment_id": best.get("experiment_id"),
                "best_objective_score": best.get("objective_score") or scorecard.get("best_objective_score"),
                "recommended_exploration_fraction": scorecard.get("exploration_budget_recommendation"),
                "recommended_exploitation_fraction": scorecard.get("exploitation_budget_recommendation"),
                "search_priority": scorecard.get("search_priority"),
                "confidence": scorecard.get("confidence"),
                "evidence_weight": scorecard.get("evidence_weight"),
                "stagnation_experiments": scorecard.get("stagnation_experiments"),
                "duplicate_saturation": scorecard.get("duplicate_saturation"),
                "dead_zone_density": scorecard.get("dead_zone_density"),
                "recovery_signal": scorecard.get("recovery_signal"),
                "reason": "; ".join(reason_parts),
            }
        )
    return _json_safe(next_focus)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def analysis_score_summary_payload(report: AnalysisReport) -> dict[str, Any]:
    summary = report.summary or {}
    metrics_summary = report.metrics_summary or {}
    return {
        "report_id": report.report_id,
        "timestamp_utc": report.timestamp_utc,
        "batch_ids": report.batch_ids,
        "lineage": summary.get("lineage", {}),
        "score_summary": summary.get("score_summary", {}),
        "family_trends": summary.get("family_trends", {}),
        "next_focus": report.next_focus,
        "best_viable_result": summary.get("best_viable_result"),
        "best_baseline_beating_result": summary.get("best_baseline_beating_result"),
        "duplicate_saturation": metrics_summary.get("duplicate_saturation", {}),
        "dead_zone_density": metrics_summary.get("dead_zone_density", {}),
        "stagnation_signals": metrics_summary.get("stagnation_signals", {}),
        "status": report.status,
    }


def save_analysis_score_summary(report: AnalysisReport, *, workspace_root: str = ".") -> dict[str, str]:
    paths = ensure_helper_dirs(workspace_root)
    score_dir = paths["reports_dir"] / "score_summaries"
    payload = analysis_score_summary_payload(report)
    timestamped_path = score_dir / f"{report.report_id}_score_summary.json"
    latest_path = score_dir / "latest.json"
    _atomic_write_json(timestamped_path, payload)
    _atomic_write_json(latest_path, payload)
    return {"score_summary_path": str(timestamped_path), "latest_score_summary_path": str(latest_path)}


def build_analysis_report(
    *,
    workspace_root: str = ".",
    experiments_dir: str = "experiments",
    overall_limit: int = 10,
    per_family_limit: int = 5,
) -> AnalysisReport:
    ensure_helper_dirs(workspace_root)
    index = load_results_index(experiments_dir)
    top_overall = top_results_overall(limit=overall_limit, base_dir=experiments_dir)
    top_family = top_results_per_family(limit=per_family_limit, base_dir=experiments_dir)
    latest_batch = latest_non_empty_batch(base_dir=experiments_dir)
    baselines = load_baseline_registry()
    families = sorted(
        family
        for family in set(index["strategy_family"].dropna().astype(str).tolist()) | set(top_family.keys())
        if family and family.lower() != "nan"
    )
    scorecards = build_family_scorecards(families=families, base_dir=experiments_dir) if families else {}
    scorecard_records = _json_safe(scorecards_to_records(scorecards))

    status_counts = Counter(index["status"].astype(str)) if not index.empty else Counter()
    family_counts = Counter(index["strategy_family"].astype(str)) if not index.empty else Counter()
    next_focus = _next_focus_from_scorecards(scorecard_records=scorecard_records, top_family=top_family)
    family_trends = _family_trends(index)
    score_summary = _score_summary(scorecard_records)
    lineage = _lineage_summary(index, latest_batch=latest_batch)

    timestamp = utc_now_iso()
    batch_ids = [latest_batch["batch_id"]] if latest_batch else []
    summary = {
        "official_results_count": int(len(index)),
        "status_counts": dict(status_counts),
        "family_counts": dict(family_counts),
        "baseline_names": sorted(record.get("name") for record in baselines if record.get("name")),
        "latest_non_empty_batch": latest_batch,
        "top_overall": _records(top_overall, limit=overall_limit),
        "top_per_family": {family: _records(frame, limit=per_family_limit) for family, frame in sorted(top_family.items())},
        "best_viable_result": _best_viable_result(index),
        "best_baseline_beating_result": _best_baseline_beating_result(index),
        "family_trends": family_trends,
        "lineage": lineage,
        "family_scorecards": scorecard_records,
        "score_summary": score_summary,
    }

    return AnalysisReport(
        report_id=f"analysis_{timestamp.replace(':', '').replace('-', '')}",
        batch_ids=batch_ids,
        summary=summary,
        next_focus=next_focus,
        timestamp_utc=timestamp,
        metrics_summary={
            "per_family_top_counts": {family: len(frame) for family, frame in top_family.items()},
            "top_overall_count": int(len(top_overall)),
            "duplicate_saturation": {family: item.get("duplicate_saturation") for family, item in score_summary.items()},
            "dead_zone_density": {family: item.get("dead_zone_density") for family, item in score_summary.items()},
            "stagnation_signals": {
                family: {
                    "stagnation_experiments": item.get("stagnation_experiments"),
                    "recovery_signal": item.get("recovery_signal"),
                }
                for family, item in score_summary.items()
            },
            "score_summary": score_summary,
            "lineage": lineage,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_analysis_report(
        workspace_root=args.workspace_root,
        experiments_dir=args.experiments_dir,
        overall_limit=args.overall,
        per_family_limit=args.per_family,
    )
    path = save_analysis_report(report, workspace_root=args.workspace_root)
    save_analysis_score_summary(report, workspace_root=args.workspace_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

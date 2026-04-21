#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.schemas import (
    ProposalRecord,
    ensure_helper_dirs,
    load_idea_records,
    load_latest_analysis_report,
    save_proposal_record,
)
from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_spaces import list_searchable_families


BASE_FAMILY_WEIGHTS = {
    "momentum": 0.65,
    "superstock": 0.20,
    "fear_greed_contrarian_overlay": 0.04,
    "momentum_fear_greed_overlay": 0.04,
    "sector_breadth_overlay": 0.06,
    "volatility_compression_expansion": 0.06,
    "ml_ranker": 0.10,
    "rl_bandit": 0.05,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only helper that writes proposal records")
    parser.add_argument("--workspace-root", default=".", help="Repo root for helper queues/reports")
    parser.add_argument("--experiments-dir", default="experiments", help="Official experiment store to read")
    parser.add_argument("--family", default="all", help="Specific family or all")
    parser.add_argument("--n", type=int, default=24, help="Maximum candidate experiments to plan")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exploration-fraction", type=float, default=0.65, help="Proposal exploration fraction")
    parser.add_argument("--exploitation-fraction", type=float, default=0.35, help="Proposal exploitation fraction")
    parser.add_argument("--min-viable-fill-rate", type=float, default=0.50, help="Minimum proposal fill rate before execution")
    parser.add_argument("--min-large-search-candidates", type=int, default=48, help="Minimum candidates for large-search proposals")
    parser.add_argument("--no-helper-ideas", action="store_true", help="Ignore queued helper ideas during planning")
    parser.add_argument("--no-analysis-guidance", action="store_true", help="Ignore analysis guidance during planning")
    return parser.parse_args(argv)


def _families(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list_searchable_families()
    return [normalized]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_score_summary(workspace_root: str) -> dict[str, Any]:
    return _load_json_file(Path(workspace_root) / "reports" / "score_summaries" / "latest.json")


def _persisted_family_scorecards(experiments_dir: str) -> dict[str, Any]:
    payload = _load_json_file(Path(experiments_dir) / "scorecards" / "family_scorecards.json")
    families = payload.get("families")
    return families if isinstance(families, dict) else {}


def _planning_context(*, workspace_root: str, experiments_dir: str, use_analysis_guidance: bool) -> dict[str, Any]:
    if not use_analysis_guidance:
        return {"analysis_guidance_enabled": False}
    latest_analysis = load_latest_analysis_report(workspace_root) or {}
    latest_score = _latest_score_summary(workspace_root)
    persisted_scorecards = _persisted_family_scorecards(experiments_dir)
    analysis_summary = latest_analysis.get("summary") or {}
    score_summary = latest_score.get("score_summary") or analysis_summary.get("score_summary") or {}
    score_summary = {**persisted_scorecards, **score_summary}
    next_focus = latest_score.get("next_focus") or latest_analysis.get("next_focus") or []
    return {
        "analysis_guidance_enabled": True,
        "latest_analysis": latest_analysis,
        "latest_score_summary": latest_score,
        "analysis_report_id": latest_analysis.get("report_id"),
        "score_summary_report_id": latest_score.get("report_id"),
        "analysis_batch_ids": latest_analysis.get("batch_ids") or latest_score.get("batch_ids") or [],
        "next_focus": next_focus,
        "score_summary": score_summary,
        "family_trends": latest_score.get("family_trends") or analysis_summary.get("family_trends") or {},
        "best_viable_result": latest_score.get("best_viable_result") or analysis_summary.get("best_viable_result"),
        "best_baseline_beating_result": latest_score.get("best_baseline_beating_result")
        or analysis_summary.get("best_baseline_beating_result"),
        "lineage": latest_score.get("lineage") or analysis_summary.get("lineage") or {},
        "persisted_scorecards_loaded": bool(persisted_scorecards),
    }


def _focus_by_family(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("family")): item
        for item in context.get("next_focus", [])
        if isinstance(item, dict) and item.get("family")
    }


def _cycle_mode(*, selected_families: list[str], max_experiments: int, context: dict[str, Any]) -> str:
    score_summary = context.get("score_summary") or {}
    focus_by_family = _focus_by_family(context)
    stagnation = max((int((score_summary.get(family) or {}).get("stagnation_experiments") or 0) for family in selected_families), default=0)
    if stagnation >= 50 or any((focus_by_family.get(family) or {}).get("focus") == "recover" for family in selected_families):
        return "stagnation_escape"
    if max_experiments >= 50:
        return "large-search"
    if any((focus_by_family.get(family) or {}).get("focus") == "refine" for family in selected_families):
        return "local_refinement"
    if any((focus_by_family.get(family) or {}).get("focus") == "deprioritize" for family in selected_families):
        return "diagnostics"
    return "normal_exploration"


def _adjust_exploration_split(
    *,
    exploration_fraction: float,
    exploitation_fraction: float,
    cycle_mode: str,
) -> tuple[float, float]:
    if cycle_mode == "stagnation_escape":
        exploration = max(exploration_fraction, 0.80)
    elif cycle_mode == "large-search":
        exploration = max(exploration_fraction, 0.70)
    elif cycle_mode == "local_refinement":
        exploration = min(max(exploration_fraction, 0.45), 0.60)
    elif cycle_mode == "diagnostics":
        exploration = max(exploration_fraction, 0.70)
    else:
        exploration = exploration_fraction
    exploration = round(max(0.0, min(1.0, exploration)), 6)
    return exploration, round(1.0 - exploration, 6)


def _normalize_budget(raw: dict[str, int], *, total: int, weights: dict[str, float]) -> dict[str, int]:
    budgets = {family: max(0, int(value)) for family, value in raw.items()}
    if not budgets:
        return {}
    while sum(budgets.values()) < total:
        family = max(budgets, key=lambda item: (weights.get(item, 0.0), -budgets[item], item))
        budgets[family] += 1
    while sum(budgets.values()) > total:
        removable = [family for family, value in budgets.items() if value > 0]
        if not removable:
            break
        family = min(removable, key=lambda item: (weights.get(item, 0.0), -budgets[item], item))
        budgets[family] -= 1
    return {family: value for family, value in budgets.items() if value > 0}


def _family_budget_plan(
    *,
    selected_families: list[str],
    max_experiments: int,
    context: dict[str, Any],
    cycle_mode: str,
) -> tuple[dict[str, int] | None, dict[str, Any]]:
    if not context.get("analysis_guidance_enabled") or not (context.get("score_summary") or context.get("next_focus")):
        return None, {
            "mode": "fallback_internal",
            "reason": "no analysis or scorecard guidance available; use existing proposal generator negotiation",
        }
    if len(selected_families) == 1:
        family = selected_families[0]
        scorecard = (context.get("score_summary") or {}).get(family) or {}
        focus = _focus_by_family(context).get(family) or {}
        return {family: max_experiments}, {
            "mode": "single_family",
            "families": {
                family: {
                    "budget": max_experiments,
                    "focus": focus.get("focus"),
                    "reason": "single selected family receives the full planning budget",
                    "scorecard": scorecard,
                }
            },
        }

    score_summary = context.get("score_summary") or {}
    focus_by_family = _focus_by_family(context)
    weights: dict[str, float] = {}
    families_report: dict[str, Any] = {}
    for family in selected_families:
        scorecard = score_summary.get(family) or {}
        focus = focus_by_family.get(family) or {}
        focus_value = str(focus.get("focus") or "unknown")
        viable_rate = _safe_float(scorecard.get("viable_rate"), 0.0)
        search_priority = _safe_float(scorecard.get("search_priority"), 0.0)
        confidence = _safe_float(scorecard.get("confidence"), 0.0)
        dead_zone_density = _safe_float(scorecard.get("dead_zone_density"), 0.0)
        duplicate_saturation = _safe_float(scorecard.get("duplicate_saturation"), 0.0)
        stagnation = int(scorecard.get("stagnation_experiments") or 0)
        weight = BASE_FAMILY_WEIGHTS.get(family, 0.05)
        weight += viable_rate * 0.20
        weight += search_priority * max(0.15, confidence) * 0.35
        if focus_value == "refine":
            weight *= 1.25
        elif focus_value in {"explore", "recover"}:
            weight *= 1.10
        elif focus_value == "deprioritize":
            weight *= 0.55
        if family == "momentum" and viable_rate > 0.05:
            weight *= 1.15
        if family in {"rl_bandit", "ml_ranker"} and viable_rate <= 0.0 and dead_zone_density >= 0.50:
            weight *= 0.55
        if duplicate_saturation >= 0.70:
            weight *= 0.75
        if cycle_mode == "stagnation_escape" and stagnation >= 50:
            weight *= 1.20
        weights[family] = max(weight, 0.01)
        if family == "fear_greed_contrarian_overlay":
            weights[family] = min(weights[family], 0.08)
        if family == "momentum_fear_greed_overlay":
            weights[family] = min(weights[family], 0.08)
        families_report[family] = {
            "base_weight": BASE_FAMILY_WEIGHTS.get(family, 0.05),
            "computed_weight": round(weights[family], 6),
            "focus": focus_value,
            "viable_rate": viable_rate,
            "search_priority": search_priority,
            "confidence": confidence,
            "dead_zone_density": dead_zone_density,
            "duplicate_saturation": duplicate_saturation,
            "stagnation_experiments": stagnation,
        }

    total_weight = sum(weights.values()) or float(len(weights))
    raw = {family: int(round(max_experiments * weights[family] / total_weight)) for family in selected_families}
    min_alive = 1 if max_experiments >= len(selected_families) else 0
    for family in selected_families:
        if raw.get(family, 0) < min_alive:
            raw[family] = min_alive
    budgets = _normalize_budget(raw, total=max_experiments, weights=weights)
    for family, budget in budgets.items():
        families_report.setdefault(family, {})["budget"] = budget
        if family in {"rl_bandit", "ml_ranker"} and budget <= max(1, int(max_experiments * 0.15)):
            families_report[family]["reason"] = "retained with a controlled exploratory budget due to weaker evidence"
        elif family == "fear_greed_contrarian_overlay":
            families_report[family]["reason"] = (
                "limited exploratory budget: diagnostics promoted Fear & Greed to a market exposure overlay"
            )
        elif family == "momentum_fear_greed_overlay":
            families_report[family]["reason"] = (
                "limited exploratory budget: tests Fear & Greed as a momentum sleeve exposure scaler"
            )
        elif family == "momentum":
            families_report[family]["reason"] = "primary budget target when scorecard/viability evidence supports it"
        else:
            families_report[family]["reason"] = "allocated by scorecard weight and next-focus guidance"
    for family in selected_families:
        if family in budgets:
            continue
        report = families_report.setdefault(family, {})
        report["budget"] = 0
        if report.get("focus") == "deprioritize" and _safe_float(report.get("dead_zone_density"), 0.0) >= 0.50:
            report["reason"] = "not retained this round because next-focus and scorecard evidence show high dead-zone risk"
        else:
            report["reason"] = "not retained after budget normalization; revisit when evidence improves or requested budget increases"

    return budgets, {
        "mode": "analysis_scorecard_weighted",
        "cycle_mode": cycle_mode,
        "raw_budgets": raw,
        "normalized_budgets": budgets,
        "families": families_report,
        "reason": "budget weighted by latest analysis focus, family scorecards, stagnation, duplicate, and dead-zone signals",
    }


def _planning_adjustments(
    *,
    selected_families: list[str],
    max_experiments: int,
    exploration_fraction: float,
    exploitation_fraction: float,
    min_large_search_candidates: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    cycle_mode = _cycle_mode(selected_families=selected_families, max_experiments=max_experiments, context=context)
    adjusted_exploration, adjusted_exploitation = _adjust_exploration_split(
        exploration_fraction=exploration_fraction,
        exploitation_fraction=exploitation_fraction,
        cycle_mode=cycle_mode,
    )
    budgets, budget_rationale = _family_budget_plan(
        selected_families=selected_families,
        max_experiments=max_experiments,
        context=context,
        cycle_mode=cycle_mode,
    )
    if cycle_mode in {"large-search", "stagnation_escape"}:
        adjusted_min_large = max(min_large_search_candidates, min(max_experiments, max(48, int(round(max_experiments * 0.50)))))
    else:
        adjusted_min_large = min_large_search_candidates
    return {
        "cycle_mode": cycle_mode,
        "exploration_fraction": adjusted_exploration,
        "exploitation_fraction": adjusted_exploitation,
        "per_family_budgets": budgets,
        "min_large_search_candidates": adjusted_min_large,
        "family_budget_rationale": budget_rationale,
    }


def build_planning_proposal(
    *,
    workspace_root: str = ".",
    experiments_dir: str = "experiments",
    families: list[str] | None = None,
    max_experiments: int = 24,
    seed: int = 42,
    exploration_fraction: float = 0.65,
    exploitation_fraction: float = 0.35,
    min_viable_fill_rate: float = 0.50,
    min_large_search_candidates: int = 48,
    use_idea_queue: bool = True,
    use_analysis_guidance: bool = True,
) -> ProposalRecord:
    ensure_helper_dirs(workspace_root)
    selected_families = families or list_searchable_families()
    ideas = load_idea_records(workspace_root) if use_idea_queue else []
    planning_context = _planning_context(
        workspace_root=workspace_root,
        experiments_dir=experiments_dir,
        use_analysis_guidance=use_analysis_guidance,
    )
    latest_analysis = planning_context.get("latest_analysis") if use_analysis_guidance else None
    adjustments = _planning_adjustments(
        selected_families=selected_families,
        max_experiments=max_experiments,
        exploration_fraction=exploration_fraction,
        exploitation_fraction=exploitation_fraction,
        min_large_search_candidates=min_large_search_candidates,
        context=planning_context,
    )

    request = build_proposal_request(
        strategy_families=selected_families,
        max_experiments=max_experiments,
        seed=seed,
        exploration_fraction=adjustments["exploration_fraction"],
        exploitation_fraction=adjustments["exploitation_fraction"],
        per_family_budgets=adjustments["per_family_budgets"],
        use_idea_queue=use_idea_queue,
        use_analysis_guidance=use_analysis_guidance,
        min_viable_fill_rate=min_viable_fill_rate,
        min_large_search_candidates=adjustments["min_large_search_candidates"],
        persist_scorecards=False,
        persist_memory=False,
        persist_proposal=False,
    )
    proposal = generate_next_round_proposal(request, base_dir=experiments_dir)

    candidate_specs: list[dict[str, object]] = []
    family_budget: dict[str, int] = {}
    for family, configs in proposal.candidate_configs.items():
        metadata_items = (proposal.candidate_metadata or {}).get(family, [])
        family_budget[family] = len(configs)
        for idx, config in enumerate(configs):
            metadata = metadata_items[idx] if idx < len(metadata_items) else {}
            candidate_specs.append(
                {
                    "family": family,
                    "params": config,
                    "metadata": metadata,
                }
            )

    idea_ids = list(proposal.request.source_idea_ids or [])
    planning_rationale = {
        "reasoning_summary": proposal.reasoning_summary,
        "helper_idea_count": len(ideas),
        "analysis_guidance_used": bool(latest_analysis or planning_context.get("latest_score_summary") or planning_context.get("score_summary")),
        "cycle_mode": adjustments["cycle_mode"],
        "family_budget_rationale": adjustments["family_budget_rationale"],
        "analysis_report_ids_used": {
            "analysis_report_id": planning_context.get("analysis_report_id"),
            "score_summary_report_id": planning_context.get("score_summary_report_id"),
        },
        "scorecard_snapshot": planning_context.get("score_summary") or {},
        "best_viable_result": planning_context.get("best_viable_result"),
        "best_baseline_beating_result": planning_context.get("best_baseline_beating_result"),
        "stagnation_signals": {
            family: (planning_context.get("score_summary") or {}).get(family, {}).get("stagnation_experiments")
            for family in selected_families
        },
        "adjusted_request": {
            "exploration_fraction": proposal.request.exploration_fraction,
            "exploitation_fraction": proposal.request.exploitation_fraction,
            "min_large_search_candidates": proposal.request.min_large_search_candidates,
            "per_family_budgets": proposal.request.per_family_budgets,
        },
    }
    analysis_provenance = None
    if latest_analysis:
        analysis_provenance = {
            "report_id": latest_analysis.get("report_id"),
            "batch_ids": latest_analysis.get("batch_ids", []),
            "next_focus": latest_analysis.get("next_focus", []),
            "score_summary_report_id": planning_context.get("score_summary_report_id"),
        }

    return ProposalRecord(
        proposal_id=proposal.request.proposal_id,
        strategy_families=list(proposal.request.strategy_families),
        source_idea_ids=idea_ids,
        candidate_specs=candidate_specs,
        exploration_fraction=proposal.request.exploration_fraction,
        exploitation_fraction=proposal.request.exploitation_fraction,
        family_budget=family_budget,
        timestamp_utc=proposal.request.timestamp_utc,
        objective_name=proposal.request.objective_name,
        baseline_name=proposal.request.baseline_name,
        planning_rationale=planning_rationale,
        analysis_provenance=analysis_provenance,
        quality_report=proposal.reasoning_summary.get("proposal_quality"),
        metadata={
            "generated_by": "planning_agent",
            "helper_idea_ids_seen": [record.get("idea_id") for record in ideas if record.get("idea_id")],
            "cycle_mode": adjustments["cycle_mode"],
            "family_budget_rationale": adjustments["family_budget_rationale"],
            "proposal_status": proposal.status,
            "proposal_path": proposal.proposal_path,
            "summary_path": proposal.summary_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_planning_proposal(
        workspace_root=args.workspace_root,
        experiments_dir=args.experiments_dir,
        families=_families(args.family),
        max_experiments=args.n,
        seed=args.seed,
        exploration_fraction=args.exploration_fraction,
        exploitation_fraction=args.exploitation_fraction,
        min_viable_fill_rate=args.min_viable_fill_rate,
        min_large_search_candidates=args.min_large_search_candidates,
        use_idea_queue=not args.no_helper_ideas,
        use_analysis_guidance=not args.no_analysis_guidance,
    )
    path = save_proposal_record(record, workspace_root=args.workspace_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

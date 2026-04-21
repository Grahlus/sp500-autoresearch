from __future__ import annotations

import json
from typing import Any


MAX_PREVIEW_ITEMS = 5
MAX_PREVIEW_STRING = 500


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _scalar_or_short(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_PREVIEW_STRING:
        return value[:MAX_PREVIEW_STRING] + "...[truncated]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _compact_mapping_preview(value: dict[str, Any], *, max_items: int = MAX_PREVIEW_ITEMS) -> dict[str, Any]:
    items = list(value.items())
    return {
        "kind": "dict",
        "count": len(items),
        "preview": {str(k): compact_json_value(v) for k, v in items[:max_items]},
        "truncated": len(items) > max_items,
    }


def _compact_sequence_preview(value: list[Any], *, max_items: int = MAX_PREVIEW_ITEMS) -> dict[str, Any]:
    return {
        "kind": "list",
        "count": len(value),
        "first_items": [compact_json_value(item) for item in value[:max_items]],
        "truncated": len(value) > max_items,
    }


def _branch_ref(branch: Any) -> Any:
    if not isinstance(branch, dict):
        return compact_json_value(branch)
    keep = (
        "family",
        "branch_root_config_hash",
        "branch_root_experiment_id",
        "best_branch_config_hash",
        "best_branch_experiment_id",
        "best_branch_objective_score",
        "branch_budget",
        "branch_budget_share",
        "branch_budget_stance",
        "branch_budget_reason",
        "branch_state",
        "branch_score",
        "branch_effective_score",
        "branch_trust_score",
        "descendant_count",
        "branch_node_count",
        "updated_at",
    )
    return {key: branch.get(key) for key in keep if key in branch}


def compact_branch_budgets(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    families: dict[str, Any] = {}
    total = 0
    for family, branches in value.items():
        if not isinstance(branches, list):
            families[str(family)] = compact_json_value(branches)
            continue
        total += len(branches)
        families[str(family)] = {
            "count": len(branches),
            "first_items": [_branch_ref(branch) for branch in branches[:MAX_PREVIEW_ITEMS]],
            "truncated": len(branches) > MAX_PREVIEW_ITEMS,
        }
    return {
        "family_count": len(families),
        "total_branch_count": total,
        "families": families,
    }


def compact_branch_budget_rationale(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key == "families" and isinstance(item, dict):
            compact["families"] = {
                str(family): compact_branch_budget_rationale(family_item)
                for family, family_item in item.items()
            }
        elif key == "branches" and isinstance(item, list):
            compact["branch_count"] = len(item)
            compact["first_branches"] = [_branch_ref(branch) for branch in item[:MAX_PREVIEW_ITEMS]]
            compact["branches_truncated"] = len(item) > MAX_PREVIEW_ITEMS
        else:
            compact[key] = compact_json_value(item)
    return compact


def compact_lineage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key == "by_config_hash" and isinstance(item, dict):
            compact[key] = {
                "count": len(item),
                "first_config_hashes": list(item.keys())[:MAX_PREVIEW_ITEMS],
                "truncated": len(item) > MAX_PREVIEW_ITEMS,
            }
        elif key in {"records", "branch_summaries"} and isinstance(item, (dict, list)):
            compact[key] = compact_json_value(item)
        else:
            compact[key] = compact_json_value(item)
    return compact


def compact_runtime_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key == "branch_budgets":
            compact[key] = compact_branch_budgets(item)
        elif key == "branch_budget_rationale":
            compact[key] = compact_branch_budget_rationale(item)
        elif key == "lineage_summary":
            compact[key] = compact_lineage_summary(item)
        elif key in {"rationale", "used_signals"} and isinstance(item, dict):
            compact[key] = compact_reasoning_summary(item)
        else:
            compact[key] = compact_json_value(item)
    return compact


def compact_json_value(value: Any) -> Any:
    scalar = _scalar_or_short(value)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, list):
        return _compact_sequence_preview(value)
    if isinstance(value, dict):
        if _json_bytes(value) <= 16_000:
            return {
                str(key): compact_json_value(item)
                for key, item in value.items()
            }
        return _compact_mapping_preview(value)
    return str(value)


def compact_reasoning_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "runtime_decision":
            compact[key] = compact_runtime_decision(value)
        elif key == "branch_budgets":
            compact[key] = compact_branch_budgets(value)
        elif key == "branch_budget_rationale":
            compact[key] = compact_branch_budget_rationale(value)
        elif key == "family_budget_rationale" and isinstance(value, dict):
            compact[key] = compact_reasoning_summary(value)
        elif key == "lineage_summary":
            compact[key] = compact_lineage_summary(value)
        elif key == "families" and isinstance(value, dict):
            compact[key] = {
                str(family): compact_reasoning_summary(family_summary)
                if isinstance(family_summary, dict)
                else compact_json_value(family_summary)
                for family, family_summary in value.items()
            }
        else:
            compact[key] = compact_json_value(value)
    return compact


def compact_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        return {}
    compact = dict(request)
    if "branch_budgets" in compact:
        compact["branch_budgets"] = compact_branch_budgets(compact.get("branch_budgets"))
    if "branch_budget_rationale" in compact:
        compact["branch_budget_rationale"] = compact_branch_budget_rationale(compact.get("branch_budget_rationale"))
    return {str(key): compact_json_value(value) for key, value in compact.items()}


def compact_candidate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    compact: dict[str, Any] = {}
    keep = (
        "config_hash",
        "source_type",
        "template_id",
        "strategy_type",
        "proposal_role",
        "exploration_mode",
        "parent_config_hash",
        "near_duplicate_of",
        "novelty_score",
        "selection_score",
        "duplicate_risk",
        "dead_zone_risk",
        "source_idea_ids",
        "source_region_id",
        "source_grid_search_id",
        "region_class",
        "promotion_recommendation",
        "region_state",
        "allowed_override_params",
        "allowed_override_values",
        "off_grid_params",
        "validation_override_reason",
        "is_new_idea",
        "is_structurally_novel",
        "is_uncommon_idea",
    )
    for family, items in metadata.items():
        if not isinstance(items, list):
            compact[str(family)] = compact_json_value(items)
            continue
        compact[str(family)] = [
            {key: compact_json_value(item.get(key)) for key in keep if isinstance(item, dict) and key in item}
            for item in items
        ]
    return compact


def candidate_metadata_summary(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        str(family): {
            "count": len(items),
            "first_items": compact_candidate_metadata({family: items}).get(str(family), [])[:MAX_PREVIEW_ITEMS],
            "truncated": len(items) > MAX_PREVIEW_ITEMS,
        }
        for family, items in metadata.items()
        if isinstance(items, list)
    }


def candidate_family_counts(candidate_configs: Any) -> dict[str, int]:
    if not isinstance(candidate_configs, dict):
        return {}
    return {
        str(family): len(configs)
        for family, configs in candidate_configs.items()
        if isinstance(configs, list)
    }


def compact_proposal_for_disk(proposal: dict[str, Any]) -> dict[str, Any]:
    request = proposal.get("request") if isinstance(proposal.get("request"), dict) else {}
    candidate_configs = proposal.get("candidate_configs") if isinstance(proposal.get("candidate_configs"), dict) else {}
    compact = {
        "artifact_format": "compact_proposal_v1",
        "proposal_id": request.get("proposal_id") or proposal.get("proposal_id"),
        "status": proposal.get("status"),
        "request": compact_request(request),
        "candidate_count": sum(candidate_family_counts(candidate_configs).values()),
        "candidate_family_counts": candidate_family_counts(candidate_configs),
        "candidate_configs": candidate_configs,
        "candidate_metadata": compact_candidate_metadata(proposal.get("candidate_metadata")),
        "candidate_metadata_summary": candidate_metadata_summary(proposal.get("candidate_metadata")),
        "reasoning_summary": compact_reasoning_summary(proposal.get("reasoning_summary")),
        "proposal_path": proposal.get("proposal_path"),
        "summary_path": proposal.get("summary_path"),
    }
    compact["artifact_compaction"] = {
        "source_format": "full_proposal_result",
        "removed_full_fields": [
            "request.branch_budgets",
            "request.branch_budget_rationale",
            "candidate_metadata full rows",
            "reasoning_summary branch/lineage/runtime bulk payloads",
        ],
    }
    return compact


def compact_summary_for_disk(reasoning_summary: Any, proposal: dict[str, Any] | None = None) -> dict[str, Any]:
    compact = compact_reasoning_summary(reasoning_summary)
    if proposal:
        request = proposal.get("request") if isinstance(proposal.get("request"), dict) else {}
        candidate_configs = proposal.get("candidate_configs")
        compact.setdefault("proposal_id", request.get("proposal_id") or proposal.get("proposal_id"))
        compact.setdefault("candidate_count", sum(candidate_family_counts(candidate_configs).values()))
        compact.setdefault("candidate_family_counts", candidate_family_counts(candidate_configs))
    compact["artifact_format"] = "compact_proposal_summary_v1"
    return compact

#!/usr/bin/env python3
"""Family Discovery Agent.

Discovers structurally novel strategy families using MiniMax LLM.
Two-step pipeline:
  1. Generation call: propose 15 diverse new family ideas
  2. Clustering/ranking call: cluster, deduplicate, rank the top candidates

Outputs FamilyCandidateRecord objects saved to queues/family_candidates/.
Promoted candidates (controlled_probe) are also seeded as IdeaRecord objects
into the standard idea queue so the existing proposal pipeline can consume them.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Memory and web modules (imported lazily to avoid circular imports at module level)
# from agents.family_discovery_memory import ...
# from agents.family_discovery_web import ...

from agents.schemas import (
    FAMILY_CANDIDATE_STATUSES,
    FamilyCandidateRecord,
    IdeaRecord,
    _atomic_write_json,
    ensure_helper_dirs,
    load_active_family_candidates,
    load_family_candidates,
    save_family_candidate,
    save_idea_record,
    update_family_candidate_status,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROVIDER = "minimax"
_MODEL = "MiniMax-M2.7"

_EXISTING_FAMILIES_SUMMARY = """\
EXISTING STRATEGY FAMILIES (already implemented or tried — do NOT propose these):
 1. momentum                    — Jegadeesh-Titman cross-sectional price momentum (12M-1M formation, weekly rebalance)
 2. superstock                  — High-conviction technical breakout screen with trend and quality filters
 3. ml_ranker                   — LightGBM/XGBoost ranking signal on OHLCV price/volume features
 4. rl_bandit                   — Reinforcement learning bandit for position sizing decisions
 5. breadth_timing_overlay      — Market breadth timing overlay (paused)
 6. vix_regime_sector_tilt      — VIX-based sector rotation tilt (paused)
 7. mean_reversion               — Short-term price mean reversion (redesign — cost-dominated)
 8. volatility_compression_expansion — Volatility regime compression/expansion detection (retired)
 9. fear_greed_contrarian        — CNN Fear&Greed contrarian basket timing (retired/redesign)
10. fear_greed_contrarian_overlay — Fear&Greed as market exposure overlay (redesign)
11. sector_breadth_overlay       — Sector breadth overlay on base signal (redesign)
12. amihud_illiquidity_premium   — Amihud illiquidity premium selection (retired — fragility issues)
13. momentum_fear_greed_overlay  — Momentum with Fear&Greed gating (redesign)
14. event_revision_drift         — Post-earnings drift (paused — no earnings data)"""

_GENERATION_SYSTEM = (
    "You are a quantitative equity research strategist specializing in systematic trading. "
    "Return only valid JSON arrays. Be creative but realistic and grounded in market microstructure."
)

_GENERATION_PROMPT = """\
You are discovering NEW strategy families for an SP500/Russell-1000 equity research system.

=== DATA AVAILABLE (hard constraint — you cannot use anything else) ===
- Daily OHLCV (open, high, low, close, volume) for ~841 US large-cap tickers + ETFs
- 11 years of history (2013–2024)
- Daily VIX index (CBOE volatility index, 0–80+ scale)
- CNN Fear & Greed index (daily, 0–100 scale)
- NO options data  |  NO earnings data  |  NO fundamentals  |  NO tick data  |  NO sentiment feeds

=== EXECUTION ENGINE (hard constraints) ===
- Walk-forward backtesting: 14 rolling windows, 3-year train / 6-month test
- Objective: mean Sharpe across 14 windows (target > 0.3, good > 0.8)
- Portfolio: 3–50 positions, equal-weight preferred, $100k start
- Transaction costs: $20 commission + 5 bps slippage per trade
- Trade frequency limit: < 150 round-trip trades/year (costs kill high-freq)
- No leverage, no short-selling (long-only universe)

{existing_families}

=== YOUR TASK ===
Propose exactly 15 NEW and STRUCTURALLY DISTINCT strategy families.

Rules:
1. Each must be mechanistically different from ALL 14 existing families above
2. Each must be fully implementable using ONLY daily OHLCV + VIX + F&G
3. Each must have a clear, falsifiable edge hypothesis
4. Each must be SURPRISING — not just "momentum with a different lookback"
5. Holding horizon must be days to weeks (not HFT, not multi-year)
6. Cost budget: keep rebalance frequency low enough to survive $20/trade friction

Think broadly about:
- How intraday OHLC RANGE patterns (not just close-to-close) encode institutional vs retail behavior
- How VOLUME PROFILE changes (e.g., drying volume vs swelling volume) predict continuation/reversal
- How CROSS-SECTIONAL DISPERSION of returns, vol, or price-to-range changes at regime transitions
- How BREADTH DYNAMICS at multiple horizons (1w vs 4w vs 13w) reveal rotation vs trend
- How RELATIVE PRICE POSITION within multi-year range predicts different alpha regimes
- How VIX TERM STRUCTURE proxies can be inferred from rolling OHLCV patterns
- How CALENDAR and SEQUENTIAL patterns (turn-of-month, pre-holiday drift, earnings-cluster seasonality)
- How SECTOR LEADERSHIP ROTATION measured by price can be exploited independently of raw momentum
- How VOLATILITY-ADJUSTED TREND signals differ from simple price momentum in different regimes
- How REALIZED vs EXPECTED vol spreads (using close-to-close vs HL range) create positioning opportunities

Return a JSON array of exactly 15 objects. Each object must have ALL these fields:
[
  {{
    "family_name": "snake_case_identifier (no spaces)",
    "edge_source": "one sentence: the specific behavioral/structural/microstructure phenomenon",
    "why_it_should_exist": "one paragraph: mechanism and evidence this generates alpha",
    "why_not_momentum": "one sentence: how this differs mechanistically from price momentum ranking",
    "why_not_superstock": "one sentence: how this differs from breakout/quality screen selection",
    "required_data": "which OHLCV+VIX+FG features this needs (be specific: open, high, low, close, volume, vix, fg)",
    "implementation_complexity": "low|medium|high",
    "expected_holding_horizon": "e.g. 5-15 days, 2-6 weeks, 4-8 weeks",
    "first_test_family_template": {{
      "signal_construction": "exact description: how to compute the ranking signal per stock per day",
      "universe_filter": "how to pre-filter the ~841 tickers",
      "position_sizing": "equal-weight|vol-scaled|rank-weighted",
      "rebalance_freq": "e.g. weekly, bi-weekly, monthly",
      "top_n_pct": "e.g. 0.02-0.04 (top 2-4% = 17-34 stocks)",
      "exit_rule": "rank-exit|time-exit|threshold-exit",
      "key_params": {{"param_name": "suggested_range_or_value"}}
    }},
    "novelty_reason": "why this is structurally distinct from ALL 14 existing families",
    "orthogonality_claim": "in which market regimes does this earn when momentum does NOT earn",
    "expected_sharpe_range": "rough walk-forward estimate e.g. 0.3-0.7",
    "failure_modes": "top 2 reasons this might fail in walk-forward backtests"
  }}
]

Return ONLY the JSON array. No markdown fences, no commentary, no other text."""

_CLUSTERING_SYSTEM = (
    "You are a quantitative research manager evaluating strategy proposals. "
    "Return only valid JSON objects. Be analytical and critical."
)

_CLUSTERING_PROMPT = """\
You are evaluating {n_candidates} proposed strategy families to select the best candidates for \
controlled testing in an equity research system.

=== SELECTION CRITERIA (in order of importance) ===
1. Orthogonality to existing families (distinct edge source, different regime behavior)
2. Feasibility with available data (OHLCV daily + VIX + Fear&Greed only)
3. Expected walk-forward robustness (would survive 14 rolling 6-month test windows?)
4. Research value (how much can we learn from running this, even if it fails?)
5. Implementation simplicity (prefer classical signal over ML/RL where competitive)

=== EXISTING FAMILIES TO STAY ORTHOGONAL TO ===
{existing_families}

=== PROPOSED CANDIDATES ===
{candidates_json}

=== YOUR TASK ===
1. Group similar candidates into 4-6 clusters based on their MECHANISM (not their domain)
2. Score every candidate on all 4 criteria (0.0–1.0 each)
3. Compute ranking_score = 0.40*orthogonality + 0.30*feasibility + 0.20*robustness + 0.10*research_value
4. Within each cluster, mark the single best representative (is_best_in_cluster=true)
5. Identify near-duplicates and mark them with duplicate_of pointing to the better version
6. Select the top 6–8 family_names from cluster representatives as final top_candidates

Return a JSON object with EXACTLY this schema:
{{
  "clusters": [
    {{
      "cluster_id": "cluster_1",
      "cluster_label": "descriptive mechanism label",
      "members": ["family_name_a", "family_name_b"],
      "best_representative": "family_name",
      "cluster_rationale": "why these are mechanistically similar"
    }}
  ],
  "scored_candidates": [
    {{
      "family_name": "name",
      "cluster_id": "cluster_1",
      "orthogonality_score": 0.0,
      "feasibility_score": 0.0,
      "expected_robustness": 0.0,
      "research_value": 0.0,
      "ranking_score": 0.0,
      "is_best_in_cluster": true,
      "duplicate_of": null,
      "ranking_rationale": "one sentence explanation"
    }}
  ],
  "top_candidates": ["family_name_1", "family_name_2"],
  "discovery_summary": "2-3 sentence synthesis of what was discovered and why these families are worth testing"
}}

Return ONLY the JSON object. No markdown fences, no other text."""

_GENERATION_PROMPT_CLOSING = (
    "Return ONLY the JSON array. No markdown fences, no commentary, no other text."
)


def _build_generation_prompt(
    *,
    names_to_avoid: list[str] | None = None,
    web_synthesis: str | None = None,
    focus_themes: list[str] | None = None,
    n_ideas: int = 15,
) -> str:
    """Build a dynamic generation prompt with memory context and web synthesis injected."""
    base = _GENERATION_PROMPT.format(existing_families=_EXISTING_FAMILIES_SUMMARY)
    # Remove the static closing instruction; we'll re-append it last
    if base.endswith(_GENERATION_PROMPT_CLOSING):
        base = base[: -len(_GENERATION_PROMPT_CLOSING)].rstrip()

    extra: list[str] = []

    if names_to_avoid:
        avoid_lines = "\n".join(f"  - {n}" for n in names_to_avoid[:20])
        extra.append(
            "\n=== RECENTLY PROPOSED FAMILIES — DO NOT RE-PROPOSE ===\n"
            "These were generated in recent discovery runs. Avoid ALL of them AND any near-variants:\n"
            f"{avoid_lines}\n"
            "Generate ideas with COMPLETELY DIFFERENT mechanisms, signal constructions, and naming.\n"
        )

    if web_synthesis:
        extra.append(
            "\n=== RECENT RESEARCH CONTEXT — USE AS INSPIRATION ===\n"
            f"{web_synthesis}\n"
            "Use these mechanism insights as INSPIRATION for new family ideas. "
            "Ground novel ideas in these concepts where they apply to OHLCV/VIX/FG signals.\n"
        )

    if focus_themes:
        theme_lines = "\n".join(f"  - {t}" for t in focus_themes[:8])
        extra.append(
            "\n=== PRIORITY ANGLES FOR THIS RUN ===\n"
            f"Especially explore these underrepresented angles:\n{theme_lines}\n"
        )

    # Replace the static "Propose exactly 15" with the dynamic n_ideas count
    prompt = base + "".join(extra) + f"\n{_GENERATION_PROMPT_CLOSING}"
    if n_ideas != 15:
        prompt = prompt.replace(
            "Propose exactly 15 NEW and STRUCTURALLY DISTINCT",
            f"Propose exactly {n_ideas} NEW and STRUCTURALLY DISTINCT",
        ).replace(
            "Return a JSON array of exactly 15 objects.",
            f"Return a JSON array of exactly {n_ideas} objects.",
        )
    return prompt


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _call_minimax(
    system: str,
    user: str,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    retries: int = 2,
) -> str:
    from llm.providers import build_minimax_provider

    provider = build_minimax_provider()

    def _attempt(temp: float) -> str:
        resp = provider.create_message(
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return resp.text.strip()

    text = _strip_think_blocks(_attempt(temperature))
    for attempt in range(retries):
        if text:
            return text
        text = _strip_think_blocks(_attempt(max(0.01, temperature - 0.1 * (attempt + 1))))
    return text


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by MiniMax-M2 models.

    Handles both closed blocks and truncated responses where </think> was cut off.
    """
    import re
    # Closed block: <think>...</think> → remove it
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped:
        return stripped
    # Unclosed block (max_tokens cut off inside <think>): find last JSON start after <think>
    if "<think>" in text:
        last_bracket = max(text.rfind("\n["), text.rfind("\n{"))
        if last_bracket >= 0:
            return text[last_bracket:].strip()
    return text.strip()


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
        return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        return []


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Generation step
# ---------------------------------------------------------------------------

def _generate_family_ideas(
    *,
    n_ideas: int = 15,
    temperature: float = 0.75,
    names_to_avoid: list[str] | None = None,
    web_synthesis: str | None = None,
    focus_themes: list[str] | None = None,
) -> list[dict[str, Any]]:
    prompt = _build_generation_prompt(
        names_to_avoid=names_to_avoid,
        web_synthesis=web_synthesis,
        focus_themes=focus_themes,
        n_ideas=n_ideas,
    )
    text = _call_minimax(_GENERATION_SYSTEM, prompt, max_tokens=16000, temperature=temperature)
    ideas = _parse_json_array(text)
    # retry with higher temperature if underfilled
    if len(ideas) < max(5, n_ideas // 2):
        text = _call_minimax(_GENERATION_SYSTEM, prompt, max_tokens=16000, temperature=min(0.95, temperature + 0.15))
        ideas = _parse_json_array(text)
    return ideas[:n_ideas]


# ---------------------------------------------------------------------------
# Clustering/ranking step
# ---------------------------------------------------------------------------

def _cluster_and_rank(ideas: list[dict[str, Any]]) -> dict[str, Any]:
    if not ideas:
        return {}
    candidates_json = json.dumps(
        [{"family_name": d.get("family_name"), "edge_source": d.get("edge_source"),
          "why_not_momentum": d.get("why_not_momentum"), "orthogonality_claim": d.get("orthogonality_claim"),
          "implementation_complexity": d.get("implementation_complexity"),
          "required_data": d.get("required_data"), "failure_modes": d.get("failure_modes")}
         for d in ideas],
        indent=2,
    )
    prompt = _CLUSTERING_PROMPT.format(
        n_candidates=len(ideas),
        existing_families=_EXISTING_FAMILIES_SUMMARY,
        candidates_json=candidates_json,
    )
    text = _call_minimax(_CLUSTERING_SYSTEM, prompt, max_tokens=8192, temperature=0.3)
    result = _parse_json_object(text)
    if not result or not result.get("scored_candidates"):
        # fallback: assign trivial scores in insertion order
        result = {
            "clusters": [{"cluster_id": "cluster_1", "cluster_label": "all", "members": [d.get("family_name") for d in ideas], "best_representative": ideas[0].get("family_name"), "cluster_rationale": "unclustered"}],
            "scored_candidates": [
                {"family_name": d.get("family_name"), "cluster_id": "cluster_1",
                 "orthogonality_score": 0.5, "feasibility_score": 0.5,
                 "expected_robustness": 0.5, "research_value": 0.5,
                 "ranking_score": 0.5, "is_best_in_cluster": (i == 0), "duplicate_of": None,
                 "ranking_rationale": "fallback score"}
                for i, d in enumerate(ideas)
            ],
            "top_candidates": [d.get("family_name") for d in ideas[:6]],
            "discovery_summary": "Fallback: LLM clustering unavailable.",
        }
    return result


# ---------------------------------------------------------------------------
# Build FamilyCandidateRecord from raw idea + ranking info
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = 0.5) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _build_candidate_record(
    idea: dict[str, Any],
    scoring: dict[str, Any],
    batch_id: str,
    cluster_map: dict[str, dict[str, Any]],
) -> FamilyCandidateRecord:
    family_name = _safe_str(idea.get("family_name"), "unknown_family")
    cluster_id = _safe_str(scoring.get("cluster_id"))
    cluster_info = cluster_map.get(cluster_id, {})

    ts = utc_now_iso()
    safe_name = family_name.replace(" ", "_").replace("-", "_").lower()
    candidate_id = f"fam_{safe_name}_{ts.replace(':', '').replace('-', '').replace('+', '')[:20]}"

    first_test = idea.get("first_test_family_template")
    if not isinstance(first_test, dict):
        first_test = {"signal_construction": _safe_str(idea.get("edge_source"))}

    return FamilyCandidateRecord(
        candidate_id=candidate_id,
        family_name=family_name,
        edge_source=_safe_str(idea.get("edge_source")),
        why_it_should_exist=_safe_str(idea.get("why_it_should_exist")),
        why_not_momentum=_safe_str(idea.get("why_not_momentum")),
        why_not_superstock=_safe_str(idea.get("why_not_superstock")),
        required_data=_safe_str(idea.get("required_data")),
        implementation_complexity=_safe_str(idea.get("implementation_complexity"), "medium"),
        expected_holding_horizon=_safe_str(idea.get("expected_holding_horizon")),
        first_test_family_template=first_test,
        novelty_reason=_safe_str(idea.get("novelty_reason")),
        timestamp_utc=ts,
        status="new_family_candidate",
        cluster_id=cluster_id or None,
        cluster_label=_safe_str(cluster_info.get("cluster_label")) or None,
        orthogonality_score=_safe_float(scoring.get("orthogonality_score")),
        feasibility_score=_safe_float(scoring.get("feasibility_score")),
        expected_robustness=_safe_float(scoring.get("expected_robustness")),
        research_value=_safe_float(scoring.get("research_value")),
        ranking_score=_safe_float(scoring.get("ranking_score")),
        duplicate_of=_safe_str(scoring.get("duplicate_of")) or None,
        is_best_in_cluster=bool(scoring.get("is_best_in_cluster", False)),
        orthogonality_claim=_safe_str(idea.get("orthogonality_claim")) or None,
        expected_sharpe_range=_safe_str(idea.get("expected_sharpe_range")) or None,
        failure_modes=_safe_str(idea.get("failure_modes")) or None,
        discovery_batch_id=batch_id,
        metadata={
            "ranking_rationale": scoring.get("ranking_rationale"),
            "cluster_rationale": cluster_info.get("cluster_rationale"),
            "idea_raw": idea,
        },
    )


# ---------------------------------------------------------------------------
# Seed IdeaRecord from a controlled_probe candidate
# ---------------------------------------------------------------------------

def _infer_synth_class(family_name: str, edge_source: str) -> str:
    """Map a discovery candidate concept to the closest structural synthesis class.

    This controls which config override block fires in experiment_refinement.py's
    _synthesized_structural_config(), ensuring the probe explores a distinct region
    of parameter space rather than defaulting to the momentum baseline.
    """
    combined = (family_name + " " + edge_source).lower()
    # Signal construction / ranking mechanism concepts (OHLC-derived signals)
    if any(w in combined for w in ["range", "ohlc", "hl_", "intraday", "candle", "body", "wick",
                                     "close_position", "range_imbalance", "range_position",
                                     "price_position", "directional_imprint"]):
        return "ranking_rethink"
    # Breadth, market-wide signals
    if any(w in combined for w in ["breadth", "advance", "decline", "width", "thrust", "sequential"]):
        return "breadth_filter"
    # Regime / macro-state gating
    if any(w in combined for w in ["regime", "transition", "vix", "fear", "risk_off", "risk_on",
                                     "macro", "state", "crash", "drawdown"]):
        return "regime_filter"
    # Volume dynamics
    if any(w in combined for w in ["volume", "dryup", "dry_up", "accumulation", "distribution",
                                     "turnover", "liquidity", "amihud"]):
        return "regime_filter"
    # Per-stock state models
    if any(w in combined for w in ["imprint", "directional", "per_stock", "individual", "memory"]):
        return "state_model"
    # Sector / cross-asset
    if any(w in combined for w in ["sector", "rotation", "tilt", "cross_sectional", "dispersion"]):
        return "cross_family_mechanism"
    # Volatility
    if any(w in combined for w in ["volatility", "vol_", "realized_vol", "implied", "compression"]):
        return "regime_filter"
    # Calendar / seasonality
    if any(w in combined for w in ["calendar", "seasonal", "month", "turn_of", "holiday", "event"]):
        return "portfolio_overlay"
    # Default: treat as a new ranking concept (most discovery families propose signal alternatives)
    return "ranking_rethink"


def _seed_idea_from_candidate(
    candidate: FamilyCandidateRecord,
    workspace_root: str,
    *,
    existing_families: list[str] | None = None,
) -> IdeaRecord | None:
    """Convert a controlled_probe candidate into a structural IdeaRecord for the idea queue.

    We route it to the closest existing family as a structural-extension probe so the
    existing proposal pipeline can consume it without requiring a new strategy implementation.
    The template_similarity_class is derived from the candidate's concept so that
    _synthesized_structural_config() produces a distinctive, non-default parameter config.
    """
    if not existing_families:
        try:
            from experiment_spaces import list_searchable_families
            existing_families = list_searchable_families()
        except Exception:
            existing_families = ["momentum"]

    # Prefer momentum as the seed family; it has the most templates and is most likely
    # to accept structural-extension ideas.
    target_family = "momentum" if "momentum" in existing_families else (existing_families[0] if existing_families else "momentum")

    ts = utc_now_iso()
    safe_name = candidate.family_name.replace(" ", "_").lower()
    idea_id = f"idea_family_discovery_{safe_name}_{ts.replace(':', '').replace('-', '').replace('+', '')[:20]}"

    hypothesis = (
        f"[FamilyDiscovery] Probe for new '{candidate.family_name}' family concept seeded into {target_family}: "
        f"{candidate.edge_source}"
    )

    # Infer the most appropriate synthesis class from the candidate's concept.
    # This controls which parameter override block fires in _synthesized_structural_config().
    synth_class = _infer_synth_class(candidate.family_name, candidate.edge_source)

    first_test = candidate.first_test_family_template
    # Include actual momentum-compatible parameter hints so the config is distinctive.
    suggested_config: dict[str, Any] = {
        "family_discovery_candidate": candidate.family_name,
        "edge_source": candidate.edge_source[:120],
        "signal_construction": str(first_test.get("signal_construction", ""))[:200],
        "rebalance_freq": first_test.get("rebalance_freq", "weekly"),
        "top_n_pct": first_test.get("top_n_pct", 0.025),
        "synthesis_class": synth_class,
    }
    if isinstance(first_test.get("key_params"), dict):
        suggested_config["key_params"] = first_test["key_params"]

    return IdeaRecord(
        idea_id=idea_id,
        family=target_family,
        strategy_type="classical",
        hypothesis=hypothesis,
        source="structural_extension",
        priority=0.80,
        estimated_cost="low_medium_cpu",
        timestamp_utc=ts,
        novelty_score=round(max(0.70, _safe_float(candidate.orthogonality_score, 0.75)), 4),
        rationale=(
            f"Family-discovery probe: {candidate.why_it_should_exist[:300]} "
            f"Orthogonality={candidate.orthogonality_score:.2f} "
            f"Feasibility={candidate.feasibility_score:.2f}"
        ),
        suggested_template_id=f"family_discovery_{safe_name}",
        template_id=f"family_discovery_{safe_name}",
        suggested_config=suggested_config,
        idea_source="family_discovery",
        idea_provider=_PROVIDER,
        idea_model=_MODEL,
        idea_kind="new_family_discovery_probe",
        novelty_reason=(
            f"Family discovery: {candidate.novelty_reason[:200]}. "
            f"Why not momentum: {candidate.why_not_momentum[:120]}"
        ),
        is_new_idea=True,
        is_structurally_novel=True,
        is_out_of_box=True,
        is_uncommon_idea=True,
        structural_distance=round(min(0.97, max(0.70, _safe_float(candidate.orthogonality_score, 0.80))), 3),
        template_similarity_class=synth_class,
        uncommon_idea_reason=(
            f"Brand-new family concept: {candidate.family_name}. "
            f"{candidate.why_not_momentum[:100]}"
        ),
        metadata={
            "family_candidate_id": candidate.candidate_id,
            "discovery_family_name": candidate.family_name,
            "ranking_score": candidate.ranking_score,
            "orthogonality_score": candidate.orthogonality_score,
            "feasibility_score": candidate.feasibility_score,
            "expected_holding_horizon": candidate.expected_holding_horizon,
            "required_data": candidate.required_data,
            "is_new_idea": True,
            "is_branch_repeat": False,
            "is_structurally_novel": True,
            "is_out_of_box": True,
            "is_uncommon_idea": True,
            "consumable_by_proposal_flow": True,
            "inferred_synth_class": synth_class,
        },
    )


# ---------------------------------------------------------------------------
# Discovery report persistence
# ---------------------------------------------------------------------------

def _save_discovery_report(report: dict[str, Any], workspace_root: str) -> Path:
    reports_dir = Path(workspace_root) / "reports" / "family_discovery"
    reports_dir.mkdir(parents=True, exist_ok=True)
    batch_id = report.get("batch_id", "unknown")
    path = reports_dir / f"{batch_id}.json"
    _atomic_write_json(path, report)
    latest = reports_dir / "latest.json"
    _atomic_write_json(latest, report)
    return path


def load_latest_family_discovery_report(workspace_root: str = ".") -> dict[str, Any] | None:
    path = Path(workspace_root) / "reports" / "family_discovery" / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point: discover_and_rank_families
# ---------------------------------------------------------------------------

def discover_and_rank_families(
    *,
    workspace_root: str = ".",
    n_ideas: int = 15,
    top_k: int = 8,
    auto_promote_top: int = 3,
    seed_ideas_for_probe: bool = True,
    temperature: float = 0.75,
    max_active_probes: int = 6,
    probe_budget: int = 2,
    web_mode: str = "arxiv",
    max_web_queries: int = 5,
    max_web_results_per_query: int = 3,
) -> dict[str, Any]:
    """Run the family discovery pipeline with optional web research and memory-based anti-repeat.

    Args:
        workspace_root: Repository root path.
        n_ideas: How many family ideas to generate.
        top_k: How many top candidates to select.
        auto_promote_top: Number of top candidates to auto-promote to controlled_probe.
        seed_ideas_for_probe: If True, create IdeaRecord objects for promoted candidates.
        temperature: LLM generation temperature.
        max_active_probes: Hard cap on concurrent controlled_probe candidates.
        probe_budget: Max new IdeaRecords to seed per run.
        web_mode: "none" | "arxiv" | "hybrid" — web research mode.
        max_web_queries: Max search queries to run per discovery cycle.
        max_web_results_per_query: Max arXiv results per query.

    Returns:
        Discovery report dict with web_research, novelty, and repeat metadata.
    """
    from agents.family_discovery_memory import (
        compute_repeat_score,
        get_avoidance_context,
        update_memory_after_run,
    )

    ensure_helper_dirs(workspace_root)
    ts = utc_now_iso()
    batch_id = f"family_discovery_{ts.replace(':', '').replace('-', '').replace('+', '')[:20]}"

    print(f"[family_discovery] batch_id={batch_id}  web_mode={web_mode}", flush=True)

    # -----------------------------------------------------------------------
    # Load memory context: recently proposed families, themes to avoid
    # -----------------------------------------------------------------------
    avoidance = get_avoidance_context(workspace_root)
    names_to_avoid = avoidance.get("names_to_avoid", [])
    themes_to_avoid = avoidance.get("themes_to_avoid", [])
    total_past_runs = avoidance.get("total_past_runs", 0)
    if names_to_avoid:
        print(f"[family_discovery] memory: avoiding {len(names_to_avoid)} recently proposed families", flush=True)

    # -----------------------------------------------------------------------
    # Web research stage (optional)
    # -----------------------------------------------------------------------
    web_result: dict[str, Any] = {"mode": web_mode, "findings": [], "queries_used": [], "themes_used": [], "n_results": 0, "sources": [], "synthesis": ""}
    focus_themes: list[str] = []

    if web_mode != "none":
        try:
            from agents.family_discovery_web import (
                plan_search_queries,
                run_bounded_web_research,
                synthesize_web_findings,
            )

            print(f"[family_discovery] web_research: planning {max_web_queries} search queries ...", flush=True)
            queries = plan_search_queries(
                existing_families=_EXISTING_FAMILIES_SUMMARY,
                themes_to_avoid=themes_to_avoid,
                n_queries=max_web_queries,
            )
            focus_themes = [q.get("theme", "") for q in queries if q.get("theme")]

            raw_web = run_bounded_web_research(
                queries,
                max_queries=max_web_queries,
                max_results_per_query=max_web_results_per_query,
            )
            web_result.update(raw_web)

            if raw_web.get("n_results", 0) > 0:
                print(f"[family_discovery] web_research: synthesizing {raw_web['n_results']} papers ...", flush=True)
                synthesis = synthesize_web_findings(raw_web.get("findings", []))
                web_result["synthesis"] = synthesis
                if synthesis:
                    print(f"[family_discovery] web_research: synthesis ready ({len(synthesis)} chars)", flush=True)
        except Exception as _web_err:
            print(f"[family_discovery] web_research: skipped err={_web_err}", flush=True)
            web_result["error"] = str(_web_err)

    # -----------------------------------------------------------------------
    # Generation step: enriched with memory + web context
    # -----------------------------------------------------------------------
    print(f"[family_discovery] step1: generating {n_ideas} family ideas via MiniMax ...", flush=True)

    raw_ideas = _generate_family_ideas(
        n_ideas=n_ideas,
        temperature=temperature,
        names_to_avoid=names_to_avoid or None,
        web_synthesis=web_result.get("synthesis") or None,
        focus_themes=focus_themes or None,
    )
    print(f"[family_discovery] step1: received {len(raw_ideas)} ideas", flush=True)

    if not raw_ideas:
        return {"batch_id": batch_id, "status": "failed", "reason": "generation returned no ideas", "timestamp_utc": ts}

    print(f"[family_discovery] step2: clustering and ranking {len(raw_ideas)} candidates ...", flush=True)
    ranking_result = _cluster_and_rank(raw_ideas)

    scored_map: dict[str, dict[str, Any]] = {
        s["family_name"]: s for s in (ranking_result.get("scored_candidates") or [])
    }
    cluster_map: dict[str, dict[str, Any]] = {
        c["cluster_id"]: c for c in (ranking_result.get("clusters") or [])
    }
    idea_map: dict[str, dict[str, Any]] = {d.get("family_name", ""): d for d in raw_ideas}

    # Build FamilyCandidateRecord for every idea
    all_records: list[FamilyCandidateRecord] = []
    for idea in raw_ideas:
        family_name = idea.get("family_name", "")
        scoring = scored_map.get(family_name, {"family_name": family_name, "ranking_score": 0.5, "is_best_in_cluster": False})
        record = _build_candidate_record(idea, scoring, batch_id, cluster_map)
        all_records.append(record)

    # Sort by ranking_score descending, cluster representatives first
    def _sort_key(r: FamilyCandidateRecord) -> tuple[int, float]:
        return (0 if r.is_best_in_cluster else 1, -(_safe_float(r.ranking_score, 0.0)))

    all_records.sort(key=_sort_key)

    # Select top_k candidates
    top_names_from_llm: list[str] = ranking_result.get("top_candidates") or []
    # Prefer LLM-selected top_k, fall back to ranking_score order
    top_records = [r for r in all_records if r.family_name in top_names_from_llm][:top_k]
    if len(top_records) < top_k:
        remaining = [r for r in all_records if r.family_name not in {t.family_name for t in top_records}]
        top_records.extend(remaining[: top_k - len(top_records)])

    # -----------------------------------------------------------------------
    # Compute repeat scores for every candidate (memory-aware novelty)
    # -----------------------------------------------------------------------
    repeat_meta: dict[str, dict[str, Any]] = {}
    n_penalized = 0
    for rec in all_records:
        penalty, reason = compute_repeat_score(rec.family_name, rec.edge_source, workspace_root)
        is_fresh = penalty < 0.30
        repeat_meta[rec.candidate_id] = {
            "repeat_penalty": round(penalty, 3),
            "repeat_reason": reason,
            "is_fresh_family": is_fresh,
            "web_research_used": web_mode != "none",
        }
        if penalty >= 0.30:
            n_penalized += 1

    if n_penalized:
        print(f"[family_discovery] repeat_suppression: {n_penalized}/{len(all_records)} candidates penalized", flush=True)

    # Re-score with repeat penalty: effective_score = ranking_score * (1 - 0.5 * penalty)
    # (0.5 weight so penalty softens but doesn't completely negate LLM ranking)
    def _effective_score(r: FamilyCandidateRecord) -> float:
        base = _safe_float(r.ranking_score, 0.5)
        penalty = repeat_meta.get(r.candidate_id, {}).get("repeat_penalty", 0.0)
        return base * (1.0 - 0.5 * penalty)

    # Re-sort top_k using effective score
    all_records.sort(key=lambda r: (0 if r.is_best_in_cluster else 1, -_effective_score(r)))
    top_records = [r for r in all_records if r.family_name in top_names_from_llm][:top_k]
    if len(top_records) < top_k:
        remaining = [r for r in all_records if r.family_name not in {t.family_name for t in top_records}]
        top_records.extend(remaining[: top_k - len(top_records)])

    # Patch repeat metadata into record metadata before saving
    patched_records: list[FamilyCandidateRecord] = []
    from dataclasses import replace as dc_replace_record
    for rec in all_records:
        rma = repeat_meta.get(rec.candidate_id, {})
        merged_meta = dict(rec.metadata or {})
        merged_meta.update(rma)
        patched_records.append(dc_replace_record(rec, metadata=merged_meta))
    all_records = patched_records

    # Save ALL candidates
    for record in all_records:
        save_family_candidate(record, workspace_root)

    print(f"[family_discovery] saved {len(all_records)} candidates to queues/family_candidates/", flush=True)

    # -----------------------------------------------------------------------
    # Enforce max_active_probes cap
    # -----------------------------------------------------------------------
    _current_probes: list[dict] = []
    try:
        from family_candidate_store import get_controlled_probes
        _current_probes = get_controlled_probes(workspace_root)
    except Exception:
        pass
    _probe_slots = max(0, max_active_probes - len(_current_probes))
    _effective_auto_promote = min(auto_promote_top, _probe_slots)
    if _probe_slots < auto_promote_top:
        print(
            f"[family_discovery] probe cap: {len(_current_probes)} active probes, "
            f"promoting {_effective_auto_promote}/{auto_promote_top} (max_active_probes={max_active_probes})",
            flush=True,
        )

    # Promote top N (prefer fresh candidates)
    probe_seeds: list[IdeaRecord] = []
    promoted: list[str] = []
    _seeds_this_run = 0
    # Sort top_records so fresh candidates are promoted first
    fresh_first = sorted(
        top_records[:_effective_auto_promote + 2],
        key=lambda r: repeat_meta.get(r.candidate_id, {}).get("repeat_penalty", 0.0),
    )
    for record in fresh_first[:_effective_auto_promote]:
        update_family_candidate_status(
            record.candidate_id,
            "controlled_probe",
            workspace_root,
            extra_updates={"promoted_at": utc_now_iso(), "promotion_reason": "auto_promote_top"},
        )
        promoted.append(record.family_name)
        if seed_ideas_for_probe and record.duplicate_of is None and _seeds_this_run < probe_budget:
            idea = _seed_idea_from_candidate(record, workspace_root)
            if idea is not None:
                try:
                    save_idea_record(idea, workspace_root=workspace_root)
                    probe_seeds.append(idea)
                    _seeds_this_run += 1
                except Exception:
                    pass

    if promoted:
        print(f"[family_discovery] promoted to controlled_probe: {promoted}", flush=True)
    if probe_seeds:
        print(f"[family_discovery] seeded {len(probe_seeds)} IdeaRecords into idea queue", flush=True)

    # -----------------------------------------------------------------------
    # Update discovery memory
    # -----------------------------------------------------------------------
    try:
        edge_sources = [r.edge_source for r in all_records if r.edge_source]
        update_memory_after_run(
            workspace_root,
            batch_id=batch_id,
            family_names=[r.family_name for r in all_records],
            mechanisms=edge_sources,
            search_themes=web_result.get("themes_used") or focus_themes,
            search_queries=web_result.get("queries_used"),
            web_sources=web_result.get("sources"),
            promoted=promoted,
        )
        print(f"[family_discovery] memory updated ({len(all_records)} names added)", flush=True)
    except Exception as _mem_err:
        print(f"[family_discovery] memory update failed: {_mem_err}", flush=True)

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    n_fresh = sum(1 for m in repeat_meta.values() if m.get("is_fresh_family"))
    n_penalized_final = len(all_records) - n_fresh

    top_candidate_dicts = []
    for r in top_records:
        rma = repeat_meta.get(r.candidate_id, {})
        top_candidate_dicts.append({
            "family_name": r.family_name,
            "edge_source": r.edge_source,
            "ranking_score": r.ranking_score,
            "effective_score": round(_effective_score(r), 4),
            "repeat_penalty": rma.get("repeat_penalty", 0.0),
            "repeat_reason": rma.get("repeat_reason", "fresh"),
            "is_fresh_family": rma.get("is_fresh_family", True),
            "web_research_used": rma.get("web_research_used", False),
            "orthogonality_score": r.orthogonality_score,
            "feasibility_score": r.feasibility_score,
            "expected_robustness": r.expected_robustness,
            "research_value": r.research_value,
            "cluster_label": r.cluster_label,
            "is_best_in_cluster": r.is_best_in_cluster,
            "status": r.status if r.family_name not in promoted else "controlled_probe",
            "expected_holding_horizon": r.expected_holding_horizon,
            "implementation_complexity": r.implementation_complexity,
            "why_not_momentum": r.why_not_momentum,
            "orthogonality_claim": r.orthogonality_claim,
            "expected_sharpe_range": r.expected_sharpe_range,
            "failure_modes": r.failure_modes,
            "candidate_id": r.candidate_id,
        })

    report: dict[str, Any] = {
        "batch_id": batch_id,
        "timestamp_utc": ts,
        "status": "complete",
        "web_mode": web_mode,
        "n_generated": len(raw_ideas),
        "n_saved": len(all_records),
        "n_top": len(top_records),
        "n_promoted": len(promoted),
        "n_probe_seeds": len(probe_seeds),
        "n_fresh": n_fresh,
        "n_penalized": n_penalized_final,
        "past_discovery_runs": total_past_runs,
        "promoted_families": promoted,
        "probe_seed_idea_ids": [i.idea_id for i in probe_seeds],
        "top_candidates": top_candidate_dicts,
        "clusters": ranking_result.get("clusters") or [],
        "discovery_summary": ranking_result.get("discovery_summary", ""),
        "all_candidate_ids": [r.candidate_id for r in all_records],
        "candidates": [asdict(r) for r in all_records],
        "web_research": {
            "mode": web_mode,
            "n_queries": len(web_result.get("queries_used", [])),
            "n_results": web_result.get("n_results", 0),
            "queries_used": web_result.get("queries_used", []),
            "themes_used": web_result.get("themes_used", []),
            "sources": web_result.get("sources", []),
            "synthesis_length": len(web_result.get("synthesis", "")),
            "error": web_result.get("error"),
        },
        "memory_context": {
            "names_avoided": len(names_to_avoid),
            "themes_avoided": len(themes_to_avoid),
            "total_past_runs": total_past_runs,
        },
    }

    report_path = _save_discovery_report(report, workspace_root)
    print(f"[family_discovery] report saved to {report_path}", flush=True)
    return report


# ---------------------------------------------------------------------------
# Family refinement loop
# ---------------------------------------------------------------------------

_REFINEMENT_SYSTEM = (
    "You are a quantitative research analyst refining a strategy family concept. "
    "Return only valid JSON."
)

_REFINEMENT_PROMPT = """\
You are refining a strategy family candidate that has been promoted to controlled probe status.

FAMILY CONCEPT:
{concept_json}

CURRENT RESEARCH CONTEXT:
- Best momentum walk-forward Sharpe: {best_momentum_sharpe}
- Number of experiments run so far for this family: {n_experiments}
- Key failure modes to avoid: {failure_modes}

YOUR TASK: Deepen and sharpen this family concept. Produce:
1. A more precise signal construction recipe
2. 3 concrete parameter configurations to test first (spanning the parameter space)
3. Regime conditions where this family should outperform
4. Regime conditions where it will likely underperform
5. A go/no-go criterion: what walk-forward Sharpe threshold justifies continued investment?
6. A suggested first-batch experiment plan (6–12 experiments, parameter grid)

Return a JSON object:
{{
  "refined_signal": "precise signal construction (step by step)",
  "first_configs": [
    {{"config_label": "conservative", "params": {{}}}},
    {{"config_label": "baseline", "params": {{}}}},
    {{"config_label": "aggressive", "params": {{}}}}
  ],
  "outperform_regimes": ["list of market regimes where this earns"],
  "underperform_regimes": ["list of market regimes where this fails"],
  "go_nogo_criterion": "e.g. mean WF Sharpe > 0.4 across 14 windows",
  "first_batch_plan": {{
    "n_experiments": 8,
    "param_grid": {{"param_name": ["value1", "value2"]}},
    "rationale": "why this grid"
  }},
  "refinement_notes": "key insights and open questions"
}}

Return ONLY the JSON object."""


def refine_family_candidate(
    candidate_id: str,
    *,
    workspace_root: str = ".",
    experiments_dir: str = "experiments",
) -> dict[str, Any] | None:
    """Refine an existing family candidate's concept via a targeted LLM call.

    Returns the refinement dict or None if the candidate is not found.
    """
    all_candidates = load_family_candidates(workspace_root)
    candidate = next((c for c in all_candidates if c.get("candidate_id") == candidate_id), None)
    if candidate is None:
        print(f"[family_discovery] candidate {candidate_id} not found", flush=True)
        return None

    # Load best momentum Sharpe for context
    best_momentum_sharpe = "unknown"
    try:
        from experiment_best_results import top_results_per_family
        ranked = top_results_per_family(limit=1, base_dir=experiments_dir)
        best_momentum = ranked.get("momentum")
        if best_momentum is not None and not best_momentum.empty:
            best_momentum_sharpe = str(best_momentum.iloc[0].get("wf_sharpe", "unknown"))
    except Exception:
        pass

    concept_json = json.dumps(
        {k: v for k, v in candidate.items() if k not in {"metadata", "candidate_id", "timestamp_utc", "discovery_batch_id"}},
        indent=2,
    )
    prompt = _REFINEMENT_PROMPT.format(
        concept_json=concept_json,
        best_momentum_sharpe=best_momentum_sharpe,
        n_experiments=0,
        failure_modes=candidate.get("failure_modes", "unknown"),
    )
    text = _call_minimax(_REFINEMENT_SYSTEM, prompt, max_tokens=2048, temperature=0.4)
    refinement = _parse_json_object(text)
    if not refinement:
        return None

    # Persist refinement notes back to candidate record
    update_family_candidate_status(
        candidate_id, candidate.get("status", "controlled_probe"), workspace_root,
        extra_updates={"refinement_notes": refinement.get("refinement_notes"), "refinement": refinement},
    )
    print(f"[family_discovery] refined candidate {candidate_id}", flush=True)
    return refinement


# ---------------------------------------------------------------------------
# Compact report printer
# ---------------------------------------------------------------------------

def print_discovery_report(report: dict[str, Any]) -> None:
    print("=" * 72, flush=True)
    print(f"FAMILY DISCOVERY REPORT — batch {report.get('batch_id', '?')}", flush=True)
    print("=" * 72, flush=True)

    # Header stats
    n_gen = report.get("n_generated", 0)
    n_fresh = report.get("n_fresh", n_gen)
    n_pen = report.get("n_penalized", 0)
    past = report.get("past_discovery_runs", 0)
    print(
        f"Generated: {n_gen}  Top-K: {report.get('n_top', 0)}  "
        f"Promoted: {report.get('n_promoted', 0)}  "
        f"Fresh: {n_fresh}  Penalized: {n_pen}  PastRuns: {past}",
        flush=True,
    )

    # Web research summary
    wr = report.get("web_research") or {}
    web_mode = wr.get("mode", report.get("web_mode", "none"))
    if web_mode and web_mode != "none":
        print(
            f"Web: mode={web_mode}  queries={wr.get('n_queries', 0)}  "
            f"papers={wr.get('n_results', 0)}  "
            f"themes={','.join((wr.get('themes_used') or [])[:6])}",
            flush=True,
        )
        if wr.get("error"):
            print(f"Web error: {wr['error']}", flush=True)

    # Memory context
    mc = report.get("memory_context") or {}
    if mc.get("names_avoided"):
        print(f"Memory: avoided {mc['names_avoided']} recent names, {mc.get('themes_avoided', 0)} themes", flush=True)

    print(flush=True)

    summary = report.get("discovery_summary", "")
    if summary:
        print(f"Summary: {summary[:300]}", flush=True)
        print(flush=True)

    clusters = report.get("clusters") or []
    if clusters:
        print(f"Clusters ({len(clusters)}):", flush=True)
        for c in clusters:
            print(f"  [{c.get('cluster_id')}] {c.get('cluster_label', '?'):<30} "
                  f"— members: {c.get('members')}", flush=True)
        print(flush=True)

    promoted_set = set(report.get("promoted_families") or [])
    candidates = report.get("top_candidates") or []
    print("Top Candidates (ranked by effective_score = rank × (1 - 0.5×penalty)):", flush=True)
    for i, c in enumerate(candidates, 1):
        name = c.get("family_name", "?")
        promoted_marker = " ← PROMOTED" if name in promoted_set else ""
        penalty = c.get("repeat_penalty", 0.0)
        fresh_tag = "FRESH" if c.get("is_fresh_family", True) else f"REPEAT({penalty:.2f})"
        eff = c.get("effective_score") or c.get("ranking_score", 0.0)
        print(
            f"  #{i:2d}  {name:<38s}  eff={eff:.3f}  "
            f"orth={c.get('orthogonality_score', 0):.2f}  "
            f"feas={c.get('feasibility_score', 0):.2f}  "
            f"[{fresh_tag}]{promoted_marker}",
            flush=True,
        )
        print(f"       edge: {c.get('edge_source', '')[:80]}", flush=True)
        if penalty >= 0.30:
            print(f"       REPEAT_REASON: {c.get('repeat_reason', '?')}", flush=True)
        print(f"       why_distinct: {c.get('why_not_momentum', '')[:80]}", flush=True)
        print(f"       horizon: {c.get('expected_holding_horizon', '?')}  "
              f"complexity: {c.get('implementation_complexity', '?')}  "
              f"sharpe_est: {c.get('expected_sharpe_range', '?')}", flush=True)
        print(flush=True)

    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Family discovery agent — discovers new strategy families via MiniMax")
    parser.add_argument("--workspace-root", default=".", help="Repository root")
    parser.add_argument("--experiments-dir", default="experiments", help="Experiment store path")
    parser.add_argument("--n-ideas", type=int, default=15, help="Number of family ideas to generate")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top candidates to select")
    parser.add_argument("--auto-promote", type=int, default=3, help="Number of top candidates to auto-promote to controlled_probe")
    parser.add_argument("--no-seed-ideas", action="store_true", help="Do not seed IdeaRecords for promoted candidates")
    parser.add_argument("--temperature", type=float, default=0.75, help="LLM generation temperature")
    parser.add_argument("--refine", default=None, help="Refine a specific candidate_id instead of running discovery")
    parser.add_argument("--web-mode", default="arxiv", choices=["none", "arxiv", "hybrid"],
                        help="Web research mode: none | arxiv | hybrid (default: arxiv)")
    parser.add_argument("--max-web-queries", type=int, default=5, help="Max arXiv queries per run (default: 5)")
    parser.add_argument("--max-web-results", type=int, default=3, help="Max arXiv results per query (default: 3)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.refine:
        result = refine_family_candidate(
            args.refine,
            workspace_root=args.workspace_root,
            experiments_dir=args.experiments_dir,
        )
        if result:
            print(json.dumps(result, indent=2))
            return 0
        print(f"[family_discovery] refinement failed for {args.refine}")
        return 1

    report = discover_and_rank_families(
        workspace_root=args.workspace_root,
        n_ideas=args.n_ideas,
        top_k=args.top_k,
        auto_promote_top=args.auto_promote,
        seed_ideas_for_probe=not args.no_seed_ideas,
        temperature=args.temperature,
        web_mode=args.web_mode,
        max_web_queries=args.max_web_queries,
        max_web_results_per_query=args.max_web_results,
    )
    print_discovery_report(report)
    return 0 if report.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

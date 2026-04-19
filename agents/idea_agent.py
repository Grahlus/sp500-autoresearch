#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

import pandas as pd
import requests
from lxml import html as lxml_html

from agents.schemas import IdeaRecord, ensure_helper_dirs, load_idea_records, load_latest_analysis_report, save_idea_record, utc_now_iso
from experiment_idea_yield import load_latest_idea_yield_summary
from experiment_best_results import top_results_per_family
from experiment_idea_library import list_idea_templates
from experiment_spaces import list_searchable_families
from experiment_store import load_results_index


FAMILY_STRATEGY_TYPES = {
    "momentum": "classical",
    "superstock": "classical",
    "ml_ranker": "ml",
    "rl_bandit": "rl",
}

OUT_OF_BOX_QUOTA_RATIO = 0.40
MAX_NEARBY_TEMPLATE_IDEAS = 2
MAX_HISTORY_MINING_IDEAS = 1
MAX_CONFIRMATION_REPRODUCE_IDEAS = 1
MAX_STRUCTURAL_IDEAS = 2

STRUCTURAL_SIMILARITY_CLASSES = {
    "portfolio_overlay": "portfolio_overlay",
    "regime_filter": "regime_filter",
    "ranking_rethink": "ranking_rethink",
    "state_model": "state_model",
    "position_sizing": "position_sizing",
    "cross_family_mechanism": "cross_family_mechanism",
    "architecture_family": "architecture_family",
    "policy_family": "policy_family",
    "crash_defense": "crash_defense",
    "breadth_filter": "breadth_filter",
    "literature_transfer": "literature_transfer",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only helper that writes idea records")
    parser.add_argument("--workspace-root", default=".", help="Repo root for helper queues/reports")
    parser.add_argument("--experiments-dir", default="experiments", help="Official experiment store to read")
    parser.add_argument("--family", default="all", help="Specific family or all")
    parser.add_argument("--limit", type=int, default=4, help="Maximum idea records to emit")
    return parser.parse_args(argv)


def _families(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list_searchable_families()
    return [normalized]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


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


def _load_latest_score_summary(workspace_root: str) -> dict[str, Any]:
    path = Path(workspace_root) / "reports" / "score_summaries" / "latest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _load_persisted_family_scorecards(experiments_dir: str) -> dict[str, Any]:
    path = Path(experiments_dir) / "scorecards" / "family_scorecards.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    families = payload.get("families")
    return families if isinstance(families, dict) else {}


def _analysis_context(workspace_root: str, experiments_dir: str) -> dict[str, Any]:
    latest_report = load_latest_analysis_report(workspace_root) or {}
    latest_score_summary = _load_latest_score_summary(workspace_root)
    persisted_scorecards = _load_persisted_family_scorecards(experiments_dir)
    latest_idea_yield = load_latest_idea_yield_summary(experiments_dir) or {}
    score_summary = latest_score_summary.get("score_summary") or (latest_report.get("summary") or {}).get("score_summary") or {}
    score_summary = {**persisted_scorecards, **score_summary}
    family_trends = latest_score_summary.get("family_trends") or (latest_report.get("summary") or {}).get("family_trends") or {}
    next_focus = latest_score_summary.get("next_focus") or latest_report.get("next_focus") or []
    return {
        "latest_report_id": latest_report.get("report_id"),
        "latest_score_summary_report_id": latest_score_summary.get("report_id"),
        "persisted_scorecards_loaded": bool(persisted_scorecards),
        "score_summary": score_summary,
        "family_trends": family_trends,
        "next_focus": next_focus,
        "idea_yield_report_id": latest_idea_yield.get("timestamp_utc"),
        "idea_yield_summary": latest_idea_yield,
        "best_viable_result": latest_score_summary.get("best_viable_result") or (latest_report.get("summary") or {}).get("best_viable_result"),
        "best_baseline_beating_result": latest_score_summary.get("best_baseline_beating_result")
        or (latest_report.get("summary") or {}).get("best_baseline_beating_result"),
    }


def _focus_for_family(context: dict[str, Any], family: str) -> dict[str, Any]:
    for item in context.get("next_focus") or []:
        if item.get("family") == family:
            return item
    return {}


def _idea_yield_for_family(context: dict[str, Any], family: str) -> dict[str, Any]:
    summary = context.get("idea_yield_summary") or {}
    families = summary.get("families") if isinstance(summary, dict) else {}
    if isinstance(families, dict):
        family_summary = families.get(family) or {}
        return family_summary if isinstance(family_summary, dict) else {}
    return {}


def _family_rows(index: pd.DataFrame, family: str) -> pd.DataFrame:
    if index.empty or "strategy_family" not in index:
        return pd.DataFrame()
    return index[index["strategy_family"].astype(str) == family].copy()


def _weak_patterns(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"weak_count": 0, "dead_zone_count": 0, "status_counts": {}}
    status_counts = frame.get("status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
    dead_zone_risk = pd.to_numeric(frame.get("dead_zone_risk", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    viable = frame.get("viable", pd.Series(dtype=object)).map(_truthy)
    objective = pd.to_numeric(frame.get("objective_score", pd.Series(dtype=float)), errors="coerce")
    weak = (~viable) | objective.fillna(0.0).lt(0.0)
    return {
        "weak_count": int(weak.sum()),
        "dead_zone_count": int((dead_zone_risk >= 0.50).sum()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }


def _load_recent_template_usage(index: pd.DataFrame, *, family: str, window: int = 40) -> Counter:
    """Return a Counter of template_id frequency in the most recent `window` experiments for a family."""
    if index.empty or "strategy_family" not in index.columns or "template_id" not in index.columns:
        return Counter()
    family_rows = index[index["strategy_family"].astype(str) == family].copy()
    if family_rows.empty:
        return Counter()
    if "timestamp_utc" in family_rows.columns:
        family_rows = family_rows.sort_values("timestamp_utc", ascending=False)
    recent = family_rows.head(window)
    template_ids = recent["template_id"].dropna().astype(str).tolist()
    return Counter(tid for tid in template_ids if tid and tid != "nan")


def _idea_signature_from_payload(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    family = str(payload.get("family") or "").strip().lower()
    idea_source = str(payload.get("idea_source") or payload.get("source") or "").strip().lower()
    source = str(payload.get("source") or "").strip().lower()
    idea_kind = str(payload.get("idea_kind") or metadata.get("idea_kind") or "").strip().lower()
    template_id = str(payload.get("template_id") or payload.get("suggested_template_id") or "").strip().lower()
    paper_title = str(payload.get("paper_title") or metadata.get("paper_title") or "").strip().lower()
    web_topic_slug = str(metadata.get("web_topic_slug") or "").strip().lower()
    return "|".join(part for part in [family, idea_source or source, template_id, idea_kind, paper_title, web_topic_slug] if part)


def _candidate_signature(record: IdeaRecord) -> str:
    return _idea_signature_from_payload(
        {
            "family": record.family,
            "idea_source": record.idea_source,
            "source": record.source,
            "idea_kind": record.idea_kind,
            "template_id": record.template_id,
            "paper_title": record.paper_title,
            "metadata": record.metadata or {},
        }
    )


def _load_recent_idea_signatures(workspace_root: str, *, window: int = 40) -> Counter[str]:
    records = load_idea_records(workspace_root)
    if not records:
        return Counter()
    records.sort(key=lambda payload: str(payload.get("timestamp_utc") or payload.get("idea_id") or ""))
    recent = records[-window:]
    signatures = [_idea_signature_from_payload(payload) for payload in recent]
    return Counter(signature for signature in signatures if signature)


def _candidate_category(record: IdeaRecord) -> str:
    if bool(record.is_out_of_box):
        return "out_of_box"
    if bool(record.is_structurally_novel):
        return "structural_novel"
    if record.source == "history_mining":
        return "history_mining"
    if record.idea_kind in {"branch_refinement", "confirmation_reproduce"}:
        return "confirmation_reproduce"
    if record.source in {"template_expansion", "template_library", "cross_family_hybrid"}:
        return "nearby_template"
    return "other"


def _freshness_bonus(signature: str, recent_idea_signatures: Counter[str]) -> float:
    if not signature:
        return 0.0
    count = recent_idea_signatures.get(signature, 0)
    if count <= 0:
        return 0.0
    return min(0.35, 0.12 * count)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Web-research helpers (deterministic search + MiniMax synthesis)
# ---------------------------------------------------------------------------

_WEB_RESEARCH_COOLDOWN_HOURS = 2
# How long to back off after a session-limit or auth error (hours)
_WEB_RESEARCH_SESSION_BACKOFF_HOURS = 5
_WEB_RESEARCH_ERROR_BACKOFF_HOURS = 1
_WEB_RESEARCH_LOCK_NAME = "web_research_state.json"
_WEB_RESEARCH_STATUS_NAME = "web_research_status.json"

# Rotating topic pool — each run picks the next unused topic so searches never
# repeat until the full cycle is exhausted.  Covers both orthodox momentum
# refinements AND unorthodox/alternative approaches.
_WEB_RESEARCH_TOPICS: list[dict[str, str]] = [
    # --- orthodox momentum refinements ---
    {
        "slug": "momentum_lookback_optimisation",
        "focus": "optimal lookback window length for cross-sectional price momentum",
        "angle": "formation period 6-12-18 months, skip 1 month, US equities 2020-2025",
    },
    {
        "slug": "momentum_volatility_scaling",
        "focus": "volatility-scaled or risk-managed momentum strategies",
        "angle": "vol-targeting, inverse-vol weighting, downside deviation scaling",
    },
    {
        "slug": "residual_momentum",
        "focus": "residual or industry-adjusted momentum signals",
        "angle": "orthogonalise returns to market/industry before ranking",
    },
    {
        "slug": "momentum_crash_protection",
        "focus": "momentum crash risk management and drawdown protection",
        "angle": "bear-market filters, VIX gating, short-term reversal overlay",
    },
    {
        "slug": "52_week_high_anchor",
        "focus": "52-week high anchoring and reference-price momentum",
        "angle": "George & Hwang PTH ratio, anchoring bias exploitation",
    },
    # --- unorthodox / alternative approaches ---
    {
        "slug": "earnings_momentum_drift",
        "focus": "post-earnings announcement drift (PEAD) combined with price momentum",
        "angle": "SUE scores, analyst revision momentum, earnings surprise persistence",
    },
    {
        "slug": "options_flow_signal",
        "focus": "options market signals predicting equity returns",
        "angle": "put-call ratio, implied volatility skew, unusual options activity as equity filter",
    },
    {
        "slug": "short_interest_contrarian",
        "focus": "short interest and short squeeze dynamics as equity selection signal",
        "angle": "days-to-cover, short utilisation, short-squeeze momentum",
    },
    {
        "slug": "insider_trading_signal",
        "focus": "insider trading filings (Form 4) as a predictive equity signal",
        "angle": "cluster buys, officer vs director buys, net insider buying rate",
    },
    {
        "slug": "alternative_data_sentiment",
        "focus": "alternative data sentiment signals: news, social media, web traffic",
        "angle": "RavenPack sentiment, Reddit/Twitter mentions, Google Trends as momentum overlay",
    },
    {
        "slug": "regime_switching_momentum",
        "focus": "regime-conditional momentum — adapting to market regimes",
        "angle": "HMM regimes, macro-regime gating, time-series momentum as filter",
    },
    {
        "slug": "fundamental_momentum",
        "focus": "fundamental momentum: earnings growth, revenue acceleration, margin expansion",
        "angle": "operating leverage, ROE trend, cash-flow momentum",
    },
    {
        "slug": "frog_in_pan_information_diffusion",
        "focus": "information diffusion rate and frog-in-the-pan momentum signal",
        "angle": "sign consistency of daily returns during formation period",
    },
    {
        "slug": "low_frequency_traders_momentum",
        "focus": "microstructure-based momentum: order flow imbalance and institutional footprint",
        "angle": "Amihud illiquidity, bid-ask spread dynamics, volume-weighted momentum",
    },
    {
        "slug": "calendar_seasonality",
        "focus": "calendar anomalies and seasonality overlaid on momentum",
        "angle": "January effect, Halloween indicator, turn-of-month, earnings season clusters",
    },
    {
        "slug": "supply_chain_momentum",
        "focus": "supply-chain and customer-supplier return predictability",
        "angle": "customer momentum, input-output network linkages, sector contagion",
    },
    {
        "slug": "machine_learning_momentum",
        "focus": "machine learning feature engineering for momentum signal improvement",
        "angle": "gradient boosting, LSTM, transformer on price/volume features vs linear momentum",
    },
    {
        "slug": "esg_momentum_interaction",
        "focus": "ESG scores interacting with price momentum for return prediction",
        "angle": "ESG momentum tilt, controversy event momentum, greenwashing screens",
    },
    {
        "slug": "factor_timing_momentum",
        "focus": "factor timing and momentum-of-factors (momentum applied to factor returns)",
        "angle": "value momentum, quality momentum, factor rotation strategies",
    },
    {
        "slug": "analyst_revision_momentum",
        "focus": "analyst earnings revision momentum and estimate dispersion signals",
        "angle": "FY1 EPS revision breadth, price target revision, recommendation upgrades",
    },
]

_JSON_SCHEMA = """\
For each idea found, return a JSON array where each element has exactly these fields:
{
  "title": "paper or strategy title",
  "source": "URL, arxiv/SSRN identifier, or publication name",
  "positive_result": true or false,
  "hypothesis": "one sentence — the actionable strategy idea",
  "idea_kind": "strong structural category such as new_risk_overlay_concept, new_regime_filter_concept, new_per_stock_state_model, literature_inspired_structural_template",
  "template_similarity_class": "portfolio_overlay | regime_filter | ranking_rethink | state_model | position_sizing | cross_family_mechanism | architecture_family | policy_family | crash_defense | breadth_filter | literature_transfer",
  "structural_distance": 0.0,
  "is_structurally_novel": true or false,
  "is_out_of_box": true or false,
  "is_uncommon_idea": true or false,
  "uncommon_idea_reason": "short reason why this is structurally different",
  "lookback_days": integer trading days or null,
  "skip_days": integer trading days to skip at end of formation (e.g. 21) or null,
  "hold_days": integer trading days holding period or null,
  "top_n_pct": float 0-1 fraction of universe (e.g. 0.02 = top 2%) or null,
  "techniques": ["short", "technique", "names"],
  "rationale": "2-3 sentences on why this is actionable and novel"
}
Return ONLY the JSON array, no markdown, no other text."""

_WEB_RESEARCH_PROMPT_TEMPLATE = """\
You are a quantitative finance research assistant. The strategy under development is a \
cross-sectional equity momentum strategy (SP500/Russell 1000 universe, daily data, \
walk-forward backtesting, mean Sharpe across 14 windows as objective).

TODAY'S RESEARCH FOCUS: {focus}
SEARCH ANGLE: {angle}

You are given a JSON array of search results. Turn the most relevant results into 3-5 \
actionable research ideas. Prefer recent, concrete, implementable papers or notes. \
Use only the supplied search results; do not invent citations. Prefer mechanism shifts, \
structural templates, and ideas that can become new template families over small parameter tweaks.

{json_schema}

SEARCH RESULTS:
{search_results}

If the search results are weak, return the strongest adjacent ideas anyway, but keep \
them grounded in the supplied sources."""


def _load_web_research_state(workspace_root: str) -> dict[str, Any]:
    lock_path = Path(workspace_root) / "queues" / "web_research" / _WEB_RESEARCH_LOCK_NAME
    if not lock_path.exists():
        return {}
    try:
        return json.loads(lock_path.read_text())
    except Exception:
        return {}


def _save_web_research_state(workspace_root: str, state: dict[str, Any]) -> None:
    lock_dir = Path(workspace_root) / "queues" / "web_research"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _WEB_RESEARCH_LOCK_NAME
    try:
        lock_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception:
        pass


def _save_web_research_status(workspace_root: str, status: dict[str, Any]) -> None:
    status_dir = Path(workspace_root) / "queues" / "web_research"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / _WEB_RESEARCH_STATUS_NAME
    try:
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    except Exception:
        pass


def _parse_iso_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except Exception:
        return None


def _derive_next_retry_ts(state: dict[str, Any]) -> float | None:
    next_retry = _parse_iso_timestamp(state.get("next_retry_at"))
    if next_retry is not None:
        return next_retry
    session_backoff_until = state.get("session_backoff_until")
    try:
        if session_backoff_until is not None:
            return float(session_backoff_until)
    except (TypeError, ValueError):
        pass
    last_attempt = _parse_iso_timestamp(state.get("last_attempt_at"))
    if last_attempt is None:
        last_attempt = _parse_iso_timestamp(state.get("last_run_ts"))
    if last_attempt is None:
        return None
    return last_attempt + (_WEB_RESEARCH_COOLDOWN_HOURS * 3600.0)


def _web_research_status(workspace_root: str) -> dict[str, Any]:
    state = _load_web_research_state(workspace_root)
    next_retry_ts = _derive_next_retry_ts(state)
    now_ts = time.time()
    status_state = str(state.get("backoff_state") or "ready")
    if next_retry_ts is not None and now_ts < next_retry_ts:
        if status_state == "ready":
            status_state = "cooldown"
    else:
        status_state = "ready"
    ideas_dir = Path(workspace_root) / "queues" / "ideas"
    web_ideas = 0
    if ideas_dir.exists():
        for path in ideas_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            metadata = payload.get("metadata") or {}
            if payload.get("web_search_used") or payload.get("idea_source") == "web_search" or metadata.get("web_search_used"):
                web_ideas += 1
    queued_total = len(list(ideas_dir.glob("*.json"))) if ideas_dir.exists() else 0
    return {
        "timestamp_utc": datetime.fromtimestamp(now_ts, UTC).isoformat(),
        "backoff_state": status_state,
        "backoff_reason": state.get("backoff_reason"),
        "session_limit_hit": bool(state.get("session_limit_hit", False)),
        "last_attempt_at": state.get("last_attempt_at") or _iso_timestamp(_parse_iso_timestamp(state.get("last_run_ts"))),
        "last_success_at": state.get("last_success_at"),
        "last_failure_at": state.get("last_failure_at"),
        "next_retry_at": _iso_timestamp(next_retry_ts),
        "queued_idea_count": queued_total,
        "queued_web_idea_count": web_ideas,
        "last_topic_slug": state.get("last_topic_slug"),
        "papers_found": int(state.get("papers_found") or 0),
        "used_topic_slugs": list(state.get("used_topic_slugs") or []),
        "web_search_available": bool(os.getenv("ANTHROPIC_API_KEY")),
    }


def _web_research_cooldown_ok(workspace_root: str) -> bool:
    """
    Return True if it is safe to attempt a web research call.
    Blocked if:
      - normal cooldown not expired, OR
      - a session-limit backoff is active
    """
    state = _load_web_research_state(workspace_root)
    next_retry_ts = _derive_next_retry_ts(state)
    if next_retry_ts is None:
        return True
    return time.time() >= next_retry_ts


def _next_topic(state: dict[str, Any]) -> dict[str, str]:
    """Pick the next topic that hasn't been used recently, rotating through the pool."""
    used: list[str] = state.get("used_topic_slugs", [])
    # Find first topic not in used list
    for topic in _WEB_RESEARCH_TOPICS:
        if topic["slug"] not in used:
            return topic
    # Full cycle complete — reset and start over
    return _WEB_RESEARCH_TOPICS[0]


def _search_web_research_results(topic: dict[str, str], *, timeout: int = 20, limit: int = 8) -> list[dict[str, Any]]:
    query = f"{topic['focus']} {topic['angle']}"
    results: list[dict[str, Any]] = []

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = lxml_html.fromstring(response.text)
        for node in root.xpath("//div[contains(@class, 'result')]"):
            anchors = node.xpath(".//a[contains(@class, 'result__a')]")
            if not anchors:
                continue
            anchor = anchors[0]
            title = " ".join(anchor.text_content().split()).strip()
            href = str(anchor.get("href") or "").strip()
            snippets = node.xpath(".//a[contains(@class, 'result__snippet')] | .//div[contains(@class, 'result__snippet')]")
            snippet = " ".join(snippets[0].text_content().split()).strip() if snippets else ""
            if not title or not href:
                continue
            results.append({"title": title, "source": href, "snippet": snippet})
            if len(results) >= limit:
                break
    except Exception:
        results = []

    if results:
        return results

    # DuckDuckGo HTML is often blocked in headless environments; fall back to
    # arXiv's public Atom API so the MiniMax synthesis path still has sources.
    raw_tokens = re.findall(r"[a-z0-9]+", query.lower())
    stopwords = {
        "and", "or", "the", "a", "an", "of", "for", "to", "with", "in", "on", "by", "from", "as", "at", "into",
        "over", "under", "between", "using", "via", "risk", "managed", "model", "models", "strategies",
    }
    cleaned_tokens: list[str] = []
    for token in raw_tokens:
        if token in stopwords or token in cleaned_tokens:
            continue
        cleaned_tokens.append(token)
        if len(cleaned_tokens) >= 8:
            break
    candidate_queries = [
        " ".join(cleaned_tokens[:8]).strip(),
        " ".join(re.findall(r"[a-z0-9]+", topic["focus"].lower())[:6]).strip(),
    ]

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for candidate in candidate_queries:
        if not candidate:
            continue
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": candidate,
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split()).strip()
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split()).strip()
            source = ""
            for link in entry.findall("atom:link", ns):
                if link.get("rel") == "alternate" and link.get("href"):
                    source = str(link.get("href")).strip()
                    break
            if not source:
                source = str(entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            if not title or not source:
                continue
            results.append({"title": title, "source": source, "snippet": summary})
            if len(results) >= limit:
                break
        if results:
            break
    return results


def _call_minimax_web_synthesis(
    *,
    topic: dict[str, str],
    search_results: list[dict[str, Any]],
    max_ideas: int = 4,
) -> list[dict[str, Any]]:
    from llm.providers import build_minimax_provider

    provider = build_minimax_provider()
    prompt = _WEB_RESEARCH_PROMPT_TEMPLATE.format(
        focus=topic["focus"],
        angle=topic["angle"],
        json_schema=_JSON_SCHEMA,
        search_results=json.dumps(search_results, indent=2),
    )

    def _parse_response(text: str) -> list[dict[str, Any]]:
        raw_text = text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```", 2)[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.rsplit("```", 1)[0].strip()
        data = json.loads(raw_text)
        if not isinstance(data, list):
            return []
        return [item for item in data[:max_ideas] if isinstance(item, dict)]

    response = provider.create_message(
        system="Return only JSON. Focus on mechanism shifts, structural templates, and explicit novelty.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.1,
    )
    data = _parse_response(response.text)
    if data:
        return data

    retry_prompt = (
        prompt
        + "\n\n"
        + "Important: return a non-empty JSON array with exactly 3 to 5 objects. "
        + "If the sources are weak, synthesize the strongest adjacent actionable ideas from them."
    )
    retry_response = provider.create_message(
        system="Return only JSON. Focus on mechanism shifts, structural templates, and explicit novelty.",
        messages=[{"role": "user", "content": retry_prompt}],
        max_tokens=2048,
        temperature=0.2,
    )
    return _parse_response(retry_response.text)


def _web_research_ideas(
    workspace_root: str,
    *,
    families: list[str] | None = None,
    max_ideas: int = 4,
    recent_idea_signatures: Counter[str] | None = None,
) -> list[IdeaRecord]:
    """Use deterministic web search + MiniMax synthesis to queue idea records."""
    if not _web_research_cooldown_ok(workspace_root):
        _save_web_research_status(workspace_root, _web_research_status(workspace_root))
        return []

    state = _load_web_research_state(workspace_root)
    topic = _next_topic(state)
    target_families = families or ["momentum"]
    ideas: list[IdeaRecord] = []
    papers: list[dict[str, Any]] = []
    search_results: list[dict[str, Any]] = []
    recent_idea_signatures = recent_idea_signatures or Counter()

    try:
        search_results = _search_web_research_results(topic)
        papers = _call_minimax_web_synthesis(
            topic=topic,
            search_results=search_results,
            max_ideas=max_ideas,
        )

        for i, paper in enumerate(papers[:max_ideas]):
            title = str(paper.get("title") or "")
            if not title:
                continue
            positive = bool(paper.get("positive_result", True))
            hypothesis = str(paper.get("hypothesis") or f"Research-seeded idea from: {title[:100]}")
            rationale = (
                f"[Topic: {topic['slug']}] "
                + str(paper.get("rationale") or f"Source: {paper.get('source', 'unknown')}")
            )
            techniques = paper.get("techniques") or []
            suggested_config: dict[str, Any] = {"web_topic": topic["slug"]}
            if paper.get("lookback_days") is not None:
                suggested_config["lookback_days"] = int(paper["lookback_days"])
            if paper.get("skip_days") is not None:
                suggested_config["skip_days"] = int(paper["skip_days"])
            if paper.get("hold_days") is not None:
                suggested_config["hold_days"] = int(paper["hold_days"])
            if paper.get("top_n_pct") is not None:
                suggested_config["top_n_pct"] = float(paper["top_n_pct"])
            if techniques:
                suggested_config["techniques"] = techniques

            family = target_families[i % len(target_families)]
            structural_distance = _safe_float(paper.get("structural_distance"), 0.82 if positive else 0.68)
            template_similarity_class = str(paper.get("template_similarity_class") or "literature_transfer")
            idea_kind = str(paper.get("idea_kind") or _structural_idea_kind("literature_transfer"))
            if not paper.get("uncommon_idea_reason"):
                paper["uncommon_idea_reason"] = "MiniMax web synthesis selected a structural hypothesis rather than a parameter tweak."
            priority = 0.84 if positive else 0.50
            novelty_score = 0.94 if positive else 0.66
            out_of_box = bool(paper.get("is_out_of_box")) if paper.get("is_out_of_box") is not None else structural_distance >= 0.75

            idea = _build_idea(
                family=family,
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                hypothesis=hypothesis,
                source="structural_extension",
                priority=priority,
                novelty_score=novelty_score,
                rationale=rationale,
                suggested_template_id=None,
                suggested_config=suggested_config,
                metadata={
                    "is_new_idea": True,
                    "is_branch_repeat": False,
                    "idea_kind": idea_kind,
                    "consumable_by_proposal_flow": True,
                    "paper_source": paper.get("source", ""),
                    "paper_title": title,
                    "positive_result": positive,
                    "techniques": techniques,
                    "web_topic_slug": topic["slug"],
                    "web_search_used": True,
                    "structural_distance": structural_distance,
                    "template_similarity_class": template_similarity_class,
                    "is_structurally_novel": bool(paper.get("is_structurally_novel")) if paper.get("is_structurally_novel") is not None else structural_distance >= 0.60,
                    "is_out_of_box": out_of_box,
                    "is_uncommon_idea": bool(paper.get("is_uncommon_idea")) if paper.get("is_uncommon_idea") is not None else out_of_box,
                    "uncommon_idea_reason": paper.get("uncommon_idea_reason"),
                },
                idea_source="web_search",
                source_idea_ids=[],
                paper_title=title,
                web_search_used=True,
                idea_provider="minimax",
                idea_model="MiniMax-M2.7",
                structural_distance=structural_distance,
                template_similarity_class=template_similarity_class,
                uncommon_idea_reason=str(paper.get("uncommon_idea_reason") or ""),
                is_structurally_novel=bool(paper.get("is_structurally_novel")) if paper.get("is_structurally_novel") is not None else structural_distance >= 0.60,
                is_out_of_box=out_of_box,
                is_uncommon_idea=bool(paper.get("is_uncommon_idea")) if paper.get("is_uncommon_idea") is not None else out_of_box,
            )
            if _candidate_signature(idea) in recent_idea_signatures:
                continue
            ideas.append(idea)
            try:
                save_idea_record(idea, workspace_root=workspace_root)
            except Exception:
                pass

        used: list[str] = state.get("used_topic_slugs", [])
        if topic["slug"] not in used:
            used = used + [topic["slug"]]
        all_slugs = {t["slug"] for t in _WEB_RESEARCH_TOPICS}
        if all_slugs.issubset(set(used)):
            used = []

        state.update({
            "last_attempt_at": datetime.now(UTC).isoformat(),
            "last_success_at": datetime.now(UTC).isoformat(),
            "last_run_ts": time.time(),
            "last_topic_slug": topic["slug"],
            "used_topic_slugs": used,
            "papers_found": len(papers),
            "session_backoff_until": 0,
            "next_retry_at": _iso_timestamp(time.time() + (_WEB_RESEARCH_COOLDOWN_HOURS * 3600.0)),
            "backoff_state": "cooldown",
            "backoff_reason": "cooldown",
            "session_limit_hit": False,
        })
        results_dir = Path(workspace_root) / "queues" / "web_research"
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            (results_dir / f"results_{topic['slug']}.json").write_text(
                json.dumps({"topic": topic, "search_results": search_results, "papers": papers}, indent=2)
            )
        except Exception:
            pass

    except Exception:
        failure_ts = time.time()
        state.update({
            "last_attempt_at": datetime.now(UTC).isoformat(),
            "last_failure_at": datetime.now(UTC).isoformat(),
            "last_run_ts": failure_ts,
            "session_limit_hit": False,
            "next_retry_at": _iso_timestamp(failure_ts + (_WEB_RESEARCH_ERROR_BACKOFF_HOURS * 3600.0)),
            "backoff_state": "error_backoff",
            "backoff_reason": "error_backoff",
        })

    _save_web_research_state(workspace_root, state)
    _save_web_research_status(workspace_root, _web_research_status(workspace_root))
    return ideas


def _priority_from_context(
    *,
    strategy_type: str,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    has_template: bool,
    viable_count: int,
    idea_yield: dict[str, Any] | None = None,
) -> float:
    priority = 0.55
    priority += _safe_float(scorecard.get("search_priority"), 0.0) * 0.25
    priority += min(max(_safe_float(scorecard.get("viable_rate"), 0.0), 0.0), 1.0) * 0.10
    if idea_yield:
        priority += _safe_float(idea_yield.get("search_priority"), 0.0) * 0.12
        idea_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"
        if idea_state == "promising":
            priority += 0.10
        elif idea_state == "active":
            priority += 0.06
        elif idea_state == "stale":
            priority -= 0.05
        elif idea_state == "retired":
            priority -= 0.12
        priority += _safe_float(idea_yield.get("idea_quality_score"), 0.0) * 0.08
        priority -= _safe_float(idea_yield.get("idea_decay_score"), 0.0) * 0.06
        priority += _safe_float(idea_yield.get("idea_promising_share"), 0.0) * 0.06
        priority -= _safe_float(idea_yield.get("idea_retired_share"), 0.0) * 0.08
    if focus.get("focus") == "refine":
        priority += 0.12
    elif focus.get("focus") == "recover":
        priority += 0.08
    elif focus.get("focus") == "deprioritize":
        priority -= 0.18
    if strategy_type == "ml":
        priority += 0.03
    elif strategy_type == "rl":
        priority -= 0.10
        if viable_count <= 0 and _safe_float(scorecard.get("dead_zone_density"), 0.0) >= 0.50:
            priority -= 0.10
    if has_template:
        priority += 0.05
    return round(max(0.05, min(0.95, priority)), 6)


def _novelty_from_context(
    *,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    source: str,
    strategy_type: str,
    idea_yield: dict[str, Any] | None = None,
) -> float:
    novelty = 0.55
    novelty += _safe_float(scorecard.get("duplicate_saturation"), 0.0) * 0.20
    novelty += _safe_float(scorecard.get("dead_zone_density"), 0.0) * 0.10
    if idea_yield:
        novelty += _safe_float(idea_yield.get("idea_quality_score"), 0.0) * 0.08
        novelty -= _safe_float(idea_yield.get("idea_decay_score"), 0.0) * 0.06
        idea_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"
        if idea_state == "promising":
            novelty += 0.08
        elif idea_state == "active":
            novelty += 0.04
        elif idea_state == "stale":
            novelty -= 0.04
        elif idea_state == "retired":
            novelty -= 0.08
        novelty += _safe_float(idea_yield.get("idea_fresh_share"), 0.0) * 0.05
    if focus.get("focus") in {"explore", "recover"}:
        novelty += 0.10
    if source in {"coverage_gap", "template_library"}:
        novelty += 0.10
    if strategy_type in {"ml", "rl"}:
        novelty += 0.05
    return round(max(0.05, min(0.95, novelty)), 6)


def _idea_kind(*, source: str, strategy_type: str, focus: dict[str, Any], template_tags: list[str] | None = None) -> str:
    focus_value = str(focus.get("focus") or "explore")
    tags = {str(tag).strip().lower() for tag in (template_tags or []) if str(tag).strip()}
    if source == "coverage_gap":
        return "coverage_gap_probe"
    if source == "structural_extension":
        if "portfolio" in tags:
            return "new_portfolio_exposure_control"
        if "state" in tags:
            return "new_per_stock_state_model"
        if "risk" in tags or "volatility" in tags:
            return "new_risk_overlay_concept"
        return "new_strategy_logic_concept"
    if source == "history_mining":
        return "branch_refinement" if focus_value in {"refine", "recover"} else "historical_rethink"
    if source == "template_library":
        if {"breakout", "volatility", "quality"} & tags:
            return "template_variation"
        if strategy_type == "ml":
            return "ml_architecture_probe"
        if strategy_type == "rl":
            return "rl_policy_probe"
        return "template_variation"
    if source == "cross_family_hybrid":
        return "cross_family_hybrid"
    if source == "model_based":
        return "ml_architecture_probe"
    if source == "policy_learning":
        return "rl_policy_probe"
    if source == "external_seed":
        return "external_seed_probe"
    return "new_idea_probe"


def _novelty_reason(
    *,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    source: str,
    strategy_type: str,
    weak_patterns: dict[str, Any],
    idea_yield: dict[str, Any] | None = None,
) -> str:
    reasons: list[str] = []
    focus_value = str(focus.get("focus") or "explore")
    if focus_value in {"explore", "recover"}:
        reasons.append(f"focus={focus_value}")
    if _safe_float(scorecard.get("duplicate_saturation"), 0.0) >= 0.30:
        reasons.append("duplicate saturation suggests wider template coverage")
    if _safe_float(scorecard.get("dead_zone_density"), 0.0) >= 0.30:
        reasons.append("dead-zone density suggests a new search region")
    if weak_patterns.get("dead_zone_count", 0) > 0:
        reasons.append("history shows dead-zone regions")
    if weak_patterns.get("weak_count", 0) > 0:
        reasons.append("history shows weak or failed regions")
    if source in {"cross_family_hybrid", "model_based", "policy_learning"}:
        reasons.append(f"{strategy_type} rethink probe")
    elif source == "coverage_gap":
        reasons.append("coverage gap in official history")
    elif source == "history_mining":
        reasons.append("refine from the strongest current branch")
    else:
        reasons.append(f"source={source}")
    if idea_yield:
        idea_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"
        if idea_state == "promising":
            reasons.append("idea-yield category has produced viable descendants")
        elif idea_state == "active":
            reasons.append("idea-yield category is active")
        elif idea_state == "stale":
            reasons.append("idea-yield category is stale and should be revisited cautiously")
        elif idea_state == "retired":
            reasons.append("idea-yield category is retired or heavily downweighted")
        top_kind = _safe_text(idea_yield.get("idea_top_kind"), "")
        if top_kind and top_kind != "unknown":
            reasons.append(f"top idea kind={top_kind}")
    return "; ".join(reasons)


def _structural_idea_kind(category: str) -> str:
    mapping = {
        "portfolio_overlay": "new_portfolio_exposure_control",
        "regime_filter": "new_regime_filter_concept",
        "ranking_rethink": "new_ranking_concept",
        "state_model": "new_per_stock_state_model",
        "position_sizing": "new_position_sizing_concept",
        "cross_family_mechanism": "new_cross_family_borrowing_concept",
        "architecture_family": "new_ml_architecture_family",
        "policy_family": "new_rl_policy_concept",
        "crash_defense": "new_crash_defense_mechanism",
        "breadth_filter": "new_breadth_dispersion_market_state_concept",
        "literature_transfer": "literature_inspired_structural_template",
    }
    return mapping.get(category, "new_strategy_logic_concept")


def _structural_distance_for(category: str) -> float:
    mapping = {
        "portfolio_overlay": 0.90,
        "regime_filter": 0.82,
        "ranking_rethink": 0.74,
        "state_model": 0.86,
        "position_sizing": 0.78,
        "cross_family_mechanism": 0.72,
        "architecture_family": 0.92,
        "policy_family": 0.88,
        "crash_defense": 0.84,
        "breadth_filter": 0.80,
        "literature_transfer": 0.76,
    }
    return mapping.get(category, 0.70)


def _structural_similarity_class(category: str) -> str:
    return STRUCTURAL_SIMILARITY_CLASSES.get(category, "structural_extension")


def _structural_idea_blueprints(
    *,
    family: str,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    weak_patterns: dict[str, Any],
    idea_yield: dict[str, Any],
) -> list[dict[str, Any]]:
    family = family.strip().lower()
    focus_value = str(focus.get("focus") or "explore")
    dead_zone_density = _safe_float(scorecard.get("dead_zone_density"), 0.0)
    duplicate_saturation = _safe_float(scorecard.get("duplicate_saturation"), 0.0)
    yield_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"
    base_hint = "Focus on a mechanism shift, not a parameter tweak."

    common: list[dict[str, Any]] = [
        {
            "category": "portfolio_overlay",
            "title": f"{family.title()} portfolio-level realized-vol overlay",
            "hypothesis": f"Add a portfolio-level realized-vol scaling overlay to {family} so exposure expands and contracts with market-state dispersion rather than only tweaking per-stock filters.",
            "rationale": f"{base_hint} This changes the portfolio's gross exposure logic and can behave differently from local threshold tuning.",
            "template_similarity_class": "portfolio_overlay",
            "uncommon_idea_reason": "Adds a new portfolio-level control layer rather than tuning the entry rules.",
        },
        {
            "category": "regime_filter",
            "title": f"{family.title()} breadth/dispersion regime filter",
            "hypothesis": f"Gate {family} entries with a breadth-and-dispersion market-state filter so the strategy only trades when the cross-sectional tape is constructive.",
            "rationale": f"{base_hint} This introduces a new market-state concept instead of a tighter or looser threshold.",
            "template_similarity_class": "regime_filter",
            "uncommon_idea_reason": "Uses breadth and dispersion as a separate gating mechanism.",
        },
        {
            "category": "state_model",
            "title": f"{family.title()} per-stock state model",
            "hypothesis": f"Track a per-stock state model for {family} that distinguishes breakout continuation, drift decay, and crash-risk states before ranking and sizing.",
            "rationale": f"{base_hint} This changes the representation of each stock and creates a different decision surface from a flat rank list.",
            "template_similarity_class": "state_model",
            "uncommon_idea_reason": "Introduces stock-level state instead of static ranking alone.",
        },
        {
            "category": "ranking_rethink",
            "title": f"{family.title()} residual-strength ranking",
            "hypothesis": f"Rank {family} candidates by residual or context-adjusted strength rather than the raw signal alone, so the selection rule changes structurally.",
            "rationale": f"{base_hint} The ranking target itself is different, not just the cutoff.",
            "template_similarity_class": "ranking_rethink",
            "uncommon_idea_reason": "Changes the ranking target instead of a familiar threshold.",
        },
        {
            "category": "position_sizing",
            "title": f"{family.title()} volatility-aware sizing envelope",
            "hypothesis": f"Use a position-sizing envelope for {family} that scales size by rank gap, realized volatility, and concentration pressure instead of equal-weight or fixed slots.",
            "rationale": f"{base_hint} This changes how winners are expressed at the portfolio layer.",
            "template_similarity_class": "position_sizing",
            "uncommon_idea_reason": "Adds a novel sizing policy rather than changing signal generation.",
        },
    ]
    if family == "superstock":
        common.extend(
            [
                {
                    "category": "crash_defense",
                    "title": "Superstock crash-defense overlay",
                    "hypothesis": "Add a crash-defense overlay that de-risks Superstock when breadth collapses, downside volatility spikes, or high-beta winners stop confirming.",
                    "rationale": f"{base_hint} This creates a separate defense layer that can be switched on independently of the breakout logic.",
                    "template_similarity_class": "crash_defense",
                    "uncommon_idea_reason": "Makes defense a first-class mechanism, not a parameter adjustment.",
                },
                {
                    "category": "breadth_filter",
                    "title": "Superstock breadth-and-late-stage filter",
                    "hypothesis": "Condition Superstock on a breadth-and-late-stage market filter that distinguishes healthy expansion from narrow, late-cycle breakouts.",
                    "rationale": f"{base_hint} It adds a market-state classifier rather than a tighter breakout threshold.",
                    "template_similarity_class": "breadth_filter",
                    "uncommon_idea_reason": "Uses breadth structure to alter participation.",
                },
                {
                    "category": "cross_family_mechanism",
                    "title": "Superstock momentum-style follow-through logic",
                    "hypothesis": "Borrow momentum-style follow-through logic for Superstock so breakout confirmation is measured by post-entry persistence rather than only pre-entry setup quality.",
                    "rationale": f"{base_hint} This borrows the mechanism, not just the parameter values, from another family.",
                    "template_similarity_class": "cross_family_mechanism",
                    "uncommon_idea_reason": "Transfers a mechanism across families instead of copying thresholds.",
                },
            ]
        )
    elif family == "momentum":
        common.extend(
            [
                {
                    "category": "portfolio_overlay",
                    "title": "Momentum realized-vol scaling overlay",
                    "hypothesis": "Overlay momentum with portfolio-level realized-vol scaling that changes gross exposure when market dispersion or crowding rises.",
                    "rationale": f"{base_hint} This alters portfolio behavior more than a tighter stop or shorter lookback.",
                    "template_similarity_class": "portfolio_overlay",
                    "uncommon_idea_reason": "Turns volatility control into a portfolio overlay.",
                },
                {
                    "category": "crash_defense",
                    "title": "Momentum crash-defense regime switch",
                    "hypothesis": "Add a crash-defense regime switch that explicitly steps out of momentum when breadth weakens, vol-of-vol rises, or dispersion becomes unstable.",
                    "rationale": f"{base_hint} This is a mechanism change in market-state handling rather than a filter tweak.",
                    "template_similarity_class": "crash_defense",
                    "uncommon_idea_reason": "Adds a separate crash-defense mechanism.",
                },
                {
                    "category": "cross_family_mechanism",
                    "title": "Momentum breakout-style persistence gate",
                    "hypothesis": "Borrow breakout-style persistence confirmation for momentum so the strategy requires post-entry persistence rather than only formation-period ranking.",
                    "rationale": f"{base_hint} It transfers a behavior model from Superstock into momentum.",
                    "template_similarity_class": "cross_family_mechanism",
                    "uncommon_idea_reason": "Binds momentum to a different execution mechanism.",
                },
            ]
        )
    elif family == "ml_ranker":
        common.extend(
            [
                {
                    "category": "architecture_family",
                    "title": "ML ranker sequence model family",
                    "hypothesis": "Swap the ML ranker from static tabular features to a sequence-aware architecture family that models state transitions across windows.",
                    "rationale": f"{base_hint} This is a new architecture family rather than a model-size tweak.",
                    "template_similarity_class": "architecture_family",
                    "uncommon_idea_reason": "Introduces a different model family and inductive bias.",
                },
                {
                    "category": "state_model",
                    "title": "ML ranker uncertainty-aware state model",
                    "hypothesis": "Add an uncertainty-aware state model so the ranker learns when to abstain, downweight, or hold steady instead of always issuing a full ranking.",
                    "rationale": f"{base_hint} The model learns a new state representation and output policy.",
                    "template_similarity_class": "state_model",
                    "uncommon_idea_reason": "Changes the output regime, not just the feature set.",
                },
            ]
        )
    elif family == "rl_bandit":
        common.extend(
            [
                {
                    "category": "policy_family",
                    "title": "RL bandit regime-conditioned policy family",
                    "hypothesis": "Replace the bandit with a regime-conditioned policy family that switches arms by market state instead of a fixed exploration rule.",
                    "rationale": f"{base_hint} This is a policy redesign rather than a reward tweak.",
                    "template_similarity_class": "policy_family",
                    "uncommon_idea_reason": "Introduces a different policy family and state-action map.",
                },
                {
                    "category": "portfolio_overlay",
                    "title": "RL bandit portfolio exposure controller",
                    "hypothesis": "Let the RL layer control portfolio exposure and not just arm choice, so the policy allocates capital through a distinct control surface.",
                    "rationale": f"{base_hint} This extends RL into exposure management instead of arm selection only.",
                    "template_similarity_class": "portfolio_overlay",
                    "uncommon_idea_reason": "Moves RL from selection to exposure control.",
                },
            ]
        )

    enriched: list[dict[str, Any]] = []
    for idx, item in enumerate(common):
        distance = _structural_distance_for(item["category"])
        if dead_zone_density >= 0.40:
            distance = min(0.95, distance + 0.03)
        if duplicate_saturation >= 0.30:
            distance = min(0.95, distance + 0.02)
        if focus_value in {"explore", "recover"}:
            distance = min(0.95, distance + 0.02)
        if yield_state == "promising":
            distance = min(0.95, distance + 0.02)
        is_out_of_box = distance >= 0.75 and item["category"] not in {"cross_family_mechanism"}
        enriched.append(
            {
                **item,
                "source": "structural_extension",
                "idea_source": "structural_extension",
                "idea_kind": _structural_idea_kind(item["category"]),
                "structural_distance": round(distance, 3),
                "template_similarity_class": _structural_similarity_class(item["category"]),
                "is_structurally_novel": distance >= 0.60,
                "is_out_of_box": is_out_of_box,
                "is_uncommon_idea": is_out_of_box or distance >= 0.70,
            }
        )
    return enriched


def _runtime_cost(strategy_type: str, family: str) -> tuple[str, str]:
    if strategy_type == "ml":
        return "medium_cpu", "CPU-light sklearn-style ranking experiments; avoid GPU-oriented stacks."
    if strategy_type == "rl":
        return "medium_high_cpu", "RL is limited to lightweight bandit-style policies on the 8-core CPU VM."
    if family == "superstock":
        return "medium_cpu", "Classical breakout screens over the daily stock universe."
    return "low_medium_cpu", "Classical parameter/template sweep using the existing daily backtester."


def _build_idea(
    *,
    family: str,
    strategy_type: str,
    hypothesis: str,
    source: str,
    priority: float,
    novelty_score: float,
    rationale: str,
    suggested_template_id: str | None,
    suggested_config: dict[str, Any] | None,
    metadata: dict[str, Any],
    idea_source: str | None = None,
    source_idea_ids: list[str] | None = None,
    paper_title: str | None = None,
    web_search_used: bool = False,
    idea_provider: str | None = None,
    idea_model: str | None = None,
    structural_distance: float | None = None,
    template_similarity_class: str | None = None,
    uncommon_idea_reason: str | None = None,
    is_structurally_novel: bool | None = None,
    is_out_of_box: bool | None = None,
    is_uncommon_idea: bool | None = None,
) -> IdeaRecord:
    timestamp = utc_now_iso()
    estimated_cost, runtime_reason = _runtime_cost(strategy_type, family)
    safe_template = (suggested_template_id or source).replace(" ", "_")
    idea_kind = str(metadata.get("idea_kind") or _idea_kind(source=source, strategy_type=strategy_type, focus=metadata.get("focus") or {}, template_tags=metadata.get("tags") or []))
    novelty_reason = str(
        metadata.get("novelty_reason")
        or _novelty_reason(
            focus=metadata.get("focus") or {},
            scorecard=metadata.get("scorecard") or {},
            source=source,
            strategy_type=strategy_type,
            weak_patterns=metadata.get("weak_patterns") or {},
            idea_yield=metadata.get("idea_yield") or {},
        )
    )
    is_new_idea = bool(metadata.get("is_new_idea")) if metadata.get("is_new_idea") is not None else source in {
        "coverage_gap",
        "template_library",
        "cross_family_hybrid",
        "model_based",
        "policy_learning",
        "external_seed",
        "structural_extension",
    }
    is_branch_repeat = bool(metadata.get("is_branch_repeat")) if metadata.get("is_branch_repeat") is not None else source in {"history_mining"}
    structural_distance_value = metadata.get("structural_distance")
    if structural_distance_value is None:
        structural_distance_value = structural_distance
    template_similarity_class_value = metadata.get("template_similarity_class") or template_similarity_class
    uncommon_idea_reason_value = metadata.get("uncommon_idea_reason") or uncommon_idea_reason
    structural_distance_float = None if structural_distance_value is None else _safe_float(structural_distance_value, 0.0)
    is_structurally_novel_value = bool(metadata.get("is_structurally_novel")) if metadata.get("is_structurally_novel") is not None else bool(
        is_structurally_novel if is_structurally_novel is not None else structural_distance_float is not None and structural_distance_float >= 0.60
    )
    is_out_of_box_value = bool(metadata.get("is_out_of_box")) if metadata.get("is_out_of_box") is not None else bool(
        is_out_of_box if is_out_of_box is not None else structural_distance_float is not None and structural_distance_float >= 0.75
    )
    is_uncommon_idea_value = bool(metadata.get("is_uncommon_idea")) if metadata.get("is_uncommon_idea") is not None else bool(
        is_uncommon_idea if is_uncommon_idea is not None else is_out_of_box_value or is_structurally_novel_value
    )
    return IdeaRecord(
        idea_id=f"idea_{family}_{safe_template}_{timestamp.replace(':', '').replace('-', '')}",
        family=family,
        strategy_type=strategy_type,
        hypothesis=hypothesis,
        source=source,
        priority=priority,
        estimated_cost=estimated_cost,
        estimated_runtime_cost=estimated_cost,
        runtime_cost_reason=runtime_reason,
        timestamp_utc=timestamp,
        novelty_score=novelty_score,
        idea_kind=idea_kind,
        novelty_reason=novelty_reason,
        is_new_idea=is_new_idea,
        is_branch_repeat=is_branch_repeat,
        rationale=rationale,
        suggested_template_id=suggested_template_id,
        template_id=suggested_template_id,
        suggested_config=suggested_config,
        metadata=metadata,
        idea_source=idea_source or source,
        source_idea_ids=list(source_idea_ids or []),
        paper_title=paper_title,
        web_search_used=web_search_used,
        idea_provider=idea_provider,
        idea_model=idea_model,
        is_structurally_novel=is_structurally_novel_value,
        is_out_of_box=is_out_of_box_value,
        structural_distance=structural_distance_float,
        template_similarity_class=template_similarity_class_value,
        uncommon_idea_reason=uncommon_idea_reason_value,
        is_uncommon_idea=is_uncommon_idea_value,
    )


def _template_ideas(
    *,
    family: str,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    weak_patterns: dict[str, Any],
    viable_count: int,
    context: dict[str, Any],
    idea_yield: dict[str, Any] | None = None,
    recent_template_usage: Counter | None = None,
    recent_idea_signatures: Counter[str] | None = None,
) -> list[IdeaRecord]:
    templates = list_idea_templates(family)
    if not templates:
        return []
    dead_zone_density = _safe_float(scorecard.get("dead_zone_density"), 0.0)
    duplicate_saturation = _safe_float(scorecard.get("duplicate_saturation"), 0.0)
    focus_value = str(focus.get("focus") or "explore")
    idea_yield = idea_yield or {}
    recent_usage = recent_template_usage or Counter()
    recent_idea_signatures = recent_idea_signatures or Counter()
    idea_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"
    top_kinds = {
        str((entry or {}).get("idea_kind") or "").strip()
        for entry in (idea_yield.get("top") or [])
        if isinstance(entry, dict) and str((entry or {}).get("idea_kind") or "").strip()
    }

    scored_templates: list[tuple[float, Any, str]] = []
    for template in templates:
        score = 0.0
        tags = set(template.tags)
        template_idea_kind = _idea_kind(
            source=template.source_type,
            strategy_type=template.strategy_type,
            focus=focus,
            template_tags=list(template.tags),
        )
        if focus_value == "refine" and "baseline" in tags:
            score += 0.20
        if focus_value in {"explore", "recover"} and template.source_type in {"template_expansion", "cross_family_hybrid", "model_based", "policy_learning"}:
            score += 0.15
        if dead_zone_density >= 0.50 and tags.intersection({"risk-control", "volatility", "quality"}):
            score += 0.20
        if duplicate_saturation >= 0.50 and template.source_type in {"cross_family_hybrid", "model_based", "policy_learning"}:
            score += 0.15
        if template_idea_kind in top_kinds:
            score += 0.18
        if idea_state == "promising":
            score += 0.10
        elif idea_state == "active":
            score += 0.05
        elif idea_state == "stale":
            score -= 0.05
        elif idea_state == "retired":
            score -= 0.10
        if template.strategy_type == "rl" and family == "rl_bandit":
            if viable_count <= 0 and dead_zone_density >= 0.50:
                score -= 0.35
            else:
                score += 0.05
        if template.strategy_type == "ml":
            score += 0.08
        # Diversity penalty: exponential decay for recently-used templates.
        # Each recent use cuts the score by half; 3+ uses virtually deprioritises the template.
        recent_use_count = recent_usage.get(template.template_id, 0)
        if recent_use_count > 0:
            score *= 0.5 ** recent_use_count
        repeat_signature = _idea_signature_from_payload(
            {
                "family": family,
                "idea_source": template.source_type,
                "source": template.source_type,
                "idea_kind": template_idea_kind,
                "template_id": template.template_id,
            }
        )
        recent_repeat_count = recent_idea_signatures.get(repeat_signature, 0)
        if recent_repeat_count > 0:
            score *= 0.30 ** min(recent_repeat_count, 3)
        scored_templates.append((score, template, repeat_signature))
    scored_templates.sort(key=lambda item: (-item[0], item[1].template_id))
    fresh_templates = [item for item in scored_templates if recent_idea_signatures.get(item[2], 0) == 0]
    selected_templates = fresh_templates if fresh_templates else scored_templates

    ideas: list[IdeaRecord] = []
    for template_score, template, _signature in selected_templates[:MAX_NEARBY_TEMPLATE_IDEAS]:
        priority = _priority_from_context(
            strategy_type=template.strategy_type,
            focus=focus,
            scorecard=scorecard,
            has_template=True,
            viable_count=viable_count,
            idea_yield=idea_yield,
        )
        if template.strategy_type == "rl" and priority < 0.25 and family != "rl_bandit":
            continue
        novelty = _novelty_from_context(
            focus=focus,
            scorecard=scorecard,
            source="template_library",
            strategy_type=template.strategy_type,
            idea_yield=idea_yield,
        )
        structural_distance = 0.25 if template.source_type == "template_expansion" else 0.42
        if "regime" in template.tags:
            structural_distance += 0.05
        if "breakout" in template.tags:
            structural_distance += 0.05
        if template.source_type == "cross_family_hybrid":
            structural_distance += 0.08
        if recent_idea_signatures.get(_signature, 0) > 0:
            structural_distance -= 0.05
        template_similarity_class = "nearby_template"
        ideas.append(
            _build_idea(
                family=family,
                strategy_type=template.strategy_type,
                hypothesis=template.hypothesis,
                source=template.source_type,
                priority=priority,
                novelty_score=novelty,
                rationale=(
                    f"{template.reason_selected} Focus={focus_value}; "
                    f"dead_zone_density={dead_zone_density}; duplicate_saturation={duplicate_saturation}."
                ),
                suggested_template_id=template.template_id,
                suggested_config=template.config,
                metadata={
                    "tags": list(template.tags),
                    "source_family": template.source_family,
                    "template_score": round(template_score, 6),
                    "focus": focus_value,
                    "scorecard": scorecard,
                    "weak_patterns": weak_patterns,
                    "idea_yield": idea_yield,
                    "idea_yield_state": idea_state,
                    "idea_yield_quality_score": _safe_float(idea_yield.get("idea_quality_score"), 0.0),
                    "idea_yield_decay_score": _safe_float(idea_yield.get("idea_decay_score"), 1.0),
                    "idea_yield_fresh_share": _safe_float(idea_yield.get("idea_fresh_share"), 0.0),
                    "idea_yield_promising_share": _safe_float(idea_yield.get("idea_promising_share"), 0.0),
                    "idea_yield_retired_share": _safe_float(idea_yield.get("idea_retired_share"), 0.0),
                    "idea_yield_top_kind": _safe_text(idea_yield.get("idea_top_kind"), None),
                    "idea_kind": "template_variation",
                    "novelty_reason": _novelty_reason(
                        focus=focus,
                        scorecard=scorecard,
                        source=template.source_type,
                        strategy_type=template.strategy_type,
                        weak_patterns=weak_patterns,
                    ),
                    "structural_distance": round(max(0.05, structural_distance), 3),
                    "template_similarity_class": template_similarity_class,
                    "is_structurally_novel": False,
                    "is_out_of_box": False,
                    "is_uncommon_idea": False,
                    "is_new_idea": template.source_type != "history_mining",
                    "is_branch_repeat": template.source_type == "history_mining",
                    "analysis_report_id": context.get("latest_report_id"),
                    "score_summary_report_id": context.get("latest_score_summary_report_id"),
                    "idea_yield_report_id": context.get("idea_yield_report_id"),
                    "consumable_by_proposal_flow": True,
                },
                structural_distance=round(max(0.05, structural_distance), 3),
                template_similarity_class=template_similarity_class,
                is_structurally_novel=False,
                is_out_of_box=False,
                is_uncommon_idea=False,
            )
        )
    return ideas


def _structural_ideas(
    *,
    family: str,
    focus: dict[str, Any],
    scorecard: dict[str, Any],
    weak_patterns: dict[str, Any],
    context: dict[str, Any],
    idea_yield: dict[str, Any] | None = None,
    recent_idea_signatures: Counter[str] | None = None,
) -> list[IdeaRecord]:
    idea_yield = idea_yield or {}
    recent_idea_signatures = recent_idea_signatures or Counter()
    blueprints = _structural_idea_blueprints(
        family=family,
        focus=focus,
        scorecard=scorecard,
        weak_patterns=weak_patterns,
        idea_yield=idea_yield,
    )
    if not blueprints:
        return []

    focus_value = str(focus.get("focus") or "explore")
    duplicate_saturation = _safe_float(scorecard.get("duplicate_saturation"), 0.0)
    dead_zone_density = _safe_float(scorecard.get("dead_zone_density"), 0.0)
    idea_state = str(idea_yield.get("idea_state") or "untested").strip().lower() or "untested"

    ideas: list[IdeaRecord] = []
    for rank, bp in enumerate(blueprints[:MAX_STRUCTURAL_IDEAS]):
        blueprint_template_id = f"structural_{family}_{bp['category']}"
        signature = _idea_signature_from_payload(
            {
                "family": family,
                "idea_source": "structural_extension",
                "source": "structural_extension",
                "idea_kind": bp["idea_kind"],
                "template_id": blueprint_template_id,
                "paper_title": bp["title"],
            }
        )
        if recent_idea_signatures.get(signature, 0) > 0:
            continue
        distance = float(bp["structural_distance"])
        priority = 0.72 + min(0.12, distance * 0.10)
        if focus_value in {"explore", "recover"}:
            priority += 0.03
        if dead_zone_density >= 0.40:
            priority += 0.03
        if duplicate_saturation >= 0.30:
            priority += 0.02
        if idea_state in {"promising", "active"}:
            priority += 0.02
        novelty = 0.84 + min(0.10, distance * 0.08)
        rationale = (
            f"{bp['rationale']} Focus={focus_value}; dead_zone_density={dead_zone_density}; "
            f"duplicate_saturation={duplicate_saturation}; weak_patterns={weak_patterns}."
        )
        ideas.append(
            _build_idea(
                family=family,
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                hypothesis=bp["hypothesis"],
                source=bp["source"],
                priority=round(min(0.97, priority), 6),
                novelty_score=round(min(0.97, novelty), 6),
                rationale=rationale,
                suggested_template_id=blueprint_template_id,
                suggested_config={
                    "structural_blueprint": bp["title"],
                    "structural_category": bp["category"],
                },
                metadata={
                    "focus": focus,
                    "scorecard": scorecard,
                    "weak_patterns": weak_patterns,
                    "idea_yield": idea_yield,
                    "idea_kind": bp["idea_kind"],
                    "novelty_reason": f"Structural extension: {bp['uncommon_idea_reason']}; source=structural_extension",
                    "structural_distance": bp["structural_distance"],
                    "template_similarity_class": bp["template_similarity_class"],
                    "is_structurally_novel": bp["is_structurally_novel"],
                    "is_out_of_box": bp["is_out_of_box"],
                    "is_uncommon_idea": bp["is_uncommon_idea"],
                    "uncommon_idea_reason": bp["uncommon_idea_reason"],
                    "is_new_idea": True,
                    "is_branch_repeat": False,
                    "analysis_report_id": context.get("latest_report_id"),
                    "score_summary_report_id": context.get("latest_score_summary_report_id"),
                    "idea_yield_report_id": context.get("idea_yield_report_id"),
                    "consumable_by_proposal_flow": True,
                },
                idea_source="structural_extension",
                source_idea_ids=[],
                paper_title=bp["title"],
                structural_distance=bp["structural_distance"],
                template_similarity_class=bp["template_similarity_class"],
                uncommon_idea_reason=bp["uncommon_idea_reason"],
                is_structurally_novel=bp["is_structurally_novel"],
                is_out_of_box=bp["is_out_of_box"],
                is_uncommon_idea=bp["is_uncommon_idea"],
            )
        )
    return ideas


def _idea_sort_key(record: IdeaRecord) -> tuple[Any, ...]:
    category_order = {
        "out_of_box": 0,
        "structural_novel": 1,
        "other": 2,
        "nearby_template": 3,
        "confirmation_reproduce": 4,
        "history_mining": 5,
    }
    category = _candidate_category(record)
    return (
        category_order.get(category, 9),
        0 if record.web_search_used else 1,
        0 if record.is_structurally_novel else 1,
        -float(record.structural_distance or 0.0),
        -float(record.priority or 0.0),
        -float(record.novelty_score or 0.0),
        record.family or "",
        record.idea_id,
    )


def _select_ideas(candidates: list[IdeaRecord], limit: int) -> list[IdeaRecord]:
    if not candidates or limit <= 0:
        return []
    ordered = sorted(candidates, key=_idea_sort_key)
    selected: list[IdeaRecord] = []
    counts: Counter[str] = Counter()
    out_of_box_target = max(1, int(round(limit * OUT_OF_BOX_QUOTA_RATIO)))
    for record in ordered:
        category = _candidate_category(record)
        if category == "nearby_template" and counts[category] >= MAX_NEARBY_TEMPLATE_IDEAS:
            continue
        if category == "history_mining" and counts[category] >= MAX_HISTORY_MINING_IDEAS:
            continue
        if category == "confirmation_reproduce" and counts[category] >= MAX_CONFIRMATION_REPRODUCE_IDEAS:
            continue
        if len(selected) >= limit:
            break
        selected.append(record)
        counts[category] += 1

    if len(selected) < limit:
        for record in ordered:
            if record in selected:
                continue
            category = _candidate_category(record)
            if category == "nearby_template" and counts[category] >= MAX_NEARBY_TEMPLATE_IDEAS:
                continue
            if category == "history_mining" and counts[category] >= MAX_HISTORY_MINING_IDEAS:
                continue
            if category == "confirmation_reproduce" and counts[category] >= MAX_CONFIRMATION_REPRODUCE_IDEAS:
                continue
            selected.append(record)
            counts[category] += 1
            if len(selected) >= limit:
                break

    out_of_box_count = sum(1 for record in selected if _candidate_category(record) == "out_of_box")
    if out_of_box_count < out_of_box_target:
        replacements: list[IdeaRecord] = []
        for record in ordered:
            if _candidate_category(record) == "out_of_box" and record not in selected:
                replacements.append(record)
        if replacements:
            new_selected = selected[:]
            replace_idx = len(new_selected) - 1
            for record in replacements:
                while replace_idx >= 0 and _candidate_category(new_selected[replace_idx]) == "out_of_box":
                    replace_idx -= 1
                if replace_idx < 0:
                    break
                new_selected[replace_idx] = record
                replace_idx -= 1
                out_of_box_count += 1
                if out_of_box_count >= out_of_box_target:
                    break
            selected = new_selected

    deduped: list[IdeaRecord] = []
    seen: set[str] = set()
    for record in selected:
        signature = _candidate_signature(record)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(record)
    return deduped[:limit]


def generate_idea_records(
    *,
    workspace_root: str = ".",
    experiments_dir: str = "experiments",
    families: list[str] | None = None,
    limit: int = 4,
) -> list[IdeaRecord]:
    ensure_helper_dirs(workspace_root)
    index = load_results_index(experiments_dir)
    ranked = top_results_per_family(limit=3, base_dir=experiments_dir)
    selected_families = families or list_searchable_families()
    context = _analysis_context(workspace_root, experiments_dir)
    score_summary = context.get("score_summary") or {}
    recent_idea_signatures = _load_recent_idea_signatures(workspace_root, window=40)

    ideas: list[IdeaRecord] = []
    for family in selected_families:
        family_rows = _family_rows(index, family)
        viable_count = int(family_rows["viable"].map(_truthy).sum()) if not family_rows.empty and "viable" in family_rows else 0
        best_rows = ranked.get(family)
        best = best_rows.iloc[0] if best_rows is not None and not best_rows.empty else None
        focus = _focus_for_family(context, family)
        scorecard = score_summary.get(family) or {}
        family_idea_yield = _idea_yield_for_family(context, family)
        weak_patterns = _weak_patterns(family_rows)

        if best is not None:
            hypothesis = (
                f"Refine or broaden from top {family} result {best['experiment_id']} while preserving "
                "duplicate blocking and baseline-aware evaluation."
            )
            rationale = (
                f"{family} has {viable_count} viable result(s). Best observed objective={best.get('objective_score')}; "
                f"focus={focus.get('focus', 'unknown')}; weak_patterns={weak_patterns}."
            )
            priority = _priority_from_context(
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                focus=focus,
                scorecard=scorecard,
                has_template=False,
                viable_count=viable_count,
                idea_yield=family_idea_yield,
            )
            novelty_score = _novelty_from_context(
                focus=focus,
                scorecard=scorecard,
                source="history_mining",
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                idea_yield=family_idea_yield,
            )
            suggested_template_id = best.get("template_id") or None
            suggested_config = {"parent_config_hash": best.get("config_hash")}
            source = "history_mining"
            idea_kind = "branch_refinement" if focus.get("focus") in {"refine", "recover"} else "historical_rethink"
            template_similarity_class = "history_anchor"
            structural_distance = 0.18 if focus.get("focus") in {"refine", "recover"} else 0.24
            is_out_of_box = False
        else:
            hypothesis = f"Open a broader exploratory sweep for {family} because current evidence is sparse or absent."
            rationale = f"No ranked {family} runs are available yet in the official experiment store."
            priority = _priority_from_context(
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                focus=focus,
                scorecard=scorecard,
                has_template=False,
                viable_count=viable_count,
                idea_yield=family_idea_yield,
            )
            novelty_score = _novelty_from_context(
                focus=focus,
                scorecard=scorecard,
                source="coverage_gap",
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                idea_yield=family_idea_yield,
            )
            suggested_template_id = None
            suggested_config = None
            source = "coverage_gap"
            idea_kind = "coverage_gap_probe"
            template_similarity_class = "coverage_gap"
            structural_distance = 0.28
            is_out_of_box = False

        ideas.append(
            _build_idea(
                family=family,
                strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                hypothesis=hypothesis,
                source=source,
                priority=priority,
                novelty_score=novelty_score,
                rationale=rationale,
                suggested_template_id=suggested_template_id,
                suggested_config=suggested_config,
                metadata={
                    "viable_count": viable_count,
                    "official_results_seen": int(len(family_rows)),
                    "best_experiment_id": None if best is None else best.get("experiment_id"),
                    "best_objective_score": None if best is None else best.get("objective_score"),
                    "focus": focus,
                    "scorecard": scorecard,
                    "weak_patterns": weak_patterns,
                    "idea_yield": family_idea_yield,
                    "idea_kind": idea_kind,
                    "novelty_reason": _novelty_reason(
                        focus=focus,
                        scorecard=scorecard,
                        source=source,
                        strategy_type=FAMILY_STRATEGY_TYPES.get(family, "classical"),
                        weak_patterns=weak_patterns,
                        idea_yield=family_idea_yield,
                    ),
                    "is_new_idea": best is None or focus.get("focus") in {"explore", "recover"},
                    "is_branch_repeat": best is not None,
                    "structural_distance": structural_distance,
                    "template_similarity_class": template_similarity_class,
                    "is_structurally_novel": False,
                    "is_out_of_box": is_out_of_box,
                    "is_uncommon_idea": False,
                    "analysis_report_id": context.get("latest_report_id"),
                    "score_summary_report_id": context.get("latest_score_summary_report_id"),
                    "idea_yield_report_id": context.get("idea_yield_report_id"),
                    "idea_yield_state": family_idea_yield.get("idea_state"),
                    "idea_yield_quality_score": family_idea_yield.get("idea_quality_score"),
                    "idea_yield_decay_score": family_idea_yield.get("idea_decay_score"),
                    "consumable_by_proposal_flow": True,
                },
                structural_distance=structural_distance,
                template_similarity_class=template_similarity_class,
                is_structurally_novel=False,
                is_out_of_box=is_out_of_box,
                is_uncommon_idea=False,
            )
        )
        recent_template_usage = _load_recent_template_usage(index, family=family, window=40)
        ideas.extend(
            _template_ideas(
                family=family,
                focus=focus,
                scorecard=scorecard,
                weak_patterns=weak_patterns,
                viable_count=viable_count,
                context=context,
                idea_yield=family_idea_yield,
                recent_template_usage=recent_template_usage,
                recent_idea_signatures=recent_idea_signatures,
            )
        )
        ideas.extend(
            _structural_ideas(
                family=family,
                focus=focus,
                scorecard=scorecard,
                weak_patterns=weak_patterns,
                context=context,
                idea_yield=family_idea_yield,
                recent_idea_signatures=recent_idea_signatures,
            )
        )
    # Augment with web-research seeds (arxiv, rate-limited).
    # We pass up to `limit` additional web ideas so they compete in the final sort.
    web_ideas = _web_research_ideas(
        workspace_root,
        families=selected_families,
        max_ideas=min(limit, 4),
        recent_idea_signatures=recent_idea_signatures,
    )
    ideas.extend(web_ideas)

    selected = _select_ideas(ideas, limit)
    selected.sort(key=_idea_sort_key)
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    families = _families(args.family)
    records = generate_idea_records(
        workspace_root=args.workspace_root,
        experiments_dir=args.experiments_dir,
        families=families,
        limit=args.limit,
    )
    for record in records:
        path = save_idea_record(record, workspace_root=args.workspace_root)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

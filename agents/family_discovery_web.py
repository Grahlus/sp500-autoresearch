"""Bounded web research for family discovery.

Uses arXiv API to find recent academic/practitioner ideas, then MiniMax to:
  1. Plan diverse search queries (rotating from recently used themes)
  2. Synthesize findings into an actionable research context string

All web research is strictly bounded:
  - max_queries per run (default 5)
  - max_results per query (default 3)
  - graceful fallback on any network error
  - rate-limited (1.5s between requests)

Output shape:
  {
    "mode": "arxiv" | "hybrid" | "none",
    "queries_used": [...],
    "themes_used": [...],
    "n_results": N,
    "sources": [...url strings...],
    "synthesis": "...string...",
    "findings": [{query, theme, results:[{title,summary,url}]}]
  }
"""
from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_REQUEST_DELAY = 1.5  # seconds between arXiv calls (polite crawling)
_ARXIV_BASE = "https://export.arxiv.org/api/query"

# Fallback query bank used when LLM query planning fails
_FALLBACK_QUERIES = [
    {"query": "cross-sectional equity signals beyond price momentum", "theme": "cross_sectional"},
    {"query": "intraday OHLC range institutional trading patterns", "theme": "intraday_microstructure"},
    {"query": "volume dryup continuation reversal equity", "theme": "volume_dynamics"},
    {"query": "market breadth regime switching timing equity", "theme": "breadth_regime"},
    {"query": "turn-of-month calendar seasonality equity systematic", "theme": "seasonality"},
    {"query": "volatility dispersion cross-sectional equity alpha", "theme": "vol_dispersion"},
    {"query": "price range position multi-year equity returns", "theme": "range_position"},
    {"query": "sector rotation equity signals price-based", "theme": "sector_rotation"},
    {"query": "short-term reversal equity weekly signals", "theme": "short_reversal"},
    {"query": "structural breaks regime detection equity strategy", "theme": "regime_detection"},
]


# ---------------------------------------------------------------------------
# arXiv search
# ---------------------------------------------------------------------------

def search_arxiv(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """Fetch papers from arXiv API.

    Searches all fields for the query + equity trading context.
    Returns list of {title, summary, url}.
    """
    import urllib.parse
    try:
        import requests as _req
    except ImportError:
        return []

    augmented = f"{query} systematic equity"
    encoded = urllib.parse.quote_plus(augmented)
    url = f"{_ARXIV_BASE}?search_query=all:{encoded}&max_results={max_results}&sortBy=relevance"
    try:
        resp = _req.get(url, timeout=12, headers={"User-Agent": "SP500-FamilyDiscoveryAgent/1.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        results: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", _ARXIV_NS):
            title = (entry.findtext("atom:title", "", _ARXIV_NS) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", _ARXIV_NS) or "").strip().replace("\n", " ")[:280]
            link = ""
            for lnk in entry.findall("atom:link", _ARXIV_NS):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            if not link:
                id_elem = entry.find("atom:id", _ARXIV_NS)
                link = id_elem.text.strip() if id_elem is not None else ""
            if title:
                results.append({"title": title, "summary": summary, "url": link})
        return results
    except Exception:
        return []


# ---------------------------------------------------------------------------
# LLM-driven query planning
# ---------------------------------------------------------------------------

def plan_search_queries(
    *,
    existing_families: str,
    themes_to_avoid: list[str],
    n_queries: int = 6,
) -> list[dict[str, str]]:
    """Ask MiniMax to plan diverse search query angles for this run.

    Rotates away from themes_to_avoid. Returns list of {query, theme, angle}.
    """
    from agents.family_discovery_agent import _call_minimax, _parse_json_array

    avoid_str = ", ".join(themes_to_avoid[:10]) if themes_to_avoid else "none"
    prompt = f"""\
You are planning diverse academic/practitioner search queries to find NOVEL equity strategy ideas.

=== EXISTING STRATEGY FAMILIES (already implemented) ===
{existing_families}

=== RECENTLY SEARCHED THEMES — AVOID THESE THIS RUN ===
{avoid_str}

=== YOUR TASK ===
Generate exactly {n_queries} diverse search queries. Each must cover a DIFFERENT angle.
Rotate away from recently searched themes above.

Possible angles (pick different ones each run):
- Intraday OHLC range patterns and institutional order flow
- Volume accumulation/distribution dynamics
- Cross-sectional return/vol dispersion at regime transitions
- Market breadth and breath thrust signals
- Calendar, seasonal, and sequential patterns
- Multi-year price range positioning effects
- Sector leadership rotation measured by price
- Volatility state transitions cross-sectionally
- Short-term reversal vs continuation conditions
- Microstructure proxies from daily OHLCV
- Earnings/event drift proxies without earnings data
- Risk-off / risk-on equity screening
- Portfolio-level turnover and concentration effects
- "What earns alpha when momentum crashes"

Return a JSON array of exactly {n_queries} objects:
[
  {{
    "query": "5-10 word arxiv search query",
    "theme": "one_word_theme_slug",
    "angle": "one sentence what signal/mechanism this targets"
  }}
]
Return ONLY the JSON array. No other text."""

    system = (
        "You are a quantitative research strategist. "
        "Return only valid JSON arrays. Be creative and varied."
    )
    text = _call_minimax(system, prompt, max_tokens=2048, temperature=0.85)
    queries = _parse_json_array(text)

    if not queries or len(queries) < 3:
        # fallback: use built-in list, avoiding recently used themes
        avoid_set = set(t.lower() for t in themes_to_avoid)
        queries = [q for q in _FALLBACK_QUERIES if q["theme"].lower() not in avoid_set][:n_queries]
        if not queries:
            queries = _FALLBACK_QUERIES[:n_queries]

    return [q for q in queries if q.get("query")][:n_queries]


# ---------------------------------------------------------------------------
# Bounded web research runner
# ---------------------------------------------------------------------------

def run_bounded_web_research(
    queries: list[dict[str, str]],
    *,
    max_queries: int = 5,
    max_results_per_query: int = 3,
) -> dict[str, Any]:
    """Run arXiv searches for each planned query. Bounded and rate-limited."""
    all_findings: list[dict[str, Any]] = []
    queries_used: list[str] = []
    themes_used: list[str] = []
    all_sources: list[str] = []

    print(f"[fd_web] running {min(len(queries), max_queries)} arXiv queries ...", flush=True)

    for q_data in queries[:max_queries]:
        query = (q_data.get("query") or "").strip()
        theme = (q_data.get("theme") or "unknown").strip()
        if not query:
            continue

        papers = search_arxiv(query, max_results=max_results_per_query)
        n = len(papers)
        print(f"[fd_web]   {theme}: '{query}' → {n} results", flush=True)

        if papers:
            all_findings.append({
                "query": query,
                "theme": theme,
                "angle": q_data.get("angle", ""),
                "results": papers,
            })
            all_sources.extend(p["url"] for p in papers if p.get("url"))

        queries_used.append(query)
        themes_used.append(theme)
        time.sleep(_REQUEST_DELAY)

    n_results = sum(len(f["results"]) for f in all_findings)
    print(f"[fd_web] total: {len(queries_used)} queries, {n_results} papers found", flush=True)

    return {
        "mode": "arxiv",
        "findings": all_findings,
        "queries_used": queries_used,
        "themes_used": themes_used,
        "n_results": n_results,
        "sources": all_sources,
    }


# ---------------------------------------------------------------------------
# Synthesis: MiniMax turns raw paper list into actionable research context
# ---------------------------------------------------------------------------

def synthesize_web_findings(
    findings: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
) -> str:
    """Synthesize web research findings into an actionable context string.

    Returns a concise paragraph (150-300 words) of mechanism-level insights.
    Returns empty string on failure.
    """
    from agents.family_discovery_agent import _call_minimax, _strip_think_blocks

    if not findings:
        return ""

    findings_text = ""
    for f in findings:
        findings_text += f"\n[Theme: {f['theme']}] Query: \"{f['query']}\"\n"
        for r in f.get("results", [])[:3]:
            findings_text += f"  Title: {r['title']}\n"
            findings_text += f"  Abstract: {r['summary'][:220]}\n"

    prompt = f"""\
You are extracting ACTIONABLE strategy ideas from recent quantitative finance research.

=== RESEARCH FINDINGS ===
{findings_text}

=== CONSTRAINTS ===
- Only daily OHLCV price data, VIX, and CNN Fear&Greed index are available
- No tick data, no options, no earnings, no fundamentals
- Long-only US large-cap equities (~841 tickers)
- Transaction costs: $20/trade + 5 bps slippage

=== YOUR TASK ===
Extract 3-5 CONCRETE MECHANISM INSPIRATIONS for new equity strategy families.
For each insight:
- What specific signal can be constructed from OHLCV/VIX/FG?
- Why would it generate alpha orthogonal to price momentum?
- What market condition makes it work?

Write a focused paragraph of 150-250 words total. Be SPECIFIC about data and signal mechanics.
Do NOT just describe the paper — extract the MECHANISM and how it applies to our system."""

    system = (
        "You are a quantitative research synthesizer. "
        "Be specific, mechanism-focused, and grounded in available data."
    )
    text = _call_minimax(system, prompt, max_tokens=max_tokens, temperature=0.25)
    return (text or "").strip()[:1200]

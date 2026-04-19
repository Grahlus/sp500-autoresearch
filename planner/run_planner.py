"""CLI entry point for the Layer 2 periodic planner.

Usage:
    uv run python -m planner.run_planner [--base-dir experiments] [--window-hours 2]
    uv run python -m planner.run_planner --refresh-ideas  # also invoke idea agent first

Steps:
  1. (Optional) Refresh idea queue by running the idea agent
  2. Build planning context from live experiment state (includes queued ideas)
  3. Call MiniMax to produce a ResearchPlan
  4. Validate the plan; back-fill source_idea_ids from queued ideas if needed
  5. Archive the old plan (if any)
  6. Save the new plan as plans/current_plan.json

Exits 0 on success, 1 on any failure.  The executor tolerates a missing plan
file, so a failed run is never fatal to the research loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("planner.run_planner")


def _refresh_idea_queue(workspace_root: str = ".", timeout: int = 120) -> None:
    """Invoke the idea agent to refresh queued ideas. Non-fatal on failure."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "agents.idea_agent",
             "--workspace-root", workspace_root],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("Idea agent refreshed successfully")
        else:
            logger.warning("Idea agent exited %d: %s", result.returncode, result.stderr[:300])
    except subprocess.TimeoutExpired:
        logger.warning("Idea agent timed out after %ds", timeout)
    except Exception as exc:
        logger.warning("Idea agent failed: %s", exc)


def _backfill_source_idea_ids(plan, queued_ideas: list) -> "ResearchPlan":
    """If the planner left source_idea_ids empty, populate from queued ideas."""
    from dataclasses import replace
    if plan.source_idea_ids:
        return plan
    ids = [str(idea["idea_id"]) for idea in queued_ideas if idea.get("idea_id")]
    if not ids:
        return plan
    logger.info("Back-filling source_idea_ids from %d queued ideas", len(ids))
    return replace(plan, source_idea_ids=ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Layer 2 MiniMax planner")
    parser.add_argument("--base-dir", default="experiments",
                        help="Experiment base directory (default: experiments)")
    parser.add_argument("--workspace-root", default=".",
                        help="Workspace root for queues/ (default: .)")
    parser.add_argument("--window-hours", type=float, default=2.0,
                        help="Planning window length in hours (default: 2)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Planner timeout in seconds (default: 120)")
    parser.add_argument("--min-replan-interval-hours", type=float, default=2.0,
                        help="Minimum age before replacing a fresh plan (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build context + call MiniMax but do not save plan")
    parser.add_argument("--refresh-ideas", action="store_true",
                        help="Run idea agent to refresh queued ideas before building context")
    args = parser.parse_args(argv)

    from planner.plan_store import load_current_plan
    existing = load_current_plan()
    if existing.loaded and existing.plan is not None:
        from planner.cadence import should_replan

        cadence = should_replan(
            existing.plan,
            min_replan_interval_hours=args.min_replan_interval_hours,
            now=datetime.now(UTC),
        )
        if not cadence.should_replan:
            logger.info(
                "Current plan %s is fresh (age=%.2fh); skipping replan",
                existing.plan_id,
                cadence.plan_age_hours or 0.0,
            )
            return 0

    # Step 1: (Optional) refresh idea queue
    if args.refresh_ideas:
        logger.info("Refreshing idea queue via idea agent …")
        _refresh_idea_queue(workspace_root=args.workspace_root, timeout=args.timeout)

    # Step 2: Build context (includes queued ideas)
    logger.info("Building planning context from %s …", args.base_dir)
    from planner.context_builder import build_planning_context
    context = build_planning_context(base_dir=args.base_dir, workspace_root=args.workspace_root)
    queued_ideas = context.get("queued_ideas") or []
    logger.info("Context built: %d top-level keys, %d best results, %d families in scorecards, %d queued ideas",
                len(context),
                len(context.get("best_results") or []),
                len(context.get("scorecards") or {}),
                len(queued_ideas))

    # Step 3: Call MiniMax
    logger.info("Calling MiniMax planner (window=%.1fh, timeout=%ds) …",
                args.window_hours, args.timeout)
    from planner.llm_planner import call_minimax_planner
    plan = call_minimax_planner(
        context,
        window_hours=args.window_hours,
        timeout=args.timeout,
    )

    if plan is None:
        logger.error("Planner returned None — no plan saved.")
        return 1

    # Step 4: Back-fill source_idea_ids if needed
    from planner.plan_schema import ResearchPlan  # noqa: F401 (type hint)
    plan = _backfill_source_idea_ids(plan, queued_ideas)
    logger.info("Plan produced: id=%s valid_until=%s source_idea_ids=%s",
                plan.plan_id, plan.valid_until, plan.source_idea_ids)

    if args.dry_run:
        from planner.plan_schema import plan_to_dict
        print(json.dumps(plan_to_dict(plan), indent=2))
        logger.info("Dry-run: plan not saved.")
        return 0

    # Step 5: Archive old plan if present
    from planner.plan_store import archive_plan, save_plan
    if existing.plan is not None:
        dest = archive_plan(existing.plan)
        logger.info("Archived old plan %s → %s", existing.plan_id, dest)

    # Step 6: Save new plan
    save_plan(plan)
    logger.info("Saved plan %s to plans/current_plan.json", plan.plan_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

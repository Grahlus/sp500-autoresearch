#!/usr/bin/env python3
from __future__ import annotations

import argparse

from experiment_best_results import (
    ensure_global_results_index,
    save_best_results_reports,
)
from experiment_dashboard import build_best_results_dashboard, format_dashboard_markdown, save_dashboard_reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show global best experiment results")
    parser.add_argument("--base-dir", default="experiments", help="Experiment store root")
    parser.add_argument("--reports-dir", default="reports", help="Dashboard report output directory")
    parser.add_argument("--overall", type=int, default=20, help="Number of overall rows to print")
    parser.add_argument("--viable", type=int, default=20, help="Number of viable rows to print")
    parser.add_argument("--baseline", type=int, default=20, help="Number of baseline-beating rows to print")
    parser.add_argument("--per-family", type=int, default=10, help="Number of rows to print per family")
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write report files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index_path = ensure_global_results_index(args.base_dir)
    dashboard = build_best_results_dashboard(
        base_dir=args.base_dir,
        overall_limit=args.overall,
        viable_limit=args.viable,
        baseline_limit=args.baseline,
        per_family_limit=args.per_family,
    )

    print(f"global_index={index_path}")
    if not args.no_write:
        legacy_paths = save_best_results_reports(
            base_dir=args.base_dir,
            overall_limit=args.overall,
            per_family_limit=args.per_family,
        )
        dashboard_paths = save_dashboard_reports(dashboard, reports_dir=args.reports_dir)
        print(f"overall_report={legacy_paths['overall_path']}")
        print(f"per_family_report={legacy_paths['per_family_path']}")
        print(f"latest_batch_report={legacy_paths['latest_batch_path']}")
        print(f"dashboard_json={dashboard_paths['best_results_json']}")
        print(f"dashboard_md={dashboard_paths['best_results_md']}")
        print(f"family_scorecards={dashboard_paths['family_scorecards_json']}")

    print()
    print(format_dashboard_markdown(dashboard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import re
import psutil
import time
import threading
import math

def _clean_float(v):
    """Replace NaN/Inf with None for JSON safety."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v

app = FastAPI()

BASE = Path("/home/mrlearn/sp500-autoresearch")
INDEX = BASE / "experiments" / "index.csv"
SCORECARDS = BASE / "reports" / "family_scorecards.json"
LOG_FILE = BASE / "logs" / "autonomous_research.log"
STATIC = BASE / "dashboard" / "static"

STATIC.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ── caching ──────────────────────────────────────────────────────────────────

_df_cache: dict = {"df": None, "ts": 0.0}
_CACHE_TTL = 30  # seconds

def _df(low_memory=False):
    """Read index.csv with 30-second cache to avoid hammering the CSV."""
    now = time.monotonic()
    key = f"{'full' if not low_memory else 'light'}"
    if _df_cache["df"] is not None and (now - _df_cache["ts"]) < _CACHE_TTL:
        cached = _df_cache["df"]
        return cached if not low_memory else cached
    kwargs = {"low_memory": False}
    result = pd.read_csv(INDEX, **kwargs)
    _df_cache["df"] = result
    _df_cache["ts"] = now
    return result


def _find_runner_pid() -> int:
    """Return PID of the running autonomous_runner.py process, or 0.
    Uses pgrep for reliability (finds processes psutil misses in tmux containers),
    with psutil as fallback."""
    import subprocess
    # Primary: pgrep is more reliable across namespace boundaries
    try:
        result = subprocess.run(
            ["pgrep", "-f", "autonomous_runner.py"],
            capture_output=True, text=True, timeout=3
        )
        pids = [int(x) for x in result.stdout.strip().split("\n") if x]
        if pids:
            return pids[0]  # Return the outermost wrapper pid
    except Exception:
        pass
    # Fallback: psutil
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("autonomous_runner.py" in str(c) for c in cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return 0


def _is_research_active() -> tuple[bool, str]:
    """Check if research is active via research_status.txt as well as process check.
    Returns (is_active, state). If the status file was updated recently (within 5 min)
    and shows success/running, consider the system active even if no process is found."""
    status_file = BASE / "run" / "research_status.txt"
    if not status_file.exists():
        return False, "no_status_file"
    try:
        content = status_file.read_text()
        lines = dict(line.strip().split("=", 1) for line in content.strip().split("\n") if "=" in line)
        state = lines.get("state", "unknown")
        # Get the timestamp of the last state change
        last_finished = lines.get("finished_at", lines.get("started_at", ""))
        if last_finished:
            from datetime import datetime, timezone
            try:
                last_time = datetime.fromisoformat(last_finished)
                age_seconds = (datetime.now(timezone.utc) - last_time).total_seconds()
                # Consider active if state changed within last 5 minutes
                if age_seconds < 300 and state in ("running", "success"):
                    return True, state
            except Exception:
                pass
        return state == "running", state
    except Exception:
        return False, "error"


def _safe_json_value(val):
    """Convert numpy/Pandas types to JSON-serializable Python types."""
    import numpy as np
    if val is None or val == "":
        return ""
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return _clean_float(float(val))  # catches NaN/Inf
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, dict, str)):
        return val
    return str(val)


def _runner_start_time(pid: int) -> datetime | None:
    """Return the datetime when the runner process started."""
    if pid == 0:
        return None
    try:
        proc = psutil.Process(pid)
        return datetime.fromtimestamp(proc.create_time(), tz=timezone.utc)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds into human-readable string."""
    if seconds < 0:
        return "—"
    total_minutes = int(seconds // 60)
    if total_minutes < 1:
        return f"{int(seconds)}s"
    if total_minutes < 60:
        return f"{total_minutes}m"
    hours = total_minutes // 60
    mins = total_minutes % 60
    if hours < 24:
        return f"{hours}h {mins}m"
    days = hours // 24
    remaining_hours = hours % 24
    return f"{days}d {remaining_hours}h"


def _proposal_path() -> Path | None:
    """Return the most recent proposal directory."""
    proposals = BASE / "experiments" / "proposals"
    if not proposals.exists():
        return None
    dirs = sorted(proposals.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        summary = d / "summary.json"
        if summary.exists():
            return d
    return dirs[0] if dirs else None


def _read_proposal_info() -> str:
    """Extract hypothesis from the most recent proposal.json."""
    prop = _proposal_path()
    if prop is None:
        return ""
    proposal_file = prop / "proposal.json"
    if proposal_file.exists():
        try:
            data = json.loads(proposal_file.read_text())
            for family, candidates in data.get("candidate_metadata", {}).items():
                if candidates and isinstance(candidates, list):
                    hyp = candidates[0].get("hypothesis", "")
                    if hyp:
                        return hyp
            summary = prop / "summary.json"
            if summary.exists():
                s = json.loads(summary.read_text())
                return s.get("hypothesis", "") or s.get("idea", "") or ""
        except Exception:
            pass
    return ""


def _running_experiment_count() -> int:
    """Count how many experiment processes are currently running."""
    count = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("run_experiment.py" in str(c) or "backtester.py" in str(c) or "run_experiment" in str(c) for c in cmdline):
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return count


def _get_cycle_count() -> int:
    """Return the most recent cycle number from the log file (reads last 2MB only)."""
    try:
        result = subprocess.run(
            ["tail", "-n", "10000", str(LOG_FILE)],
            capture_output=True, text=True, timeout=10
        )
        cycle_nums = []
        for line in result.stdout.splitlines():
            if "cycle=" in line:
                for part in line.split():
                    if part.startswith("cycle="):
                        try:
                            cycle_nums.append(int(part.split("=")[1]))
                        except (ValueError, IndexError):
                            pass
        return max(cycle_nums) if cycle_nums else 0
    except Exception:
        return 0


def _get_last_log_timestamp() -> datetime | None:
    """Return the datetime of the last log entry (reads last 50 lines only)."""
    try:
        result = subprocess.run(
            ["tail", "-n", "50", str(LOG_FILE)],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            m = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
            if m:
                ts_str = m.group().replace(" ", "T")
                if "+" not in ts_str:
                    ts_str += "+00:00"
                elif ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                return datetime.fromisoformat(ts_str)
        return None
    except Exception:
        return None


def _count_errors_in_last_hour() -> int:
    """Count ERROR-level log entries in the last hour (reads last 500 lines only)."""
    try:
        result = subprocess.run(
            ["tail", "-n", "500", str(LOG_FILE)],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.splitlines()
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        count = 0
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            if re.search(r"\] ERROR |\sERROR\s", line) and "DtypeWarning" not in line and "error_message" not in line:
                m = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
                if m:
                    ts_str = m.group().replace(" ", "T") + "+00:00"
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts >= one_hour_ago:
                            count += 1
                    except ValueError:
                        pass
                else:
                    count += 1
        return count
    except Exception:
        return 0


def _get_last_error_line() -> str:
    """Get the most recent ERROR line from the log (reads last 200 lines only)."""
    try:
        result = subprocess.run(
            ["tail", "-n", "200", str(LOG_FILE)],
            capture_output=True, text=True, timeout=10
        )
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            if re.search(r"\] ERROR |\sERROR\s", line) and "DtypeWarning" not in line and "error_message" not in line:
                m = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*\]\s*(.+)", line)
                if m:
                    return m.group(1).strip()[:120]
                return line.strip()[:120]
        return ""
    except Exception:
        return ""


def _count_ideas_tested() -> int:
    """Approximate ideas tested — count rows in index.csv (uses light read)."""
    try:
        result = subprocess.run(
            ["wc", "-l", str(INDEX)],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split()
        return max(0, int(lines[0]) - 1)  # subtract header row
    except Exception:
        return 0


_backlog_cache: dict = {"count": None, "ts": 0.0}
_BACKLOG_TTL = 60  # seconds


def _count_ideas_backlog() -> int:
    """Count ideas in queue that have not yet been turned into experiments.
    
    An idea is "done" if its full idea_id (filename without .json) appears
    in the index.csv source_idea_ids or idea_id columns.
    """
    import glob, os, json as _json
    now = time.monotonic()
    if _backlog_cache["count"] is not None and (now - _backlog_cache["ts"]) < _BACKLOG_TTL:
        return _backlog_cache["count"]

    try:
        # 1. All queued idea IDs (filename without .json)
        queue_files = glob.glob("/home/mrlearn/sp500-autoresearch/queues/ideas/*.json")
        queue_ids = {os.path.splitext(os.path.basename(f))[0] for f in queue_files}

        # 2. All processed idea IDs from index.csv
        processed_ids: set[str] = set()
        try:
            import pandas as pd
            df = pd.read_csv(INDEX, low_memory=True, usecols=["idea_id", "source_idea_ids"])
            for v in df["idea_id"].dropna():
                processed_ids.add(str(v).strip())
            for v in df["source_idea_ids"].dropna():
                try:
                    ids = _json.loads(v)
                    for i in ids:
                        processed_ids.add(str(i).strip())
                except Exception:
                    pass
        except Exception:
            pass

        backlog = len(queue_ids - processed_ids)
        _backlog_cache["count"] = backlog
        _backlog_cache["ts"] = now
        return backlog
    except Exception:
        return 0


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/status")
async def status():
    pid = _find_runner_pid()
    research_active, research_state = _is_research_active()
    running = pid > 0 or research_active

    uptime_seconds = 0.0
    start_time = _runner_start_time(pid)
    if start_time:
        uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

    try:
        df = _df(low_memory=True)
        n_experiments = len(df)
        last_row = df.sort_values("timestamp_utc", ascending=False).iloc[0]
        last_experiment = str(last_row.get("experiment_id", ""))
    except Exception:
        n_experiments = 0
        last_experiment = ""

    current_proposal = _read_proposal_info()

    return {
        "running": running,
        "pid": pid,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_formatted": _format_uptime(uptime_seconds),
        "n_experiments": n_experiments,
        "last_experiment": last_experiment,
        "current_proposal": current_proposal,
    }


@app.get("/api/system-summary")
async def system_summary():
    """Dashboard header metrics: uptime, cycles, ideas, health, processes."""
    pid = _find_runner_pid()
    research_active, research_state = _is_research_active()
    running = pid > 0 or research_active

    uptime_seconds = 0.0
    start_time = _runner_start_time(pid)
    if start_time:
        uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

    # Downtime: if not running, compute time since last log entry
    downtime_minutes = None
    if not running:
        last_ts = _get_last_log_timestamp()
        if last_ts:
            downtime_minutes = int((datetime.now(timezone.utc) - last_ts).total_seconds() / 60)

    # Cycle count from log
    total_cycles = _get_cycle_count()

    # Total ideas tested
    total_ideas = _count_ideas_tested()

    # System health
    error_count = _count_errors_in_last_hour()
    if error_count == 0:
        system_health = "NOMINAL"
    elif error_count <= 2:
        system_health = "DEGRADED"
    else:
        system_health = "ERROR"

    # Active experiment processes
    active_procs = _running_experiment_count()

    # Last error line
    last_error = _get_last_error_line()

    # Last completed experiment from index (light read only)
    last_exp_id = ""
    last_exp_ts = None
    try:
        result = subprocess.run(
            ["tail", "-n", "1", str(INDEX)],
            capture_output=True, text=True, timeout=10
        )
        last_line = result.stdout.strip()
        if last_line:
            cols = last_line.split(",")
            if len(cols) > 0:
                last_exp_id = cols[0].strip('"')
            if len(cols) > 1:
                last_exp_ts = cols[1]
    except Exception:
        pass

    return {
        "running": running,
        "pid": pid,
        "uptime_seconds": round(uptime_seconds, 1),
        "uptime_formatted": _format_uptime(uptime_seconds),
        "downtime_minutes": downtime_minutes,
        "total_cycles": total_cycles,
        "total_ideas_tested": total_ideas,
        "idea_backlog": _count_ideas_backlog(),
        "system_health": system_health,
        "active_python_processes": active_procs,
        "last_error": last_error,
        "last_experiment_id": last_exp_id,
        "last_experiment_ts": last_exp_ts,
        "error_count_last_hour": error_count,
    }


@app.get("/api/activity")
async def activity():
    """System activity: current proposal, parallel processes, time since last experiment."""
    pid = _find_runner_pid()
    research_active, _ = _is_research_active()
    running = pid > 0 or research_active

    # Running experiment count
    n_parallel = _running_experiment_count()

    # Last experiment timestamp
    last_ts = None
    last_exp_id = ""
    try:
        df = _df(low_memory=True)
        df_sorted = df.sort_values("timestamp_utc", ascending=False)
        if len(df_sorted) > 0:
            last_row = df_sorted.iloc[0]
            last_ts = str(last_row.get("timestamp_utc", ""))
            last_exp_id = str(last_row.get("experiment_id", ""))
    except Exception:
        pass

    # Current proposal dir and hypothesis
    prop = _proposal_path()
    proposal_name = prop.name if prop else ""
    proposal_hypothesis = _read_proposal_info()

    # Ideas from latest proposal (just the count for now, full list from /api/current-ideas)
    idea_count = 0
    if prop:
        pf = prop / "proposal.json"
        if pf.exists():
            try:
                data = json.loads(pf.read_text())
                for fam, cands in data.get("candidate_metadata", {}).items():
                    if cands:
                        idea_count += len(cands)
            except Exception:
                pass

    return {
        "running": running,
        "pid": pid,
        "n_parallel": n_parallel,
        "last_experiment_id": last_exp_id,
        "last_experiment_ts": last_ts,
        "proposal_name": proposal_name,
        "proposal_hypothesis": proposal_hypothesis,
        "idea_queue_size": idea_count,
    }


@app.get("/api/discoveries")
async def discoveries():
    """Track champion Sharpe and recent discoveries."""
    try:
        df = _df(low_memory=False)
        df = df[df["sharpe"].notna()]
        df_viable = df[df.get("viable", False) == True]
        if len(df_viable) > 0:
            df = df_viable
        df = df.sort_values("sharpe", ascending=False)

        # Best overall
        best_row = df.iloc[0] if len(df) > 0 else None
        champion = None
        if best_row is not None:
            champion = {
                "experiment_id": best_row.get("experiment_id", ""),
                "strategy_family": best_row.get("strategy_family", ""),
                "sharpe": _safe_json_value(best_row.get("sharpe")),
                "calmar": _safe_json_value(best_row.get("calmar")),
                "total_return": _safe_json_value(best_row.get("total_return")),
                "timestamp_utc": str(best_row.get("timestamp_utc", "")),
            }

        # Recent high-sharpe experiments (last 10)
        recent = df.head(10)
        recent_list = []
        for _, row in recent.iterrows():
            s = float(row.get("sharpe", 0) or 0)
            recent_list.append({
                "experiment_id": row.get("experiment_id", ""),
                "strategy_family": row.get("strategy_family", ""),
                "sharpe": _safe_json_value(row.get("sharpe")),
                "viable": _safe_json_value(row.get("viable")),
                "timestamp_utc": str(row.get("timestamp_utc", "")),
            })

        # Best by family
        best_by_family = {}
        for fam in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            fam_df = df[df.get("strategy_family", "") == fam]
            if len(fam_df) > 0:
                r = fam_df.iloc[0]
                best_by_family[fam] = {
                    "experiment_id": r.get("experiment_id", ""),
                    "sharpe": _safe_json_value(r.get("sharpe")),
                    "calmar": _safe_json_value(r.get("calmar")),
                    "total_return": _safe_json_value(r.get("total_return")),
                    "timestamp_utc": str(r.get("timestamp_utc", "")),
                }

        return {"champion": champion, "recent": recent_list, "best_by_family": best_by_family}
    except Exception as e:
        return {"champion": None, "recent": [], "best_by_family": {}, "error": str(e)}


@app.get("/api/experiments")
async def experiments(
    limit: int = Query(20, ge=1, le=500),
    offset: int = Query(0, ge=0),
    viable_only: bool = Query(False),
    family: str = Query("", description="Filter by strategy family"),
):
    try:
        df = _df(low_memory=False)
        if viable_only:
            df = df[df.get("viable", False) == True]  # noqa: E712
        if family:
            df = df[df.get("strategy_family", "") == family]

        df = df.sort_values("timestamp_utc", ascending=False)
        total = len(df)
        page = df.iloc[offset : offset + limit]

        cols = [
            "experiment_id", "timestamp_utc", "strategy_family", "status",
            "sharpe", "calmar", "total_return", "annual_return",
            "max_drawdown", "viable", "trade_count", "final_value",
            "holdout_check_status", "holdout_check_outcome",
            "novelty_score", "selection_score", "template_id", "hypothesis",
            "source_type", "exploration_mode",
        ]
        available = [c for c in cols if c in page.columns]
        records = page[available].fillna("").to_dict(orient="records")
        return {"total": total, "offset": offset, "limit": limit, "experiments": records}
    except Exception as e:
        return {"total": 0, "offset": offset, "limit": limit, "experiments": [], "error": str(e)}


@app.get("/api/best")
async def best(limit: int = Query(10, ge=1, le=50)):
    try:
        df = _df(low_memory=False)
        df = df[df.get("viable", False) == True]  # noqa: E712
        df = df[df["sharpe"].notna()]
        df = df.sort_values("sharpe", ascending=False).head(limit)

        cols = [
            "experiment_id", "strategy_family", "sharpe", "calmar",
            "total_return", "annual_return", "max_drawdown", "trade_count",
            "final_value", "holdout_check_status", "holdout_check_outcome",
        ]
        available = [c for c in cols if c in df.columns]
        records = df[available].fillna("").to_dict(orient="records")
        return {"best": records}
    except Exception as e:
        return {"best": [], "error": str(e)}


@app.get("/api/best-by-family")
async def best_by_family(limit: int = Query(5, ge=1, le=20)):
    try:
        df = _df(low_memory=False)
        df = df[df["sharpe"].notna()]

        families = ["momentum", "superstock", "ml_ranker", "rl_bandit"]
        result = {}

        cols = [
            "experiment_id", "strategy_family", "sharpe", "calmar",
            "total_return", "annual_return", "max_drawdown", "trade_count",
            "final_value", "viable", "holdout_check_status", "holdout_check_outcome",
        ]

        for fam in families:
            fam_df = df[df["strategy_family"] == fam].sort_values("sharpe", ascending=False).head(limit)
            available = [c for c in cols if c in fam_df.columns]
            result[fam] = fam_df[available].fillna("").to_dict(orient="records")

        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/experiment/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Returns full experiment details including config from spec.json and per-window metrics from result.json."""
    try:
        df = _df(low_memory=False)
        row = df[df["experiment_id"] == experiment_id]

        if row.empty:
            return {"error": "Experiment not found"}, 404

        row = row.iloc[0]

        result = {}
        for col in df.columns:
            val = row[col] if col in row.index and pd.notna(row[col]) else ""
            result[col] = _safe_json_value(val)

        # Read spec.json from runs directory
        spec_path = BASE / "experiments" / "runs" / experiment_id / "spec.json"
        if spec_path.exists():
            try:
                spec = json.loads(spec_path.read_text())
                result["spec"] = {
                    "config": spec.get("config", {}),
                    "hypothesis": spec.get("hypothesis", ""),
                    "dead_zone_flags": spec.get("dead_zone_flags", []),
                    "branch_budgets": spec.get("branch_budgets", []),
                    "dataset_id": spec.get("dataset_id", ""),
                    "data_start": spec.get("data_start", ""),
                    "data_end": spec.get("data_end", ""),
                }
            except Exception:
                result["spec"] = None
        else:
            result["spec"] = None

        # Read result.json for per-window metrics
        result_path = BASE / "experiments" / "runs" / experiment_id / "result.json"
        if result_path.exists():
            try:
                res_data = json.loads(result_path.read_text())
                metrics = res_data.get("metrics", {})
                windows = metrics.get("windows", [])
                result["windows"] = [
                    {
                        "window": w.get("window", ""),
                        "sharpe": w.get("sharpe"),
                        "calmar": w.get("calmar"),
                        "total_ret": w.get("total_ret"),
                        "max_dd": w.get("max_dd"),
                        "bench_ret": w.get("bench_ret"),
                        "trades_yr": w.get("trades_yr"),
                        "n_days": w.get("n_days"),
                    }
                    for w in windows
                ] if windows else []
                result["windows_beat_spy"] = metrics.get("windows_beat_spy", "")
                result["sharpe_max"] = metrics.get("sharpe_max")
                result["sharpe_min"] = metrics.get("sharpe_min")
                result["sharpe_std"] = metrics.get("sharpe_std")
                result["robustness"] = res_data.get("robustness", {})
                result["runtime_seconds"] = res_data.get("runtime_seconds")
                result["status"] = res_data.get("status", result.get("status", ""))
            except Exception:
                result["windows"] = []
                result["windows_beat_spy"] = ""
        else:
            result["windows"] = []
            result["windows_beat_spy"] = ""

        # Lineage columns
        lineage_cols = [
            "parent_config_hash", "descendant_count", "near_duplicate_of",
            "source_idea_ids", "source_proposal_id", "novelty_score",
            "selection_score", "duplicate_risk", "dead_zone_risk",
            "targeted_follow_up_required", "targeted_follow_up_reason",
            "targeted_follow_up_type", "targeted_follow_up_priority",
        ]
        for col in lineage_cols:
            if col in df.columns:
                val = row[col] if col in row.index and pd.notna(row[col]) else ""
                result[col] = _safe_json_value(val)

        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/current-ideas")
async def current_ideas():
    """Returns the current idea queue from the most recent proposal.json candidate_metadata."""
    try:
        prop = _proposal_path()
        if prop is None:
            return {"ideas": [], "error": "No proposals directory"}

        proposal_file = prop / "proposal.json"
        if not proposal_file.exists():
            return {"proposal_dir": prop.name, "ideas": [], "error": "No proposal.json found"}

        data = json.loads(proposal_file.read_text())
        candidate_metadata = data.get("candidate_metadata", {})

        ideas = []
        for family, candidates in candidate_metadata.items():
            if not candidates or not isinstance(candidates, list):
                continue
            for cand in candidates:
                idea_id = ""
                source_ids = cand.get("source_idea_ids", [])
                if source_ids and isinstance(source_ids, list):
                    idea_id = source_ids[0]
                elif cand.get("template_id"):
                    idea_id = f"idea_{family}_{cand.get('template_id')}_{prop.name}"

                ideas.append({
                    "family": family,
                    "idea_id": idea_id,
                    "hypothesis": cand.get("hypothesis", ""),
                    "config_hash": cand.get("config_hash", ""),
                    "template_id": cand.get("template_id", ""),
                    "idea_kind": cand.get("idea_kind", ""),
                    "source_type": cand.get("source_type", ""),
                    "exploration_mode": cand.get("exploration_mode", ""),
                    "novelty_score": _clean_float(cand.get("novelty_score")),
                    "selection_score": _clean_float(cand.get("selection_score")),
                    "objective_proxy": _clean_float(cand.get("objective_proxy")),
                    "dead_zone_risk": _clean_float(cand.get("dead_zone_risk")),
                    "strategy_type": cand.get("strategy_type", ""),
                    "confirmation_state": cand.get("confirmation_state", ""),
                    "holdout_check_status": cand.get("holdout_check_status", ""),
                    "holdout_check_outcome": cand.get("holdout_check_outcome", ""),
                    "region_label": cand.get("region_label", ""),
                })

        return {"proposal_dir": prop.name, "ideas": ideas}
    except Exception as e:
        return {"ideas": [], "error": str(e)}


@app.get("/api/families")
async def families():
    try:
        data = json.loads(SCORECARDS.read_text())
        families_data = data.get("families", {})

        families = {}
        for name in ["momentum", "superstock", "ml_ranker", "rl_bandit"]:
            f = families_data.get(name, {})
            families[name] = {
                "family": name,
                "total_experiments": f.get("total_experiments", 0),
                "viable_rate": f.get("viable_rate", 0.0),
                "idea_viable_rate": f.get("idea_viable_rate", 0.0),
                "idea_state": f.get("idea_state", ""),
                "branch_state": f.get("branch_state", ""),
                "idea_attempt_count": f.get("idea_attempt_count", 0),
                "idea_robust_descendant_count": f.get("idea_robust_descendant_count", 0),
                "best_objective_score": f.get("best_objective_score"),
                "holdout_check_status": f.get("holdout_check_status", ""),
                "holdout_check_outcome": f.get("holdout_check_outcome", ""),
                "search_priority": f.get("search_priority", 0.0),
                "idea_quality_score": f.get("idea_quality_score", 0.0),
                "overfit_risk": f.get("overfit_risk", 0.0),
                "branch_decay_score": f.get("branch_decay_score", 0.0),
                "exploitation_budget_recommendation": f.get("exploitation_budget_recommendation", 0.0),
                "exploration_budget_recommendation": f.get("exploration_budget_recommendation", 0.0),
            }
        return families
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/logs")
async def logs(lines: int = Query(50, ge=1, le=500)):
    try:
        if not LOG_FILE.exists():
            return {"lines": [], "error": "log file not found"}
        content = LOG_FILE.read_text()
        all_lines = content.splitlines()
        tail = all_lines[-lines:]
        return {"lines": tail, "total": len(all_lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@app.get("/api/logs/full")
async def logs_full():
    """Return full log for download."""
    try:
        if not LOG_FILE.exists():
            return {"content": "", "error": "log file not found"}
        content = LOG_FILE.read_text()
        return {"content": content, "size_bytes": len(content)}
    except Exception as e:
        return {"content": "", "error": str(e)}


@app.get("/api/family-discovery")
async def family_discovery_status():
    """Return family discovery pipeline health: scheduler state + candidate queue."""
    try:
        from family_discovery_health import build_health_report
        report = build_health_report(workspace_root=str(Path(__file__).resolve().parents[1]))
        return report
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/holdout")
async def holdout():
    try:
        df = _df(low_memory=False)
        mask = df["holdout_check_status"].isin(["pending", "required"]) | \
               df["holdout_check_outcome"].isin(["pending", "confirmed"])
        df = df[mask].sort_values("timestamp_utc", ascending=False).head(50)

        cols = [
            "experiment_id", "strategy_family", "holdout_check_status",
            "holdout_check_outcome", "holdout_check_type", "sharpe",
            "total_return", "viable",
        ]
        available = [c for c in cols if c in df.columns]
        records = df[available].fillna("").to_dict(orient="records")
        return {"holdout_experiments": records}
    except Exception as e:
        return {"holdout_experiments": [], "error": str(e)}

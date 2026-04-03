#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/sp500-autoresearch"
LOG_DIR="$REPO/logs"
RUN_DIR="$REPO/run"
LOCK_FILE="$RUN_DIR/research.lock"
LOG_FILE="$LOG_DIR/research_loop.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"

cd "$REPO"

exec flock -n "$LOCK_FILE" bash -c '
  while true; do
    echo "[$(date -Is)] ===== cycle start =====" >> "'"$LOG_FILE"'"

    # Replace these with your real commands
    uv run python -m research.propose_next_batch >> "'"$LOG_FILE"'" 2>&1
    uv run python -m research.run_latest_batch >> "'"$LOG_FILE"'" 2>&1

    rc=$?
    echo "[$(date -Is)] cycle exit code: $rc" >> "'"$LOG_FILE"'"
    echo "[$(date -Is)] ===== cycle end =====" >> "'"$LOG_FILE"'"

    sleep 1800
  done
'

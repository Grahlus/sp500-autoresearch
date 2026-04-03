#!/usr/bin/env bash
set -euo pipefail

SESSION="research"
REPO="$HOME/sp500-autoresearch"
CMD="$REPO/scripts/run_research_loop.sh"
LOG_FILE="$REPO/logs/tmux_watchdog.log"

mkdir -p "$REPO/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[$(date -Is)] tmux session already running: $SESSION" >> "$LOG_FILE"
  exit 0
fi

echo "[$(date -Is)] starting tmux session: $SESSION" >> "$LOG_FILE"
tmux new-session -d -s "$SESSION" "cd $REPO && bash $CMD"

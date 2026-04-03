#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-autoresearch}
REPO_DIR=${REPO_DIR:-$HOME/sp500-autoresearch}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/tmux_watchdog.log}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  printf '[%s] tmux session already exists session=%s\n' "$(date -Is)" "$SESSION_NAME" >> "$LOG_FILE"
  exit 0
fi

printf '[%s] starting tmux session session=%s\n' "$(date -Is)" "$SESSION_NAME" >> "$LOG_FILE"
tmux new-session -d -s "$SESSION_NAME" "cd $REPO_DIR && bash $REPO_DIR/research_loop.sh"
printf '[%s] tmux session started session=%s\n' "$(date -Is)" "$SESSION_NAME" >> "$LOG_FILE"

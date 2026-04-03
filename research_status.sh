#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-autoresearch}
REPO_DIR=${REPO_DIR:-$HOME/sp500-autoresearch}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
RUN_DIR=${RUN_DIR:-$REPO_DIR/run}
LOG_FILE=${LOG_FILE:-$LOG_DIR/autonomous_research.log}
LOCK_FILE=${LOCK_FILE:-$RUN_DIR/autonomous_research.lock}
HEARTBEAT_FILE=${HEARTBEAT_FILE:-$RUN_DIR/last_heartbeat.txt}
STATUS_FILE=${STATUS_FILE:-$RUN_DIR/research_status.txt}

printf 'session_name=%s\n' "$SESSION_NAME"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  printf 'tmux_session=running\n'
else
  printf 'tmux_session=stopped\n'
fi

if [ -f "$HEARTBEAT_FILE" ]; then
  printf 'last_heartbeat=%s\n' "$(cat "$HEARTBEAT_FILE")"
else
  printf 'last_heartbeat=missing\n'
fi

if [ -f "$LOCK_FILE" ]; then
  printf 'lock_file=present\n'
else
  printf 'lock_file=missing\n'
fi

printf 'log_file=%s\n' "$LOG_FILE"

if [ -f "$STATUS_FILE" ]; then
  printf 'status_file=%s\n' "$STATUS_FILE"
  cat "$STATUS_FILE"
else
  printf 'status_file=missing\n'
fi

if [ -f "$LOG_FILE" ]; then
  printf 'recent_log:\n'
  tail -n 20 "$LOG_FILE"
else
  printf 'recent_log=missing\n'
fi

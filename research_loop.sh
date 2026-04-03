#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$HOME/sp500-autoresearch}
SESSION_NAME=${SESSION_NAME:-autoresearch}
BASE_DIR=${BASE_DIR:-$REPO_DIR/experiments}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
RUN_DIR=${RUN_DIR:-$REPO_DIR/run}
LOG_FILE=${LOG_FILE:-$LOG_DIR/autonomous_research.log}
LOCK_FILE=${LOCK_FILE:-$RUN_DIR/autonomous_research.lock}
SLEEP_SECONDS=${SLEEP_SECONDS:-1800}
HEARTBEAT_FILE=${HEARTBEAT_FILE:-$RUN_DIR/last_heartbeat.txt}
STATUS_FILE=${STATUS_FILE:-$RUN_DIR/research_status.txt}
FAMILY=${FAMILY:-all}
BATCH_SIZE=${BATCH_SIZE:-10}
SEED=${SEED:-42}
EXPLORATION_FRACTION=${EXPLORATION_FRACTION:-0.30}
EXPLOITATION_FRACTION=${EXPLOITATION_FRACTION:-0.70}
LONG_BREAK_EVERY=${LONG_BREAK_EVERY:-8}
LONG_BREAK_SECONDS=${LONG_BREAK_SECONDS:-21600}
ERROR_SLEEP_SECONDS=${ERROR_SLEEP_SECONDS:-3600}

mkdir -p "$LOG_DIR" "$RUN_DIR" "$BASE_DIR"
cd "$REPO_DIR"

if [ -f "$HOME/.cargo/env" ]; then
  # Ensure uv is available in non-interactive shells started by tmux/cron.
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

run_cycle() {
  local started_at rc finished_at
  started_at=$(date -Is)
  printf '%s\n' "$started_at" > "$HEARTBEAT_FILE"
  printf 'session=%s\nstate=running\nstarted_at=%s\nfamily=%s\nbatch_size=%s\nseed=%s\n' \
    "$SESSION_NAME" "$started_at" "$FAMILY" "$BATCH_SIZE" "$SEED" > "$STATUS_FILE"
  printf '[%s] cycle start session=%s family=%s batch_size=%s seed=%s\n' \
    "$started_at" "$SESSION_NAME" "$FAMILY" "$BATCH_SIZE" "$SEED" >> "$LOG_FILE"

  set +e
  uv run python autonomous_runner.py \
    --family "$FAMILY" \
    --proposal-next \
    --run-proposal \
    --n "$BATCH_SIZE" \
    --seed "$SEED" \
    --base-dir "$BASE_DIR" \
    --exploration-fraction "$EXPLORATION_FRACTION" \
    --exploitation-fraction "$EXPLOITATION_FRACTION" \
    >> "$LOG_FILE" 2>&1
  rc=$?
  set -e

  finished_at=$(date -Is)
  printf '%s\n' "$finished_at" > "$HEARTBEAT_FILE"

  if [ "$rc" -eq 0 ]; then
    printf 'session=%s\nstate=success\nfinished_at=%s\nfamily=%s\nbatch_size=%s\nseed=%s\nlast_rc=%s\n' \
      "$SESSION_NAME" "$finished_at" "$FAMILY" "$BATCH_SIZE" "$SEED" "$rc" > "$STATUS_FILE"
    printf '[%s] cycle status=success rc=%s\n' "$finished_at" "$rc" >> "$LOG_FILE"
  else
    printf 'session=%s\nstate=error\nfinished_at=%s\nfamily=%s\nbatch_size=%s\nseed=%s\nlast_rc=%s\n' \
      "$SESSION_NAME" "$finished_at" "$FAMILY" "$BATCH_SIZE" "$SEED" "$rc" > "$STATUS_FILE"
    printf '[%s] cycle status=error rc=%s\n' "$finished_at" "$rc" >> "$LOG_FILE"
  fi

  return "$rc"
}

export SESSION_NAME BASE_DIR LOG_DIR RUN_DIR LOG_FILE LOCK_FILE SLEEP_SECONDS
export LONG_BREAK_EVERY LONG_BREAK_SECONDS ERROR_SLEEP_SECONDS
export HEARTBEAT_FILE STATUS_FILE FAMILY BATCH_SIZE SEED
export EXPLORATION_FRACTION EXPLOITATION_FRACTION
export -f run_cycle

exec flock -n "$LOCK_FILE" bash -lc '
  cycle=0

  while true; do
    cycle=$((cycle + 1))

    if run_cycle; then
      sleep_for="$SLEEP_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=normal\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    else
      sleep_for="$ERROR_SLEEP_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=error_backoff\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    fi

    if [ "$LONG_BREAK_EVERY" -gt 0 ] && [ $((cycle % LONG_BREAK_EVERY)) -eq 0 ]; then
      sleep_for="$LONG_BREAK_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=long_break\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    fi

    sleep "$sleep_for"
  done
' --

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-$HOME/sp500-autoresearch}
SESSION_NAME=${SESSION_NAME:-autoresearch}
BASE_DIR=${BASE_DIR:-$REPO_DIR/experiments}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
RUN_DIR=${RUN_DIR:-$REPO_DIR/run}
LOG_FILE=${LOG_FILE:-$LOG_DIR/autonomous_research.log}
LOCK_FILE=${LOCK_FILE:-$RUN_DIR/autonomous_research.lock}
SLEEP_SECONDS=${SLEEP_SECONDS:-30}
HEARTBEAT_FILE=${HEARTBEAT_FILE:-$RUN_DIR/last_heartbeat.txt}
STATUS_FILE=${STATUS_FILE:-$RUN_DIR/research_status.txt}
FAMILY=${FAMILY:-all}
BATCH_SIZE=${BATCH_SIZE:-24}
MAX_WORKERS=${MAX_WORKERS:-6}
SEED=${SEED:-42}
EXPLORATION_FRACTION=${EXPLORATION_FRACTION:-0.30}
EXPLOITATION_FRACTION=${EXPLOITATION_FRACTION:-0.70}
LONG_BREAK_EVERY=${LONG_BREAK_EVERY:-0}
LONG_BREAK_SECONDS=${LONG_BREAK_SECONDS:-0}
ERROR_SLEEP_SECONDS=${ERROR_SLEEP_SECONDS:-60}
MIN_FREE_GB=${MIN_FREE_GB:-5}
DISK_CHECK_PATH=${DISK_CHECK_PATH:-$REPO_DIR}
MAINTENANCE_ENABLED=${MAINTENANCE_ENABLED:-1}
MAINTENANCE_KEEP_RECENT_PROPOSALS=${MAINTENANCE_KEEP_RECENT_PROPOSALS:-200}
MAINTENANCE_KEEP_RECENT_BATCHES=${MAINTENANCE_KEEP_RECENT_BATCHES:-0}
MAINTENANCE_MIN_SUMMARY_BYTES=${MAINTENANCE_MIN_SUMMARY_BYTES:-25000000}
LOG_MAX_BYTES=${LOG_MAX_BYTES:-50000000}
LOG_RETAIN_COUNT=${LOG_RETAIN_COUNT:-4}
# Layer 2 planner: 0 = disabled, >0 = seconds between planner runs (default 1h)
PLANNER_INTERVAL_SECONDS=${PLANNER_INTERVAL_SECONDS:-3600}
PLANNER_WINDOW_HOURS=${PLANNER_WINDOW_HOURS:-2}
PLANNER_MIN_REPLAN_INTERVAL_HOURS=${PLANNER_MIN_REPLAN_INTERVAL_HOURS:-2}
PLANNER_TIMESTAMP_FILE=${PLANNER_TIMESTAMP_FILE:-$RUN_DIR/last_planner_run.txt}
PLANNER_REFRESH_IDEAS=${PLANNER_REFRESH_IDEAS:-0}

mkdir -p "$LOG_DIR" "$RUN_DIR" "$BASE_DIR"
cd "$REPO_DIR"

if [ -f "$HOME/.cargo/env" ]; then
  # Ensure uv is available in non-interactive shells started by tmux/cron.
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

free_kb() {
  df -Pk "$DISK_CHECK_PATH" | awk 'NR == 2 {print $4}'
}

run_storage_maintenance() {
  [ "$MAINTENANCE_ENABLED" = "1" ] || return 0
  local reason=${1:-scheduled}
  printf '[%s] storage_maintenance start reason=%s\n' "$(date -Is)" "$reason" >> "$LOG_FILE"
  if uv run python storage_maintenance.py \
      --base-dir "$BASE_DIR" \
      --log-dir "$LOG_DIR" \
      --keep-recent-proposals "$MAINTENANCE_KEEP_RECENT_PROPOSALS" \
      --keep-recent-batches "$MAINTENANCE_KEEP_RECENT_BATCHES" \
      --min-summary-bytes "$MAINTENANCE_MIN_SUMMARY_BYTES" \
      --log-max-bytes "$LOG_MAX_BYTES" \
      --log-retain-count "$LOG_RETAIN_COUNT" \
      >> "$LOG_FILE" 2>&1; then
    printf '[%s] storage_maintenance success reason=%s\n' "$(date -Is)" "$reason" >> "$LOG_FILE"
  else
    printf '[%s] storage_maintenance failed reason=%s\n' "$(date -Is)" "$reason" >> "$LOG_FILE"
    return 1
  fi
}

disk_guard() {
  local min_free_kb available_kb
  min_free_kb=$((MIN_FREE_GB * 1024 * 1024))
  available_kb=$(free_kb)
  if [ "$available_kb" -ge "$min_free_kb" ]; then
    return 0
  fi
  printf '[%s] disk_guard low_space available_kb=%s min_free_kb=%s action=maintenance\n' \
    "$(date -Is)" "$available_kb" "$min_free_kb" >> "$LOG_FILE"
  run_storage_maintenance "disk_low" || true
  available_kb=$(free_kb)
  if [ "$available_kb" -ge "$min_free_kb" ]; then
    printf '[%s] disk_guard recovered available_kb=%s min_free_kb=%s\n' \
      "$(date -Is)" "$available_kb" "$min_free_kb" >> "$LOG_FILE"
    return 0
  fi
  printf 'session=%s\nstate=paused_low_disk\navailable_kb=%s\nmin_free_kb=%s\nchecked_at=%s\n' \
    "$SESSION_NAME" "$available_kb" "$min_free_kb" "$(date -Is)" > "$STATUS_FILE"
  printf '[%s] disk_guard paused_low_disk available_kb=%s min_free_kb=%s\n' \
    "$(date -Is)" "$available_kb" "$min_free_kb" >> "$LOG_FILE"
  return 1
}

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
    --max-workers "$MAX_WORKERS" \
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

run_planner_if_due() {
  # Run the Layer 2 planner if PLANNER_INTERVAL_SECONDS > 0 and interval has elapsed.
  # Non-fatal: a planner failure never stops the executor loop.
  [ "$PLANNER_INTERVAL_SECONDS" -gt 0 ] || return 0
  local now_ts last_ts elapsed
  now_ts=$(date +%s)
  last_ts=0
  if [ -f "$PLANNER_TIMESTAMP_FILE" ]; then
    last_ts=$(cat "$PLANNER_TIMESTAMP_FILE" 2>/dev/null || echo 0)
  fi
  elapsed=$((now_ts - last_ts))
  if [ "$elapsed" -ge "$PLANNER_INTERVAL_SECONDS" ]; then
    local planner_args=""
    [ "$PLANNER_REFRESH_IDEAS" = "1" ] && planner_args="--refresh-ideas"
    printf '[%s] planner_run starting (interval=%ds elapsed=%ds)\n' \
      "$(date -Is)" "$PLANNER_INTERVAL_SECONDS" "$elapsed" >> "$LOG_FILE"
    if uv run python -m planner.run_planner \
         --base-dir "$BASE_DIR" \
         --window-hours "$PLANNER_WINDOW_HOURS" \
         --min-replan-interval-hours "$PLANNER_MIN_REPLAN_INTERVAL_HOURS" \
         $planner_args >> "$LOG_FILE" 2>&1; then
      printf '[%s] planner_run success\n' "$(date -Is)" >> "$LOG_FILE"
    else
      printf '[%s] planner_run failed rc=$? (non-fatal, executor continues)\n' \
        "$(date -Is)" >> "$LOG_FILE"
    fi
    echo "$now_ts" > "$PLANNER_TIMESTAMP_FILE"
  fi
}

export SESSION_NAME BASE_DIR LOG_DIR RUN_DIR LOG_FILE LOCK_FILE SLEEP_SECONDS
export LONG_BREAK_EVERY LONG_BREAK_SECONDS ERROR_SLEEP_SECONDS
export MIN_FREE_GB DISK_CHECK_PATH MAINTENANCE_ENABLED MAINTENANCE_KEEP_RECENT_PROPOSALS
export MAINTENANCE_KEEP_RECENT_BATCHES MAINTENANCE_MIN_SUMMARY_BYTES LOG_MAX_BYTES LOG_RETAIN_COUNT
export PLANNER_INTERVAL_SECONDS PLANNER_TIMESTAMP_FILE PLANNER_REFRESH_IDEAS
export PLANNER_WINDOW_HOURS PLANNER_MIN_REPLAN_INTERVAL_HOURS
export HEARTBEAT_FILE STATUS_FILE FAMILY BATCH_SIZE SEED MAX_WORKERS
export EXPLORATION_FRACTION EXPLOITATION_FRACTION
export -f run_cycle run_planner_if_due free_kb run_storage_maintenance disk_guard

exec flock -n "$LOCK_FILE" bash -lc '
  cycle=0

  while true; do
    cycle=$((cycle + 1))

    if ! disk_guard; then
      sleep_for="$ERROR_SLEEP_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=paused_low_disk\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
      sleep "$sleep_for"
      continue
    fi

    # Layer 2 planner (runs at most once every PLANNER_INTERVAL_SECONDS)
    run_planner_if_due

    if run_cycle; then
      sleep_for="$SLEEP_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=normal\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    else
      sleep_for="$ERROR_SLEEP_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=error_backoff\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    fi

    run_storage_maintenance "post_cycle" || true

    if [ "$LONG_BREAK_EVERY" -gt 0 ] && [ $((cycle % LONG_BREAK_EVERY)) -eq 0 ]; then
      sleep_for="$LONG_BREAK_SECONDS"
      printf "[%s] cycle=%s next_sleep=%s reason=long_break\n" \
        "$(date -Is)" "$cycle" "$sleep_for" >> "$LOG_FILE"
    fi

    sleep "$sleep_for"
  done
' --

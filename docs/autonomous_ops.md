# Autonomous Research Ops

The unattended autonomous research path uses:

- `tmux` as the long-lived process container
- `cron` only as a watchdog that ensures the tmux session exists
- `flock` to prevent overlapping worker loops

## Manual Start

```bash
bash ./ensure_research_tmux.sh
```

This creates a detached tmux session named `autoresearch` by default.

## Attach To The Running Session

```bash
tmux attach -t autoresearch
```

To use a different session name:

```bash
SESSION_NAME=myresearch bash ./ensure_research_tmux.sh
tmux attach -t myresearch
```

## Detach From Tmux

Press:

```text
Ctrl-b d
```

## Stop The Unattended Loop

```bash
tmux kill-session -t autoresearch
```

The lock is held by the loop process, so stopping the tmux session stops the worker.

## Check Status

```bash
bash ./research_status.sh
```

This shows:

- tmux session state
- last heartbeat
- whether the lock file exists
- the main log path
- the latest lines from the main log

## Resume After Disconnect Or Reboot

Run the watchdog/launcher again:

```bash
bash ./ensure_research_tmux.sh
```

It is idempotent. If the tmux session already exists, it does nothing.

## Cron Watchdog

Example watchdog entry:

```cron
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin:/home/mrlearn/.local/bin:/home/mrlearn/.cargo/bin
MAILTO=""

*/10 * * * * /home/mrlearn/sp500-autoresearch/ensure_research_tmux.sh >> /home/mrlearn/sp500-autoresearch/logs/cron_watchdog.log 2>&1
```

This does not run experiments directly. It only ensures the tmux session exists.

## Main Runtime Files

- Main loop: `research_loop.sh`
- Tmux watchdog: `ensure_research_tmux.sh`
- Status command: `research_status.sh`
- Main log: `logs/autonomous_research.log`
- Watchdog log: `logs/tmux_watchdog.log`
- Cron log: `logs/cron_watchdog.log`
- Runtime lock: `run/autonomous_research.lock`
- Heartbeat: `run/last_heartbeat.txt`
- Status snapshot: `run/research_status.txt`

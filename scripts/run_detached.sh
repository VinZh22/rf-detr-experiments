#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Run a command FULLY DETACHED from the current shell / Claude Code session.
#
# The job is started in its own session via `setsid`, so it is reparented to
# init (PID 1) and is NOT in the caller's process group. It therefore keeps
# running after you close Claude Code / disconnect the terminal — ideal for
# long overnight training runs.
#
# Usage:
#   scripts/run_detached.sh <logfile> <command> [args...]
#
# Environment variables are inherited, so prefix them as usual:
#   CUDA_VISIBLE_DEVICES=4 scripts/run_detached.sh /tmp/run.log \
#       python scripts/train_rfdetr_dataset.py --epochs 40 ...
#
# stdout+stderr go to <logfile>; the PID is written to <logfile>.pid.
# Monitor:   tail -f <logfile>
# Stop:      kill "$(cat <logfile>.pid)"
# ------------------------------------------------------------------------
set -euo pipefail

log="${1:?usage: run_detached.sh <logfile> <command> [args...]}"
shift
[ "$#" -ge 1 ] || { echo "error: no command given" >&2; exit 2; }
mkdir -p "$(dirname "$log")"

# setsid -> new session (survives parent death); bash -c 'exec ...' keeps the PID
# equal to the launched command's PID so the .pid file is directly killable.
setsid bash -c 'exec "$@"' _ "$@" >"$log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$log.pid"
echo "detached: pid=$pid  log=$log"
echo "  monitor: tail -f $log        stop: kill $pid"

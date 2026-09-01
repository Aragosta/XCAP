#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/user/XCAP/.venv-fly/bin/python
wait_for() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }
for t in b c g; do
  wait_for "bcg_sweep[.]py"
  [ "$t" = b ] && continue
  echo "[queue] starting test $t $(date -u +%T)"
  for s in 0 1 2 3; do
    nohup $PY scripts/bcg_sweep.py --test $t --shard $s --n-shards 4 --threads 1 \
      > results/test_${t}_shard$s.log 2>&1 &
  done
  sleep 90
done
wait_for "bcg_sweep[.]py"
echo "[queue] B, C, G all done $(date -u +%T)"

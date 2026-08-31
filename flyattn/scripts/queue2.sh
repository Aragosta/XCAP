#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/user/XCAP/.venv-fly/bin/python
wait_for() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }
wait_for "sweep[.]py --grid t4"
echo "[queue] starting T1b $(date -u +%T)"
for s in 0 1 2 3; do
  nohup $PY scripts/t1b_finetune.py --shard $s --n-shards 4 --steps 400 \
        --threads 1 > results/t1b_shard$s.log 2>&1 &
done
sleep 60; wait_for "t1b_finetune[.]py"
echo "[queue] starting T5 $(date -u +%T)"
nohup $PY scripts/t5_curvature_surgery.py --threads 4 --steps 1200 > results/t5.log 2>&1 &
sleep 60; wait_for "t5_curvature_surgery[.]py"
echo "[queue] all done $(date -u +%T)"
$PY scripts/report.py

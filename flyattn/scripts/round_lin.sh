#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/user/XCAP/.venv-fly/bin/python
pids=()
for s in 0 1 2 3; do
  $PY scripts/t3lin_sweep.py --shard $s --n-shards 4 --steps ${STEPS:-500} --threads 1 \
    >> results/t3lin_shard$s.log 2>&1 & pids+=($!)
done
sleep ${ROUND:-460}
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
sleep 3
$PY -c "import glob,json;print('cells:',sum(len(json.load(open(f))['runs']) for f in glob.glob('results/t3lin_shard*.json')),'/48')"

#!/usr/bin/env bash
# One foreground round: four cells in parallel, sized to finish inside a single
# tool call. The container is suspended when the session goes idle, so a cell
# only makes progress while a call is executing.
set -u
cd "$(dirname "$0")/.."
PY=/home/user/XCAP/.venv-fly/bin/python
export FLYATTN_SEQ=${FLYATTN_SEQ:-160}
STEPS=${STEPS:-500}
pids=()
$PY scripts/bcg_sweep.py --test b --shard 0 --n-shards 2 --steps $STEPS --threads 1 >> results/test_b_shard0.log 2>&1 & pids+=($!)
$PY scripts/bcg_sweep.py --test b --shard 1 --n-shards 2 --steps $STEPS --threads 1 >> results/test_b_shard1.log 2>&1 & pids+=($!)
$PY scripts/bcg_sweep.py --test c --shard 0 --n-shards 1 --steps $STEPS --threads 1 >> results/test_c_shard0.log 2>&1 & pids+=($!)
$PY scripts/bcg_sweep.py --test g --shard 0 --n-shards 1 --steps $STEPS --threads 1 >> results/test_g_shard0.log 2>&1 & pids+=($!)
sleep ${ROUND:-460}
for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
sleep 3
for t in b c g; do
  n=$($PY -c "import glob,json;print(sum(len(json.load(open(f))['runs']) for f in glob.glob('results/test_${t}_shard*.json')))" 2>/dev/null || echo 0)
  echo "test $t: $n cells"
done

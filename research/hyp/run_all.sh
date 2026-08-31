#!/bin/bash
# Wait for training to finish, then run every probe and produce the report.
set -u
PY=/home/user/XCAP/.venv-hyp/bin/python
cd /home/user/XCAP/research/hyp

# NB: match the interpreter path, not the bare script name. A plain
# "train.py --steps" pattern also matches this script's own parent,
# whose command line contains that literal string, so the loop never
# exits and the whole sweep silently deadlocks.
while pgrep -f "venv-hyp/bin/python train.py" > /dev/null; do sleep 20; done
echo "=== training finished ==="
tail -2 runs/moe.log; tail -2 runs/dense.log

$PY probes.py --ckpt runs/moe/ckpt_best.pt   --out moe.json                    2>&1 | tail -2
$PY probes.py --ckpt runs/dense/ckpt_best.pt --out dense.json                  2>&1 | tail -2
$PY probes.py --ckpt runs/moe/ckpt_best.pt   --out moe_shuf.json --shuffle-text 2>&1 | tail -2
$PY probes.py --ckpt runs/moe/ckpt_0.pt      --out moe_init.json               2>&1 | tail -2
$PY attention_kernel.py --ckpt runs/moe/ckpt_best.pt --batches 12 > results/attn.txt 2>&1
echo "=== attention ==="; cat results/attn.txt
echo "=== report ==="
$PY report.py moe.json dense.json moe_shuf.json moe_init.json

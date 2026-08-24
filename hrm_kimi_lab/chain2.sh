#!/bin/bash
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
while pgrep -f "chain.sh" | grep -qv $$; do sleep 30; done
python3 run_all.py --steps 700 --out results --seeds 0 1 2 --only \
  hrm_hybrid_kda_mha hrm_hybrid_mha_kda >> results_run.log 2>&1
python3 -m lab.report >> results_run.log 2>&1
echo "CHAIN2 COMPLETE" >> results_run.log

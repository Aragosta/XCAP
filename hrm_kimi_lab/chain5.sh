#!/bin/bash
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
while pgrep -f "chain4.sh" | grep -qv $$; do sleep 30; done
python3 run_all.py --steps 700 --out results --seeds 0 1 --only \
  hrm_mha_dense_fullbp hrm_hybrid_kda_mha_fullbp hrm_loop5_kda_mha_fullbp >> results_run.log 2>&1
python3 -m lab.loop_scaling >> results_run.log 2>&1
python3 -m lab.report >> results_run.log 2>&1
echo "CHAIN5 COMPLETE" >> results_run.log

#!/bin/bash
# Phase 2 (confound controls + depth-matched HRM), phase 3 (seed variance),
# then the test-time loop-scaling probe and the report.
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
while pgrep -f "run_all.py" | grep -qv $$; do sleep 20; done
python3 run_all.py --steps 700 --out results --only \
  base_mha_conv hrm_mha_dense_d4 base_kda_noconv hrm_kda_dense_d4 >> results_run.log 2>&1
python3 run_all.py --steps 700 --out results --seeds 1 2 --only \
  base_mha_dense base_mha_conv hrm_mha_moe hrm_loop5_kda_mha \
  base_kda_dense base_kda_noconv hrm_kda_dense hrm_loop5_kda_mhamoe >> results_run.log 2>&1
python3 -m lab.loop_scaling >> results_run.log 2>&1
python3 -m lab.report >> results_run.log 2>&1
echo "CHAIN COMPLETE" >> results_run.log

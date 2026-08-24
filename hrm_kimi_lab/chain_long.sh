#!/bin/bash
# Focused long-run set: fewer variants, 2x the token budget, val curves.
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
python3 run_all.py --steps 1500 --batch-size 12 --seq-len 256 --lr 3e-3 \
  --out results_long --eval-every 100 --only \
  hrm_kda_x2_mhamoe_moe hrm_loop5_kda_mhamoe_fullbp hrm_kda_moe_fullbp \
  hrm_hybrid_kda_mhamoe_fullbp base_hybrid_kda_mhamoe_x hrm_mha_moe_fullbp base_mha_moe \
  >> long_run.log 2>&1
python3 -m lab.loop_scaling --results results_long >> long_run.log 2>&1
python3 -m lab.report --results results_long >> long_run.log 2>&1
echo "LONG RUN COMPLETE" >> long_run.log

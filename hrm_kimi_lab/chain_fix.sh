#!/bin/bash
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
python3 run_all.py --steps 650 --batch-size 8 --seq-len 192 --hidden 96 --heads 4 \
  --lr 3e-3 --out results_fixed --eval-every 50 --seeds 0 1 2 --only \
  hrm_loop5_kda_mhamoe_fullbp hrm_hybrid_kda_mhamoe_fullbp base_hybrid_kda_mhamoe_x \
  >> run_fixed.log 2>&1
python3 -m lab.report --results results_fixed >> run_fixed.log 2>&1
echo "FIXED RUN COMPLETE" >> run_fixed.log

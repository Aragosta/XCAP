#!/bin/bash
# One-hour, 3-seed test. Budget ~7200 job-seconds over 2 lanes x 2 threads.
# 3 variants x 3 seeds x 650 steps x batch 8 x seq 192 = 1.0M tokens per run.
# d_model 96 (not 128) and seq 192 (not 256) are what make 3 seeds affordable.
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
python3 run_all.py --steps 650 --batch-size 8 --seq-len 192 --hidden 96 --heads 4 \
  --lr 3e-3 --out results_1h --eval-every 50 --seeds 0 1 2 --only \
  hrm_loop5_kda_mhamoe_fullbp hrm_hybrid_kda_mhamoe_fullbp base_hybrid_kda_mhamoe_x \
  >> run_1h.log 2>&1
python3 -m lab.report --results results_1h >> run_1h.log 2>&1
echo "1H RUN COMPLETE" >> run_1h.log

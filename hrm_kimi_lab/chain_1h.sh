#!/bin/bash
# One-hour honest test. Budget: ~7200 job-seconds over 2 lanes x 2 threads.
# Every variant: same 2.66M-token budget, same batch order (paired comparison),
# MoE channel mixer, and gradients through every block application.
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
python3 run_all.py --steps 1300 --batch-size 8 --seq-len 256 --lr 3e-3 \
  --out results_1h --eval-every 100 --only \
  hrm_loop5_kda_mhamoe_fullbp hrm_kda_x2_mhamoe_moe \
  hrm_hybrid_kda_mhamoe_fullbp base_hybrid_kda_mhamoe_x base_mha_moe >> run_1h.log 2>&1
# noise anchor: cheapest variant repeated at a second seed
python3 run_all.py --steps 1300 --batch-size 8 --seq-len 256 --lr 3e-3 \
  --out results_1h --eval-every 100 --seeds 1 --only base_mha_moe >> run_1h.log 2>&1
python3 -m lab.loop_scaling --results results_1h --batch-size 8 >> run_1h.log 2>&1
python3 -m lab.report --results results_1h >> run_1h.log 2>&1
echo "1H RUN COMPLETE" >> run_1h.log

#!/bin/bash
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
while pgrep -f "chain3.sh" | grep -qv $$; do sleep 30; done
python3 run_all.py --steps 700 --out results --seeds 0 1 --only \
  base_hybrid_kda_mhamoe hrm_kda_x2_mhamoe >> results_run.log 2>&1
python3 run_all.py --steps 700 --out results --seeds 0 --only \
  hrm_kda_x1_mhamoe hrm_kda_x2_mhadense >> results_run.log 2>&1
python3 -m lab.loop_scaling >> results_run.log 2>&1
python3 -m lab.report >> results_run.log 2>&1
echo "CHAIN4 COMPLETE" >> results_run.log

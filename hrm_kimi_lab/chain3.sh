#!/bin/bash
cd /home/user/XCAP/hrm_kimi_lab
export PYTHONPATH=.:vendor
while pgrep -f "chain2.sh" | grep -qv $$; do sleep 30; done
python3 run_all.py --steps 700 --out results --seeds 0 1 2 --only \
  hrm_loop1_kda_mha hrm_loop3_kda_mha >> results_run.log 2>&1
python3 -m lab.loop_scaling >> results_run.log 2>&1
python3 -m lab.report >> results_run.log 2>&1
echo "CHAIN3 COMPLETE" >> results_run.log

#!/bin/bash
cd ~/patchfusionnice/src
until grep -q "queue finished" ../logs/run_rev3_g0.stdout && grep -q "queue finished" ../logs/run_rev3_g1.stdout; do sleep 60; done
grep -h FAILED ../logs/run_rev3_g0.stdout ../logs/run_rev3_g1.stdout
CUDA_VISIBLE_DEVICES=1 ~/.venv/bin/python rev3_eval.py 2>&1 | grep -v -i "warn\|nanmean\|aw = \|f = np" > ../logs/rev3_eval.log
echo REV3_EVAL_DONE >> ../logs/rev3_eval.log

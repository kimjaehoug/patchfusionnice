#!/bin/bash
cd ~/patchfusionnice/src
export CUDA_VISIBLE_DEVICES=1
(~/.venv/bin/python reset_full_eval.py 2>&1 | grep -v -i "warn\|nanmean" > ../logs/reset_full_eval.log) &
(~/.venv/bin/python formal_summary.py --enc --wide > ../logs/pre_enc_W.log 2>&1; echo ENC_DONE >> ../logs/pre_enc_W.log) &
wait

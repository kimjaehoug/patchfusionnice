#!/usr/bin/env bash
# 7-day reset test: one job at a time (each needs ~26 GB on GPU0)
cd "$(dirname "$0")/.."
while kill -0 1913802 2>/dev/null; do sleep 15; done
run(){ t=$1; sd=$2; seg=$3; CFG="{\"data\":\"_dyn7d\",\"bs\":1024,\"avail\":true,\"enc\":{\"cfo_gaug\":0.67,\"cfo_gaug_reg\":[\"lab\",\"cap\",\"src\"],\"imag2\":0.1,\"imag_stay\":true,\"twoview\":true,\"dynpatch\":true,\"dyn_mode\":\"hard\",\"dyn_seg\":$seg}}"; CUDA_VISIBLE_DEVICES=0 ~/.venv/bin/python src/train.py --model hpf_noq --seed $sd --tag ${t}_s${sd}q --final_epochs 10 --sub_frac 0.8 --cfg "$CFG" > logs/${t}_s${sd}q.stdout 2>&1; }
run w7nores 3 168; run w7reset 2 6; run w7nores 2 168; run w7reset 4 6; run w7nores 4 168

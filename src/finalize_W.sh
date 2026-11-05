#!/bin/bash
# Run after the hidden-256 rerun: build every *_W summary and regenerate the v7 manuscript.
set -x
cd ~/patchfusionnice/src
until grep -q "queue finished" ../logs/run_Wsplit0.stdout 2>/dev/null && grep -q "queue finished" ../logs/run_Wsplit1.stdout 2>/dev/null; do sleep 120; done
grep -h "FAILED" ../logs/run_Wfull.stdout ../logs/run_Wsplit0.stdout ../logs/run_Wsplit1.stdout
export CUDA_VISIBLE_DEVICES=0
~/.venv/bin/python formal_summary.py --wide > ../logs/fin_formal_W.log 2>&1
~/.venv/bin/python formal_summary.py --enc --wide > ../logs/fin_enc_W.log 2>&1
~/.venv/bin/python link_eval.py --pfx WX --seeds 2 3 4 5 --targets H E --write --out link_summary_W.json > ../logs/fin_link_W.log 2>&1
~/.venv/bin/python link_inter.py WX link_summary_W.json > ../logs/fin_inter_W.log 2>&1
~/.venv/bin/python dev_tradeoff.py W dev_tradeoff_W.json > ../logs/fin_dev_W.log 2>&1
~/.venv/bin/python damage_eval.py WX damage_W.json > ../logs/fin_damage_W.log 2>&1
bash /home/jbnu/patchfusionnice/paper/ESWA_draft/pipeline/build_v7.sh > ../logs/fin_build_v7.log 2>&1
ls -la ../results/*_W.json
echo FINALIZE_DONE

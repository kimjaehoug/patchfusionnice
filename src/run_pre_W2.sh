#!/bin/bash
cd ~/patchfusionnice/src
export CUDA_VISIBLE_DEVICES=1
~/.venv/bin/python formal_summary.py --wide > ../logs/pre_formal_W.log 2>&1
~/.venv/bin/python formal_summary.py --enc --wide > ../logs/pre_enc_W.log 2>&1
D=~/patchfusionnice/paper/ESWA_draft
bash $D/pipeline/build_v7.sh > ../logs/pre_build_v7.log 2>&1
PYTHONPATH=/tmp/claude-1001/-home-jbnu/c1f1cdd1-67f1-4063-8ef8-be02118e3617/scratchpad/pylib ~/.venv/bin/python $D/pipeline/docx2pdf.py $D/PatchFusion_ESWA_draft_v7.docx $D/PatchFusion_ESWA_draft_v7.pdf >> ../logs/pre_build_v7.log 2>&1
rm -f $D/PatchFusion_ESWA_draft_v7.html
echo PRE_W2_DONE >> ../logs/pre_build_v7.log

#!/bin/bash
cd ~/patchfusionnice/src
until grep -q "queue finished" ../logs/run_tfrac_g0.stdout && grep -q "queue finished" ../logs/run_tfrac_g1.stdout && grep -q REV3_BUILD_DONE ../logs/rev3_build.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 ~/.venv/bin/python tfrac_eval.py 2>&1 | grep -v -i "warn\|nanmean\|aw = \|f = np" > ../logs/tfrac_eval.log
D=~/patchfusionnice/paper/ESWA_draft
cd $D/pipeline && ~/.venv/bin/python mk_v6.py > /dev/null && ~/.venv/bin/python mk_v6_post.py > /dev/null && ~/.venv/bin/python mk_v7.py && cd $D && bash pipeline/build_v7.sh > ~/patchfusionnice/logs/tfrac_build.log 2>&1
PYTHONPATH=/tmp/claude-1001/-home-jbnu/c1f1cdd1-67f1-4063-8ef8-be02118e3617/scratchpad/pylib ~/.venv/bin/python pipeline/docx2pdf.py PatchFusion_ESWA_draft_v7.docx PatchFusion_ESWA_draft_v7.pdf >> ~/patchfusionnice/logs/tfrac_build.log 2>&1
rm -f PatchFusion_ESWA_draft_v7.html
echo TFRAC_BUILD_DONE >> ~/patchfusionnice/logs/tfrac_build.log

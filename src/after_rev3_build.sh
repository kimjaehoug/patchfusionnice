#!/bin/bash
D=~/patchfusionnice/paper/ESWA_draft
until grep -q REV3_EVAL_DONE ~/patchfusionnice/logs/rev3_eval.log 2>/dev/null && [ -s ~/patchfusionnice/results/time_sens_W.json ]; do sleep 60; done
cd $D/pipeline && ~/.venv/bin/python mk_v6.py > /dev/null && ~/.venv/bin/python mk_v6_post.py > /dev/null && ~/.venv/bin/python mk_v7.py && cd $D && bash pipeline/build_v7.sh > ~/patchfusionnice/logs/rev3_build.log 2>&1
PYTHONPATH=/tmp/claude-1001/-home-jbnu/c1f1cdd1-67f1-4063-8ef8-be02118e3617/scratchpad/pylib ~/.venv/bin/python pipeline/docx2pdf.py PatchFusion_ESWA_draft_v7.docx PatchFusion_ESWA_draft_v7.pdf >> ~/patchfusionnice/logs/rev3_build.log 2>&1
rm -f PatchFusion_ESWA_draft_v7.html
echo REV3_BUILD_DONE >> ~/patchfusionnice/logs/rev3_build.log

#!/bin/bash
# v11.1 completion chain (each step waits for its inputs)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export PYTHONIOENCODING=utf-8
L=data/processed/v11_1_final.log
until grep -q "inseason 2026 done" data/processed/inseason2026_a.log 2>/dev/null; do sleep 300; done
echo "=== step a $(date +%H:%M)" >> $L
"$PY" -W ignore build_v11_1_final.py prepare >> $L 2>&1
"$PY" -W ignore ranges_v11.py _v11i stateyr >> $L 2>&1
until grep -q "cells year 2024 done" data/processed/s2_cells_a.log 2>/dev/null && grep -q "cells year 2023 done" data/processed/s2_cells_b.log 2>/dev/null \
      && grep -q "OVERLAP" data/processed/s2_overlap.log 2>/dev/null; do sleep 300; done
echo "=== item 1: classification check + 2024 test $(date +%H:%M)" >> $L
"$PY" -W ignore s2_cells_classify.py >> $L 2>&1
cat data/processed/s2_overlap.log >> $L
"$PY" -W ignore s2_eval_2024.py >> $L 2>&1
grep -q "OVERLAP PASS" data/processed/s2_overlap.log || { echo "=== overlap FAILED: stop before assembly (owner/Claude decision)" >> $L; exit; }
until grep -q "cells year 2026 done" data/processed/s2_cells_a.log 2>/dev/null && grep -q "cells year 2025 done" data/processed/s2_cells_b.log 2>/dev/null; do sleep 300; done
echo "=== final classification + assembly $(date +%H:%M)" >> $L
"$PY" -W ignore s2_cells_classify.py >> $L 2>&1
"$PY" -W ignore build_v11_1_final.py assemble >> $L 2>&1
echo "=== v11.1 chain done $(date +%H:%M)" >> $L

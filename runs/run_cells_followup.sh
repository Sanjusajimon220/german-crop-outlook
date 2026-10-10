#!/bin/bash
# after the 413 fix: Bavaria 2024 rerun in parts, then 2024 test + classification check, then final classification
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export PYTHONIOENCODING=utf-8
L=data/processed/cells_followup.log
have() { [ -f data/processed/s2_district/cells_$1_$2/timeseries.csv ]; }
for attempt in 1 2 3; do have 09 2024 && break; echo "=== cells 09 2024 parts attempt $attempt $(date +%H:%M)" >> $L
  "$PY" -u -W ignore phase0_s2_cells.py 09 2024 2>&1 | grep -E "cells;|^status|Error:" >> $L; done
until grep -q "cells year 2024 done" data/processed/s2_cells_a.log && grep -q "cells year 2023 done" data/processed/s2_cells_b.log && have 09 2024; do sleep 300; done
echo "=== item 1 + classification check $(date +%H:%M)" >> $L
"$PY" -W ignore s2_cells_classify.py >> $L 2>&1
"$PY" -W ignore s2_eval_2024.py >> $L 2>&1
echo "=== item1 done $(date +%H:%M)" >> $L
until grep -q "cells year 2026 done" data/processed/s2_cells_a.log && grep -q "cells year 2025 done" data/processed/s2_cells_b.log; do sleep 300; done
for y in 2025 2026; do for st in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do have $st $y || echo "missing $st $y" >> $L; done; done
"$PY" -W ignore s2_cells_classify.py >> $L 2>&1
echo "=== all done $(date +%H:%M)" >> $L

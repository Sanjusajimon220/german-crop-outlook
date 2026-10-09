#!/bin/bash
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
until grep -q "barley refit done" data/processed/barley_refit.log; do sleep 60; done
echo "=== barley 2026 with refit-2023 model $(date +%H:%M)"
VISTA_WEATHER=v8_2026 VISTA_MODIS=1 VISTA_REFIT=2023 PYTHONIOENCODING=utf-8 "$PY" -W ignore phase2_predict2026.py winter_barley 2>&1 | tail -3
until grep -q "v10 prepared" data/processed/v10_prepare.log; do sleep 30; done
echo "=== v10 2026 predictions $(date +%H:%M)"
PYTHONIOENCODING=utf-8 "$PY" -W ignore make_pred2026_v10.py 2>&1 | tail -8
md5sum data/processed/check2026/*_v10.csv make_pred2026_v10.py > data/processed/check2026/frozen_2026_v10_md5.txt
echo "=== 2026 v10 frozen $(date +%H:%M)"

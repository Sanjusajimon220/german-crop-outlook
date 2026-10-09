#!/bin/bash
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 VISTA_MODIS=1 PYTHONIOENCODING=utf-8
until grep -q "barley refit done" data/processed/barley_refit.log; do sleep 60; done
echo "=== build barley $(date +%H:%M)"; "$PY" -W ignore build_v10.py winter_barley
mkdir -p data/processed/ranges_v10b
md5sum build_v10.py phase2_ranges_v10b.py score_v10_district.py score_ranges_v10b.py phase5_benchmarks.py data/processed/forecast/forecast_*_v10_lead*.csv > data/processed/ranges_v10b/frozen_v10_md5.txt
echo "=== frozen $(date +%H:%M)"
for c in winter_wheat winter_barley grain_maize silage_maize potato; do "$PY" -W ignore phase2_ranges_v10b.py $c 2>&1 | grep -E "year sd|rror"; done
echo "=== v10 prepared $(date +%H:%M)"

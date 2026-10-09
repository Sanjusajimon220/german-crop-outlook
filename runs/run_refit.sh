#!/bin/bash
# v10: refit-every-3-years test (training only): extra forward cuts 2011 and 2014 (predict 3 years each)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
until grep -q "rolling done" data/processed/rolling_forward.log 2>/dev/null; do sleep 60; done
for crop in winter_barley winter_wheat grain_maize silage_maize potato; do
  b=""; [ "$crop" = potato ] && b="physvar=potato_canopy"
  for cut in 2011 2014; do
    echo "=== $crop cut$cut $(date +%H:%M)"
    "$PY" -W ignore phase2_growth.py $crop forward $cut save $b 2>&1 | grep -E "saved|rror" | head -3
  done
done
echo "=== refit done $(date +%H:%M)"

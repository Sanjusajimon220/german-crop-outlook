#!/bin/bash
# v10 step 2 follow-up: rolling forward cuts for the year-wide error (current v8 model; training only)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
for crop in winter_wheat winter_barley grain_maize silage_maize potato; do
  b=""; [ "$crop" = potato ] && b="physvar=potato_canopy"
  for cut in 1996 1990; do
    echo "=== $crop cut$cut $(date +%H:%M)"
    "$PY" -W ignore phase2_growth.py $crop forward $cut save $b 2>&1 | grep -E "saved|rror" | head -3
  done
done
echo "=== rolling done $(date +%H:%M)"

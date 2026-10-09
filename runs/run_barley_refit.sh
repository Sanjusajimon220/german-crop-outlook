#!/bin/bash
# v10: barley refit schedule for the test look (pre-registered 2026-10-09)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 VISTA_MODIS=1 PYTHONIOENCODING=utf-8
for R in 2020 2023; do
  export VISTA_REFIT=$R
  echo "=== refit $R: growth model $(date +%H:%M)"
  "$PY" -W ignore phase2_growth.py winter_barley refit 2>&1 | grep -E "settings|blend:|test: new years|rror" | head -8
  echo "=== refit $R: in-season forecasts $(date +%H:%M)"
  "$PY" -W ignore phase2_forecast.py winter_barley 0.8 2>&1 | grep -E "lead|rror" | grep -v scenario | head -20
done
echo "=== barley refit done $(date +%H:%M)"

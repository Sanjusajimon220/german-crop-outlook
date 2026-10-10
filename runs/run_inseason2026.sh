#!/bin/bash
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8_2026 VISTA_MODIS=1 PYTHONIOENCODING=utf-8
( for c in winter_wheat silage_maize; do "$PY" -W ignore phase2_forecast2026.py $c; done ) > data/processed/inseason2026_a.log 2>&1 &
( VISTA_REFIT=2023 "$PY" -W ignore phase2_forecast2026.py winter_barley; for c in grain_maize potato; do "$PY" -W ignore phase2_forecast2026.py $c; done ) > data/processed/inseason2026_b.log 2>&1 &
wait
echo "=== inseason 2026 done $(date +%H:%M)" >> data/processed/inseason2026_a.log

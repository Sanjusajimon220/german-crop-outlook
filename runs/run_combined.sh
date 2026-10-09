#!/bin/bash
# v10 combined forward test (fit <= 2008, predict 2009-2017; training years only)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
run() { echo "=== $1 $2 $(date +%H:%M)"; "$PY" -W ignore phase2_growth.py $1 forward 2008 save $2 2>&1 | grep -E "blend 0\.[2468], extrapolated|saved|Error|error" | head -6; }
until [ -f data/processed/soil_moisture/sm_maize_2026.npz ]; do sleep 30; done
# round 1: soil moisture (baselines recomputed under identical settings)
run winter_wheat ""
run winter_wheat "physvar=soil"
run winter_barley ""
run winter_barley "physvar=soil"
run potato "physvar=potato_canopy"
run potato "physvar=potato_canopy physvar=soil"
run grain_maize "physvar=soil"
run grain_maize "physvar=soil physvar=stressboost"
run silage_maize "physvar=drought_heat physvar=soil"
echo "=== round1 done $(date +%H:%M)"
# round 2: canopy temperature (after the MODIS LST run)
until grep -q "MOD 2026\|MYD 2026" data/processed/modis_lst/run.log 2>/dev/null && [ $(ls data/processed/modis_lst/lst_*_2026.npz 2>/dev/null | wc -l) -ge 1 ]; do sleep 60; done
run winter_wheat "physvar=soil physvar=lst"
run winter_barley "physvar=soil physvar=lst"
run potato "physvar=potato_canopy physvar=soil physvar=lst"
run grain_maize "physvar=soil physvar=lst"
run silage_maize "physvar=drought_heat physvar=soil physvar=lst"
echo "=== round2 done $(date +%H:%M)"

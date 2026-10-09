#!/bin/bash
# v10 step 3: maize stress variants, forward validation (fit <= 2008, predict 2009-2017), training only
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
until [ $(ls data/processed/ranges_v10/ranges_potato_lead*.csv 2>/dev/null | wc -l) -eq 6 ]; do sleep 30; done
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
for crop in grain_maize silage_maize; do
  for v in "" "physvar=stressfeat" "physvar=kernel_water" "physvar=drought_heat" "physvar=kernel_water physvar=drought_heat physvar=stressfeat"; do
    echo "=== $crop $v $(date +%H:%M)"
    "$PY" -W ignore phase2_growth.py $crop forward 2008 save $v 2>&1 | grep -v "^\s*$" | tail -16
  done
done
echo "=== done $(date +%H:%M)"
# second round: flowering-window stress as boosting inputs (+ correction features)
for crop in grain_maize silage_maize; do
  for v in "physvar=stressboost" "physvar=stressboost physvar=stressfeat"; do
    echo "=== $crop $v $(date +%H:%M)"
    "$PY" -W ignore phase2_growth.py $crop forward 2008 save $v 2>&1 | grep -v "^\s*$" | tail -16
  done
done
echo "=== done2 $(date +%H:%M)"

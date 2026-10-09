#!/bin/bash
# v10 combined test, round 3: full combinations for every crop + confirmation period (cut 2002)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
run() { echo "=== $1 cut$2 $3 $(date +%H:%M)"; "$PY" -W ignore phase2_growth.py $1 forward $2 save $3 2>&1 | grep -E "blend 0\.[2468], extrapolated|saved|Error|error" | head -6; }
# starts at once (round 1/2 replaced by this full run)
declare -A BASE=( [winter_wheat]="" [winter_barley]="" [grain_maize]="" [silage_maize]="" [potato]="physvar=potato_canopy" )
for PASS in 1 2; do
if [ "$PASS" = "2" ]; then until [ -f data/processed/modis_lst/lst_MYD_2026.npz ]; do sleep 60; done; fi
for crop in winter_wheat winter_barley grain_maize silage_maize potato; do
  b="${BASE[$crop]}"
  for v in "" "physvar=soil" "physvar=stressboost" "physvar=drought_heat" "physvar=soil physvar=stressboost"            "physvar=soil physvar=drought_heat" "LST" "physvar=lst" "physvar=lst physvar=stressboost" "physvar=lst physvar=drought_heat" "physvar=soil physvar=lst" "physvar=soil physvar=lst physvar=stressboost"            "physvar=soil physvar=lst physvar=drought_heat" "physvar=soil physvar=lst physvar=stressboost physvar=drought_heat"; do
    if [ "$v" = "LST" ]; then continue; fi
    case "$v" in *lst*) [ "$PASS" = "2" ] || continue ;; *) [ "$PASS" = "1" ] || continue ;; esac
    for cut in 2008 2002; do
      f="data/processed/forward_predictions_${crop}_v8$(echo " $b $v" | sed 's/ physvar=/_/g; s/ //g')_cut${cut}.csv"
      [ -f "$f" ] && { echo "--- have $f"; continue; }
      run $crop $cut "$b $v"
    done
  done
done
done
echo "=== round3 done $(date +%H:%M)"

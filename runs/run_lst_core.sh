#!/bin/bash
# LST alone (v10.1 candidate), both forward periods; two streams in parallel
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 PYTHONIOENCODING=utf-8
stream() {
  for crop in "$@"; do
    b=""; [ "$crop" = "potato" ] && b="physvar=potato_canopy"
    for cut in 2008 2002; do
      f="data/processed/forward_predictions_${crop}_v8$(echo " $b physvar=lst" | sed 's/ physvar=/_/g; s/ //g')_cut${cut}.csv"
      [ -f "$f" ] && { echo "--- have $f"; continue; }
      echo "=== $crop cut$cut $b physvar=lst $(date +%H:%M)"
      "$PY" -W ignore phase2_growth.py $crop forward $cut save $b physvar=lst 2>&1 | grep -E "blend 0\.[2468], extrapolated|saved|Error|error" | head -6
    done
  done
}
stream winter_barley grain_maize winter_wheat > data/processed/lst_core_a.log 2>&1 &
stream silage_maize potato > data/processed/lst_core_b.log 2>&1 &
wait
echo "=== lst core done $(date +%H:%M)" >> data/processed/lst_core_a.log
"$PY" -W ignore eval_combined.py > data/processed/lst_core_eval.log 2>&1
"$PY" -W ignore eval_select.py >> data/processed/lst_core_eval.log 2>&1
echo "=== eval done $(date +%H:%M)" >> data/processed/lst_core_eval.log

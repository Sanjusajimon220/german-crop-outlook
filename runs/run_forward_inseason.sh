#!/bin/bash
# v11 step 1: in-season forecasts of training years 2009-2017 (models fitted <= 2008; scenarios 1979-2008)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
export VISTA_WEATHER=v8 VISTA_MODIS=1 VISTA_FORWARD=2008 PYTHONIOENCODING=utf-8
for cw in "winter_wheat 0.8" "winter_barley 0.8" "grain_maize 0.2" "silage_maize 0.6" "potato 0.4"; do
  set -- $cw
  echo "=== $1 $(date +%H:%M)"
  "$PY" -W ignore phase2_forecast.py $1 $2 2>&1 | grep -E "forward model|typical harvest|lead [0-9]+ weeks|rror" | head -12
done
echo "=== forward in-season done $(date +%H:%M)"

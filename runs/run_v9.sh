#!/bin/bash
export VISTA_WEATHER=v8
export VISTA_MODIS=1
P="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
declare -A W=( [winter_wheat]=0.8 [winter_barley]=0.8 [grain_maize]=0.2 [silage_maize]=0.6 [potato]=0.4 )
for c in winter_wheat winter_barley grain_maize silage_maize potato; do
  ( $P -u -W ignore phase2_growth.py $c > data/processed/v9_growth_${c}_log.txt 2>&1 &&
    $P -u -W ignore phase2_forecast.py $c ${W[$c]} > data/processed/v9_forecast_${c}_log.txt 2>&1 &&
    $P -u -W ignore phase2_intervals.py $c ${W[$c]} forward > data/processed/v9_intervals_${c}_log.txt 2>&1 &&
    $P -u -W ignore -c "import phase2_outputs as o; o.main('$c', ${W[$c]})" > data/processed/v9_outputs_${c}_log.txt 2>&1 ) &
done
wait
echo all v9 done

#!/bin/bash
# step 3: forward validation with MODIS features (fit <= 2008, predict 2009-2017), all crops in parallel
export VISTA_MODIS=1
P="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
$P -u -W ignore phase0_modis_features.py > data/processed/modis/features_log.txt 2>&1
for c in winter_wheat winter_barley grain_maize silage_maize potato; do
  $P -u -W ignore phase2_growth.py $c forward 2008 save > data/processed/fwd2008_${c}_modis_log.txt 2>&1 &
done
wait

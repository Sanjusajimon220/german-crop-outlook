#!/bin/bash
# forward predictions for the range method: cut 2002 (learn) and cut 2008 (check), all crops, in parallel
P="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
for c in winter_wheat winter_barley grain_maize silage_maize potato; do
  ( $P -u -W ignore phase2_growth.py $c forward 2002 save > data/processed/fwd2002_${c}_log.txt 2>&1;
    $P -u -W ignore phase2_growth.py $c forward 2008 save > data/processed/fwd2008_${c}_log.txt 2>&1 ) &
done
wait
echo all done

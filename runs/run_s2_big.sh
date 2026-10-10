#!/bin/bash
# v12 A: large DLR-based S2 samples 2019-2023, two streams
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
one() {
  for attempt in 1 2 3; do
    echo "=== big $1 $2 attempt $attempt $(date +%H:%M)"
    PYTHONIOENCODING=utf-8 "$PY" -u -W ignore phase0_s2_big.py $1 $2 2>&1 | grep -E "cells:|^status|Error:" | head -20
    [ -f data/processed/s2_district/big_$1_$2/timeseries.csv ] && return
    sleep 120
  done
}
year() {
  f=data/raw/croptypes_dlr/CROPTYPES_DE_P1Y_$1_V02.tif
  while true; do   # complete = > 400 MB and size unchanged for 90 s
    s1=$(stat -c %s $f 2>/dev/null || echo 0); sleep 90; s2=$(stat -c %s $f 2>/dev/null || echo 0)
    [ $s2 -gt 400000000 ] && [ $s1 -eq $s2 ] && break
  done
  for st in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
    [ -f data/processed/s2_district/big_${st}_$1/timeseries.csv ] || one $st $1
  done
  echo "=== big year $1 done $(date +%H:%M)"
}
( year 2023; year 2021; year 2019 ) > data/processed/s2_big_a.log 2>&1 &
( year 2022; year 2020 ) > data/processed/s2_big_b.log 2>&1 &
wait

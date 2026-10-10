#!/bin/bash
# DLR-map Sentinel-2 sampling: overlap check 2023 (16, 03), then Germany 2024 if the check passes
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
one() {  # state year
  for attempt in 1 2 3; do
    echo "=== $1 $2 dlr attempt $attempt $(date +%H:%M)"
    PYTHONIOENCODING=utf-8 "$PY" -u -W ignore phase0_s2_district_sample.py $1 $2 25 cheap dlr 2>&1 | grep -E "samples|^status|Exception|rror" | head -4
    [ -f data/processed/s2_district/sample_$1_$2_n25_cheap_dlr/timeseries.csv ] && return
    sleep 120
  done
}
for st in 16 03; do [ -f data/processed/s2_district/sample_${st}_2023_n25_cheap_dlr/timeseries.csv ] || one $st 2023; done
PYTHONIOENCODING=utf-8 "$PY" -W ignore s2_overlap.py 16 03 | tee data/processed/s2_overlap.log
grep -q "OVERLAP PASS" data/processed/s2_overlap.log || { echo "=== overlap failed, 2024 not started"; exit; }
for st in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  [ -f data/processed/s2_district/sample_${st}_2024_n25_cheap_dlr/timeseries.csv ] && continue
  one $st 2024
done
echo "=== year 2024 dlr done $(date +%H:%M)"

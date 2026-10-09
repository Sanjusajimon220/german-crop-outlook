#!/bin/bash
# Sentinel-2 sampled district greenness, all states, one year per stream (resumable)
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
Y=$1
for st in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
  [ -f data/processed/s2_district/sample_${st}_${Y}_n25_cheap/timeseries.csv ] && { echo "have $st $Y"; continue; }
  for attempt in 1 2 3; do
    echo "=== $st $Y attempt $attempt $(date +%H:%M)"
    PYTHONIOENCODING=utf-8 "$PY" -u -W ignore phase0_s2_district_sample.py $st $Y 25 cheap 2>&1 | grep -E "samples|^status|Exception|rror" | head -4
    [ -f data/processed/s2_district/sample_${st}_${Y}_n25_cheap/timeseries.csv ] && break
    sleep 120
  done
done
echo "=== year $Y done $(date +%H:%M)"

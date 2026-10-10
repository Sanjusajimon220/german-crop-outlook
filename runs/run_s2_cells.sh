#!/bin/bash
# v11.1 item 2: per-cell S2 series. Stream A: 2024 (training labels) then 2026; stream B: 2023 check (16, 03) then 2025
cd /d/vista
PY="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
one() {
  for attempt in 1 2 3; do
    echo "=== cells $1 $2 attempt $attempt $(date +%H:%M)"
    PYTHONIOENCODING=utf-8 "$PY" -u -W ignore phase0_s2_cells.py $1 $2 2>&1 | grep -E "cells;|^status|Exception|rror" | head -4
    [ -f data/processed/s2_district/cells_$1_$2/timeseries.csv ] && return
    sleep 120
  done
}
ALL="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16"
year() { for st in $2; do [ -f data/processed/s2_district/cells_${st}_$1/timeseries.csv ] || one $st $1; done; echo "=== cells year $1 done $(date +%H:%M)"; }
( year 2024 "$ALL"; year 2026 "$ALL" ) > data/processed/s2_cells_a.log 2>&1 &
sleep 60   # cells files of a state are created once (stream A first)
( year 2023 "16 03"; year 2025 "$ALL" ) > data/processed/s2_cells_b.log 2>&1 &
wait

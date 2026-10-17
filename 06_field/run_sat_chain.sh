#!/bin/bash
# Satellite-only re-run (pre-registered docs/project_log.md 2026-10-15). Stops at the first failure.
set -e
PY=/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe
L=data/processed/field_v2/pilot_2024
echo "$(date +%H:%M) paired test v2 nophys"
$PY -W ignore phase8c_fusion_physics.py v2 paired nophys > $L/fusion6_log.txt 2>&1
$PY -W ignore compare_v1_v2.py nophys > $L/compare_v1_v2_nophys_log.txt 2>&1
echo "$(date +%H:%M) smoothing test nophys"
$PY -W ignore phase8g_smooth_test.py nophys > $L/smooth_nophys_log.txt 2>&1
echo "$(date +%H:%M) weekly v1 sat"
$PY -W ignore phase8d_weekly_series.py > $L/weekly_sat_log.txt 2>&1
echo "$(date +%H:%M) weekly v2 sat"
$PY -W ignore phase8d_weekly_series.py v2 > $L/weekly_v2_sat_log.txt 2>&1
$PY -W ignore phase8f_combine.py sat > $L/combine_sat_log.txt 2>&1
$PY -W ignore plot_weekly_v2.py sat
echo "$(date +%H:%M) done"

#!/bin/bash
P="/c/Users/Asus/AppData/Local/Programs/Python/Python312/python.exe"
$P -u -W ignore phase2_growth.py grain_maize forward 2008 save physvar=maize_harvest > data/processed/fwd2008_grain_maize_harvest_log.txt 2>&1 &
$P -u -W ignore phase2_growth.py potato forward 2008 save physvar=potato_canopy > data/processed/fwd2008_potato_canopy_log.txt 2>&1 &
wait

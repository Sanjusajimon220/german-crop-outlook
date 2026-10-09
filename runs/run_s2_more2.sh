#!/bin/bash
cd /d/vista
until grep -q "year 2023 done" data/processed/s2_germany_2023.log; do sleep 120; done
./run_s2_germany.sh 2020 > data/processed/s2_germany_2020.log 2>&1

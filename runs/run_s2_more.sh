#!/bin/bash
cd /d/vista
until grep -q "year 2022 done" data/processed/s2_germany_2022.log; do sleep 120; done
./run_s2_germany.sh 2021 > data/processed/s2_germany_2021.log 2>&1
./run_s2_germany.sh 2019 > data/processed/s2_germany_2019.log 2>&1

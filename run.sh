#!/bin/bash
# Run any pipeline script from the repository root, e.g.
#   ./run.sh 02_district_model/phase2_growth.py winter_wheat
# Scripts import each other by module name, so every step folder is put on PYTHONPATH.
# Data are read from and written to ./data (not part of the repository; see documentation/REPRODUCE.md).
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/01_data:$(pwd)/02_district_model:$(pwd)/03_validation:$(pwd)/04_benchmarks:$(pwd)/05_platform:$(pwd)/06_field${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" "$@"

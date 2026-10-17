"""Chain e (log 16:59): after the Landsat test with V1 (chain c) -> Landsat test with V2 -> choose V1 / V2 on LAI (E3) +
NDVI (E2) -> fair date-blocked test E1b / E2b with the chosen variant."""
import os
import time

from run_pilot5 import step

P5 = ["pilot", "2021", "pilot", "2022", "pilot", "2023", "pilot", "2024", "pilot", "2025"]
log = os.path.join("data", "processed", "field_v2", "chain_log.txt")
while "fusion_loso_pilot_landsat exit" not in open(log).read():
    time.sleep(60)
step("fusion_landsat_v2", ["phase11_fusion_test.py", "landsat", "V2", *P5])
step("fusion_choose", ["phase11_fusion_test.py", "choose", *P5])
step("fusion_e1b_pilot", ["phase11_fusion_test.py", "e1b", *P5])

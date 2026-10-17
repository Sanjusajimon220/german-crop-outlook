"""After chain c (5-season test + Landsat): honest date-blocked tests E1b / E2b with the chosen variant (log 13:28/13:31)."""
import os
import time

from run_pilot5 import step

P5 = ["pilot", "2021", "pilot", "2022", "pilot", "2023", "pilot", "2024", "pilot", "2025"]
log = os.path.join("data", "processed", "field_v2", "chain_log.txt")
while "fusion_loso_pilot_landsat exit" not in open(log).read():
    time.sleep(60)
step("fusion_e1b_pilot", ["phase11_fusion_test.py", "e1b", *P5])

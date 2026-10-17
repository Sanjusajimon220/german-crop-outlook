"""Chain part 2 (replaces the waiting part of run_pilot5.py): wait for the running 2-season test to finish and for
pilot 2025 radar to be complete ('.complete' markers), then retrieval pilot 2025 -> 5-season leave-one-season-out test."""
import os
import time

from run_pilot5 import step

R = os.path.join("data", "raw", "field_v2", "pilot_2025")
done2 = os.path.join("data", "processed", "field_v2", "fusion_cross_summary_pilot2023_pilot2024.csv")
while not (os.path.exists(done2) and all(os.path.exists(os.path.join(R, k, ".complete")) for k in ("s2", "s1", "s1_desc"))):
    time.sleep(60)
step("retrieval_pilot", ["phase11_retrieval.py", "pilot", "2025"])
step("fusion_loso_pilot", ["phase11_fusion_test.py", "pilot", "2021", "pilot", "2022", "pilot", "2023", "pilot", "2024", "pilot", "2025"])

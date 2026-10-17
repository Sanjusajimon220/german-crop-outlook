"""After the fair pilot test (chain e): S2 + Landsat leaf retrieval for the 15 new area-seasons (log 2026-10-16)."""
import os
import time

from run_pilot5 import step

log = os.path.join("data", "processed", "field_v2", "chain_log.txt")
while "fusion_e1b_pilot exit" not in open(log).read():
    time.sleep(60)
args = [x for a in ("selhausen", "north", "south") for y in range(2021, 2026) for x in (a, str(y))]
step("retrieval_areas", ["phase11_retrieval.py", *args])

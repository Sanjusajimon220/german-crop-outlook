"""Chain g (log 20:46): after the leaf retrieval of the new areas -> area test rows -> fits + evaluation -> summary."""
import os
import time

from run_pilot5 import step

log = os.path.join("data", "processed", "field_v2", "chain_log.txt")
while "retrieval_areas exit" not in open(log).read():
    time.sleep(60)
for s in ("rows", "test", "summary"):
    step(f"area_test_{s}", ["phase12_area_test.py", s])

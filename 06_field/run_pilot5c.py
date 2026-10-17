"""Chain (project log 2026-10-16, 12:52 entry): retrieval pilot 2025 -> 5-season leave-one-season-out with variants
V0 / V1 / V2 -> Landsat (E3) with the chosen variant."""
from run_pilot5 import step

P5 = ["pilot", "2021", "pilot", "2022", "pilot", "2023", "pilot", "2024", "pilot", "2025"]
step("retrieval_pilot", ["phase11_retrieval.py", "pilot", "2025"])
step("fusion_loso_pilot", ["phase11_fusion_test.py", "loso", *P5])
step("fusion_loso_pilot_landsat", ["phase11_fusion_test.py", "landsat", *P5])

"""Chain (project log 2026-10-16): retrieval pilot 2021 / 2022 / 2025 -> 2-season first look (2023 <-> 2024, resumes)
-> wait for pilot 2025 radar -> 5-season leave-one-season-out test. Each step logs to data/processed/field_v2/."""
import os
import subprocess
import time

PY = r"C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe"
L = os.path.join("data", "processed", "field_v2")


def step(name, args):
    with open(os.path.join(L, f"{name}_log.txt"), "a") as out:
        out.write(f"--- {time.strftime('%H:%M')} start\n"); out.flush()
        r = subprocess.run([PY, "-W", "ignore", *args], stdout=out, stderr=subprocess.STDOUT)
        out.write(f"--- {time.strftime('%H:%M')} exit {r.returncode}\n")
    with open(os.path.join(L, "chain_log.txt"), "a") as c:
        c.write(f"{time.strftime('%H:%M')} {name} exit {r.returncode}\n")
    return r.returncode


def radar_done(area, year):
    """Both radar orbits of the season downloaded (download log of the running download job)."""
    log = os.path.join("data", "raw", "field_v2", "download_rest_log.txt")
    txt = open(log).read() if os.path.exists(log) else ""
    return all(f"{area} {year} {k} finished files" in txt for k in ("s1", "s1_desc"))


if __name__ == "__main__":
    step("retrieval_pilot", ["phase11_retrieval.py", "pilot", "2021", "pilot", "2022"])
    step("fusion_cross", ["phase11_fusion_test.py", "pilot", "2023", "pilot", "2024"])
    while not radar_done("pilot", 2025):
        time.sleep(60)
    step("retrieval_pilot", ["phase11_retrieval.py", "pilot", "2025"])
    step("fusion_loso_pilot", ["phase11_fusion_test.py", "pilot", "2021", "pilot", "2022", "pilot", "2023", "pilot", "2024", "pilot", "2025"])

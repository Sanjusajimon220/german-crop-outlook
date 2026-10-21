"""Detached runner: weekly series re-made with fix 2 (LAI), then the platform export (project log 2026-10-21 12:25).

Per area-season, restart-safe: the pre-fix weekly file is kept as weekly_pixels_V2_before_fix2.npz (never deleted), the new
one written, then the platform files re-exported. Output data/processed/field_v2/fix2_rebuild_out.txt, ends 'REBUILD DONE'.
"""
import os
import sys
import traceback

os.chdir(r"D:\vista")
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), r"D:\vista"]
out = open(r"D:\vista\data\processed\field_v2\fix2_rebuild_out.txt", "a", buffering=1, encoding="utf-8")
sys.stdout = sys.stderr = out
try:
    import fv2_platform as P
    import fv2_weekly as W
    from areas import AREAS, out_dir
    for a in AREAS:
        for y in range(2021, 2026):
            d = out_dir(a, y)
            cur, old = os.path.join(d, W.OUTFILE), os.path.join(d, "weekly_pixels_V2_before_fix2.npz")
            done_w, done_e = os.path.join(d, "fix2_weekly.done"), os.path.join(d, "fix2_export.done")
            if not os.path.exists(done_w):
                if os.path.exists(cur) and not os.path.exists(old):
                    os.replace(cur, old)
                elif os.path.exists(cur):
                    os.remove(cur)                              # a half-way new file from a stopped run (old copy kept)
                W.weekly_pixels(a, y)
                open(done_w, "w").write("ok\n")
            if not os.path.exists(done_e):
                mark = os.path.join(P.DOCS, f"{a}_{y}", ".complete")
                if os.path.exists(mark):
                    os.remove(mark)                             # export marker only; files are overwritten by the export
                P.export(a, y)
                open(done_e, "w").write("ok\n")
                print(f"{a} {y}: exported", flush=True)
    P.write_index()
    P.write_districts()
    print("REBUILD DONE")
except Exception:
    traceback.print_exc()
    print("REBUILD FAILED")

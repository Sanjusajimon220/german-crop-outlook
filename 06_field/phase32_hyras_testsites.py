"""HYRAS-DE 1 km daily weather for the flux-tower TEST sites (DE-Kli, DE-Geb; project log 2026-11-05).

Same source and variables as the NRW field weather (phase0_hyras_fields): tasmax, tasmin, hurs, pr. Keeps the cells inside
each site's 8 x 8 km box (+ 2 km margin); every national file is deleted after reading (temporary, ~2.3 GB in total).
Run from D:\\vista: python phase32_hyras_testsites.py 2018 2020 2021 2022 2023 2024 2025
-> data/processed/field/hyras_testsites_<year>.npz (dates, site, cells, x, y, [days, cells] per variable).
"""
import os
import sys

import numpy as np

VARS = {"tasmax": "air_temperature_max", "tasmin": "air_temperature_min", "hurs": "humidity", "pr": "precipitation"}
FD = os.path.join("data", "processed", "field")


def cells_in_box(gx, gy, box, margin=2000.0):
    """Flat indices of grid cells whose centre lies inside the box (x0, y0, x1, y1) widened by margin."""
    x0, y0, x1, y1 = box
    return np.flatnonzero((gx >= x0 - margin) & (gx <= x1 + margin) & (gy >= y0 - margin) & (gy <= y1 + margin))


def process_year(year):
    import netCDF4
    from areas import TEST_AREAS, bounds
    from weather_districts import fetch, utm32
    out, paths, sel = {}, [], None
    for var, var_dir in VARS.items():
        path = fetch(var_dir, f"{var}_hyras_1_{year}_v6-1_de.nc")
        paths.append(path)
        with netCDF4.Dataset(path) as d:
            if sel is None:
                gx, gy = utm32(d["lat"][:].ravel(), d["lon"][:].ravel())
                parts = [(s, cells_in_box(gx, gy, bounds(s))) for s in sorted(TEST_AREAS)]
                sel = dict(site=np.concatenate([[s] * len(k) for s, k in parts]),
                           cells=np.concatenate([k for _, k in parts]))
                sel["x"], sel["y"] = gx[sel["cells"]], gy[sel["cells"]]
            time = netCDF4.num2date(d["time"][:], d["time"].units)
            v = d[var]
            rows = []
            for a in range(0, v.shape[0], 31):
                g = v[a:a + 31].filled(np.nan).astype(np.float32)
                rows.append(g.reshape(g.shape[0], -1)[:, sel["cells"]])
            out[var] = np.concatenate(rows)
    dates = np.array([np.datetime64(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") for t in time])
    np.savez_compressed(os.path.join(FD, f"hyras_testsites_{year}.npz"), dates=dates, **sel, **out)
    for p in paths:
        os.remove(p)
    for s in sorted(set(sel["site"])):
        m = sel["site"] == s
        print(f"{year} {s}: {m.sum()} cells, rain {np.nanmean(out['pr'][:, m].sum(0)):.0f} mm, "
              f"Tmax mean {np.nanmean(out['tasmax'][:, m]):.1f} C", flush=True)


if __name__ == "__main__":
    for y in sys.argv[1:]:
        process_year(int(y))

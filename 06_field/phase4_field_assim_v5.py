"""Phase 4, step 3e: field model v5 = v4 (own soil + own rain/temperature) + own sunlight and
water demand.

Sunlight: CM SAF SARAH-3 daily surface radiation (0.05 deg, data/processed/field/sarah3_rur.npz).
SARAH is ~9 % above the district radiation the models were calibrated on, so only its RELATIVE
pattern is used: for each weather group, daily
  ratio = SARAH mean over the group's cells / SARAH mean over all wheat cells of the district
  (clipped 0.5-1.5);  light (PAR) of the group = district PAR x ratio.
Water demand (ET0) of the group = district ET0 x (0.7 x ratio + 0.3) x (T_group + 17.8) / (T_district + 17.8)
(radiation-driven part ~70 % of ET0 in a Hargreaves / Priestley-Taylor sense; humidity and wind
stay district). Everything else as v4. Run `python phase4_field_assim_v5.py run 2021` or `check 2022 06-01`.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer

import phase4_field_assim_v4 as v4
from phase4_field_assim import FD

_SARAH = {}


def sarah_cells():
    """HYRAS cell -> SARAH (row, col), via the mean position of the fields in that cell."""
    if not _SARAH:
        d = np.load(os.path.join(FD, "sarah3_rur.npz"))
        cell = pd.read_csv(os.path.join(FD, "fields_hyras_cell.csv"))
        pos = pd.read_csv(os.path.join(FD, "fields_district.csv"))[["ID", "VALIDFROM", "x", "y"]]
        c = cell.merge(pos, on=["ID", "VALIDFROM"]).groupby("cell")[["x", "y"]].mean()
        lon, lat = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True).transform(c["x"].values, c["y"].values)
        iy = np.abs(d["lat"][None, :] - lat[:, None]).argmin(1)
        ix = np.abs(d["lon"][None, :] - lon[:, None]).argmin(1)
        _SARAH.update(dates=d["dates"], sis=d["sis"], where=dict(zip(c.index, zip(iy, ix))))
    return _SARAH


def adjust(dist, j, sub, dates, grp, sub_district):
    s = sarah_cells()
    idx = np.searchsorted(s["dates"], dates)
    if (idx >= len(s["dates"])).any() or not (s["dates"][np.clip(idx, 0, len(s["dates"]) - 1)] == dates).all():
        return                                   # no SARAH for (part of) this season: keep district
    series = lambda cells: np.nanmean(np.stack([s["sis"][idx, s["where"][c][0], s["where"][c][1]] for c in cells
                                                if c in s["where"]]), 0)
    cells = list(grp["cell_group"])
    mine = [c for c, g in grp["cell_group"].items() if g == j]
    ratio = np.clip(series(mine) / series(cells), 0.5, 1.5)
    ratio = torch.tensor(np.nan_to_num(ratio, nan=1.0), dtype=torch.float32)[None, :]
    n = sub["par"].shape[1]
    ratio = ratio[:, :n]
    t_g = (sub["tmin"] + sub["tmax"]) / 2
    t_d = (sub_district["tmin"] + sub_district["tmax"]) / 2
    sub["par"] = sub["par"] * ratio
    sub["et0"] = sub["et0"] * (0.7 * ratio + 0.3) * torch.clamp((t_g + 17.8) / (t_d + 17.8), 0.7, 1.3)


if __name__ == "__main__":
    t0 = time.time()
    mode, year = sys.argv[1], int(sys.argv[2])
    v4.run(year, sys.argv[3] if mode == "check" else None, adjust=adjust, tag="v5")
    print(f"({time.time() - t0:.0f} s)")

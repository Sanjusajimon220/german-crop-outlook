"""Weekly 10 m maps of the final satellite-only green LAI, pilot cell 38_706, season 2024.

Input weekly_pixel_final_2024_sat.npz (phase8f_combine.py sat). Field boundaries from fields_rur_allcrops.gpkg (2024).
Outputs maps_weekly/lai_YYYY-MM-DD.png (one per week, LAI + source) and weekly_maps_overview_2024.png (12 weeks).
"""
import json
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_v2", "pilot_2024")
MAPS = os.path.join(OUT, "maps_weekly")
G = json.load(open(os.path.join(OUT, "grid_info.json")))
X0, Y1, R, NR, NC = G["origin_x"], G["origin_y"], G["res"], G["rows"], G["cols"]
EXT = (X0, X0 + NC * R, Y1 - NR * R, Y1)
CMAP = plt.get_cmap("YlGn").copy()
CMAP.set_bad("#e8e8e8")                                          # outside registered fields
SRC_CMAP = ListedColormap(["#1a9850", "#4575b4", "#f46d43", "#7b3294"])
SRC_NAMES = ["measured", "straight line", "radar-filled", "live"]


def grid(v, rows, cols):
    g = np.full((NR, NC), np.nan, np.float32)
    g[rows, cols] = v
    return g


def axes_km(ax):
    ax.set_xticks(np.arange(X0, X0 + NC * R + 1, 2000))
    ax.set_xticklabels([f"{k}" for k in range(0, NC * R // 1000 + 1, 2)])
    ax.set_yticks(np.arange(Y1 - NR * R, Y1 + 1, 2000))
    ax.set_yticklabels([f"{k}" for k in range(0, NR * R // 1000 + 1, 2)])


def main():
    z = np.load(os.path.join(OUT, "weekly_pixel_final_2024_sat.npz"))
    weeks, rows, cols = pd.DatetimeIndex(z["weeks"]), z["rows"], z["cols"]
    f = gpd.read_file(os.path.join(P, "fields_rur_allcrops.gpkg"), bbox=(EXT[0], EXT[2], EXT[1], EXT[3]))
    f = f[f["VALIDFROM"] == 2024]
    from shapely.geometry import box
    f = gpd.clip(f, box(EXT[0], EXT[2], EXT[1], EXT[3]))           # outlines only inside the pilot area
    os.makedirs(MAPS, exist_ok=True)
    for wi, wd in enumerate(weeks):
        lai, src = z["lai"][wi], z["src"][wi]
        ok = np.isfinite(lai)
        if not ok.any():
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 8.6), layout="constrained")
        im = a1.imshow(grid(lai, rows, cols), extent=EXT, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
        f.boundary.plot(ax=a1, color="0.35", lw=0.25)
        a1.set_title(f"Green LAI, week of {wd:%d %b %Y}", fontsize=13)
        fig.colorbar(im, ax=a1, shrink=0.8, label="green LAI (m² leaf / m² ground)")
        s = grid(np.where(ok, src, np.nan), rows, cols)
        a2.imshow(s, extent=EXT, cmap=SRC_CMAP, norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4), interpolation="nearest")
        f.boundary.plot(ax=a2, color="0.35", lw=0.25)
        share = [np.mean(src[ok] == k) for k in (1, 2, 3, 4)]
        a2.set_title("Where the value comes from", fontsize=13)
        fig.legend(handles=[Patch(color=SRC_CMAP(k), label=f"{SRC_NAMES[k]} ({100 * share[k]:.0f} % of pixels)") for k in range(4)],
                   loc="outside lower right", ncol=4, fontsize=10, frameon=True, title="source of the value")
        for ax in (a1, a2):
            axes_km(ax); ax.set_xlabel("km (east)"); ax.set_ylabel("km (north)")
            ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
        fig.suptitle("Pilot area 8 x 8 km (NRW) · 10 m pixels · satellite-only weekly series (Sentinel-2 + Sentinel-1)",
                     fontsize=11, color="0.3")
        fig.savefig(os.path.join(MAPS, f"lai_{wd:%Y-%m-%d}.png"), dpi=110)
        plt.close(fig)
    # overview: 12 weeks evenly spread over the weeks with data, one shared colour scale
    have = [w for w in range(len(weeks)) if np.isfinite(z["lai"][w]).any()]
    pick = [have[int(round(k))] for k in np.linspace(0, len(have) - 1, 12)]
    fig, axs = plt.subplots(3, 4, figsize=(16, 13.6), layout="constrained")
    for ax, wi in zip(axs.flat, pick):
        lai, src = z["lai"][wi], z["src"][wi]
        ok = np.isfinite(lai)
        im = ax.imshow(grid(lai, rows, cols), extent=EXT, cmap=CMAP, vmin=0, vmax=5, interpolation="nearest")
        ax.set_title(f"{weeks[wi]:%d %b %Y}  ·  measured {100 * np.mean(src[ok] == 1):.0f} %", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.5, aspect=40,
                 label="green LAI (m² leaf / m² ground)   ·   grey = outside registered fields   ·   each map 8 x 8 km, 10 m pixels")
    fig.suptitle("Weekly green LAI, pilot area NRW, season 2023/24 (12 of 46 weeks)", fontsize=15)
    fig.savefig(os.path.join(OUT, "weekly_maps_overview_2024.png"), dpi=100)
    print(f"{len(os.listdir(MAPS))} weekly maps in {MAPS}; overview weeks {[str(weeks[w].date()) for w in pick]}")


if __name__ == "__main__":
    main()

"""Charts and pooled table for the weekly SL2P comparison (phase26_sl2p_weekly.py; project log 2026-11-03)."""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"sl2p": "#eb6834", "ours_line": "#2a78d6", "ours": "#00857a"}
LAB = {"sl2p": "ESA SL2P (line only)", "ours_line": "ours, line only", "ours": "ours, published (radar for long gaps)"}
CROPS = ["wheat", "barley", "maize", "rapeseed", "sugar beet", "potato"]
AREAS = ["pilot", "selhausen", "north", "south"]


def pooled(pairs):
    """Exact pooled mean difference and RMSD per crop and pair (weighted by n) over all area-seasons."""
    p = pairs[pairs["pair"] != ""].dropna(subset=["mean_diff"])
    p = p.assign(sq=p["rmsd"] ** 2 * p["n"], sd=p["mean_diff"] * p["n"])
    g = p.groupby(["crop", "group", "pair"]).agg(n=("n", "sum"), sd=("sd", "sum"), sq=("sq", "sum")).reset_index()
    g["mean_diff"], g["rmsd"] = g["sd"] / g["n"], np.sqrt(g["sq"] / g["n"])
    return g.drop(columns=["sd", "sq"])


def _style(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(alpha=.25)


def curves(wk, crop, path, by="area"):
    w = wk[wk["crop"] == crop].copy()
    w["week"] = pd.to_datetime(w["week"])
    seasons = sorted(w["season"].unique())
    rows = AREAS if by == "area" else [crop]
    fig, ax = plt.subplots(len(rows), len(seasons), figsize=(3.2 * len(seasons), 2.4 * len(rows) + 0.6), sharey=True, squeeze=False)
    for i, area in enumerate(rows):
        for j, s in enumerate(seasons):
            a = ax[i, j]
            d = w[(w["season"] == s) & ((w["area"] == area) if by == "area" else True)]
            for name in ("sl2p", "ours_line", "ours"):
                x = d[d["series"] == name]
                if by != "area":                                       # average of the four area medians
                    x = x.groupby("week")[["q25", "median", "q75"]].mean().reset_index()
                x = x.sort_values("week")
                if name != "ours_line":
                    a.fill_between(x["week"], x["q25"], x["q75"], color=COL[name], alpha=.13, lw=0)
                a.plot(x["week"], x["median"], color=COL[name], lw=1.8 if name != "ours_line" else 1.2,
                       ls="-" if name != "ours_line" else "--", label=LAB[name])
            a.set_ylim(0, 7)
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
            a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[11, 2, 4, 6, 8]))
            _style(a)
            if i == 0:
                a.set_title(f"season {s}", fontsize=10)
            if j == 0:
                a.set_ylabel(f"{area}\ngreen LAI" if by == "area" else "green LAI")
    ax[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{crop.capitalize()}: weekly median of all pixels (shaded: middle 50 %)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def crops_panel(wk, path):
    w = wk[wk["crop"].isin(CROPS)].copy()
    w["week"] = pd.to_datetime(w["week"])
    seasons = sorted(w["season"].unique())
    fig, ax = plt.subplots(len(CROPS), len(seasons), figsize=(3.0 * len(seasons), 1.9 * len(CROPS) + 0.6), sharey=True)
    for i, c in enumerate(CROPS):
        for j, s in enumerate(seasons):
            a = ax[i, j]
            d = w[(w["crop"] == c) & (w["season"] == s)]
            for name in ("sl2p", "ours_line", "ours"):
                x = d[d["series"] == name].groupby("week")["median"].mean().sort_index()
                a.plot(x.index, x.values, color=COL[name], lw=1.6 if name != "ours_line" else 1.1,
                       ls="-" if name != "ours_line" else "--", label=LAB[name])
            a.set_ylim(0, 7)
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
            a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[11, 2, 4, 6, 8]))
            _style(a)
            if i == 0:
                a.set_title(f"season {s}", fontsize=10)
            if j == 0:
                a.set_ylabel(c)
    ax[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("All crops: weekly median LAI, average of the four study areas", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scatter(sm, path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    meas = sm["src"] == 1
    for a, (title, x, y, k) in zip(ax, [("Measured weeks (clear image within 3 days)", "sl2p", "ours", meas),
                                        ("Gap weeks, both line only", "sl2p", "ours_line", ~meas),
                                        ("Gap weeks, ours as published", "sl2p", "ours", ~meas)]):
        xv, yv = sm[x][k], sm[y][k]
        a.hexbin(xv, yv, gridsize=50, extent=(0, 7, 0, 7), bins="log", cmap="Greens", mincnt=1)
        a.plot([0, 7], [0, 7], "k--", lw=1)
        d = yv - xv
        a.text(0.3, 6.6, f"n = {len(d):,}\nr = {np.corrcoef(xv, yv)[0, 1]:.2f}\nmean diff = {d.mean():+.2f}\nRMSD = {np.sqrt((d ** 2).mean()):.2f}",
               va="top", fontsize=9)
        a.set_xlabel("ESA SL2P weekly LAI")
        a.set_ylabel("our weekly LAI")
        a.set_title(title, loc="left", fontsize=10)
        a.set_xlim(0, 7)
        a.set_ylim(0, 7)
        _style(a)
    fig.text(0.01, 0.01, "Random sample of pixel-weeks, all areas, seasons and crops. Below the dashed line: ours lower.", fontsize=8.5, color="#5b6b7a")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def charts(out):
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9})
    wk = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out, "weekly_*.csv")))])
    pairs = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in sorted(glob.glob(os.path.join(out, "pairs_*.csv")))])
    pairs["pair"] = pairs["pair"].fillna("")
    sm = {k: [] for k in ("sl2p", "ours_line", "ours", "src", "month", "crop")}
    for f in sorted(glob.glob(os.path.join(out, "sample_*.npz"))):
        d = np.load(f)
        for k in sm:
            sm[k].append(d[k])
    sm = {k: np.concatenate(v) for k, v in sm.items()}
    curves(wk, "wheat", os.path.join(out, "weekly_wheat_by_area.png"))
    crops_panel(wk, os.path.join(out, "weekly_all_crops.png"))
    scatter(sm, os.path.join(out, "scatter_weekly.png"))
    t = pooled(pairs)
    t.round(3).to_csv(os.path.join(out, "pooled_summary.csv"), index=False)
    r = []
    for c in CROPS + ["all crops"]:
        k = np.ones(len(sm["crop"]), bool) if c == "all crops" else sm["crop"] == c
        for name in ("ours_line", "ours"):
            r.append(dict(crop=c, pair=f"{name} - sl2p", r_sample=round(float(np.corrcoef(sm[name][k], sm["sl2p"][k])[0, 1]), 3)))
    pd.DataFrame(r).to_csv(os.path.join(out, "pooled_r_sample.csv"), index=False)
    dom = pairs[pairs["group"].str.startswith("clear:")]
    print(dom.groupby("group")["mean_diff"].describe().round(3).to_string())
    show = t[t["group"].isin(["all weeks", "SL2P measured", "SL2P gap-filled", "source radar"])]
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(r).to_string(index=False))

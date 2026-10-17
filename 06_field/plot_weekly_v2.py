"""Example weekly LAI series, v1 (old) vs v2 (fix 1: clearness per field), four pilot fields, 2024."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "pilot_2024")
FIELDS = {7493815: "wheat", 7738593: "sugar beet", 8056952: "potato", 7825673: "maize"}
COL = {"measured": "#1a9850", "re-analysis": "#f46d43", "interpolated": "#4575b4", "live": "#7b3294"}

old = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024.csv"), parse_dates=["week"])
SAT = "sat" in sys.argv                                           # final satellite-only series
new = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024_sat.csv" if SAT else "weekly_field_series_2024_v2.csv"), parse_dates=["week"])
NEW = "final (satellite-only)" if SAT else "v2"
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
for ax, (fid, name) in zip(axs.flat, FIELDS.items()):
    o, n = old[old.field_id == fid], new[new.field_id == fid]
    ax.fill_between(n.week, (n.lai - n.lai_uncertainty).clip(lower=0), n.lai + n.lai_uncertainty, color="0.85", label=f"± uncertainty ({NEW})")
    ax.plot(o.week, o.lai, "--", color="0.5", lw=1, label="v1 (before fix)")
    ax.plot(n.week, n.lai, "-", color="k", lw=1.2, label=NEW)
    for s, g in n.groupby("source"):
        ax.scatter(g.week, g.lai, s=22, color=COL[s], zorder=3, label=s)
    ax.set_title(f"{name} — field {fid}")
    ax.set_ylabel("green LAI (m²/m²)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))      # month names only, no overlapping dates
    ax.tick_params(axis="x", labelbottom=True)
for ax in axs[1]:
    ax.set_xlabel("season 2023/24 (Oct 2023 - Aug 2024)")
from matplotlib.lines import Line2D
h, l = axs.flat[0].get_legend_handles_labels()
for s_, c_ in COL.items():                                       # all source colours in the legend
    if s_ not in l and s_ != "live":
        h.append(Line2D([], [], marker="o", ls="", color=c_)); l.append(s_)
nice = {"measured": "measured (clear Sentinel-2)", "re-analysis": "radar-filled (re-analysis)",
        "interpolated": "straight line (gap <= 15 d)", "v1 (before fix)": "old version (v1)",
        NEW: "final weekly value", f"± uncertainty ({NEW})": "± uncertainty (typical error)"}
fig.legend(h, [nice.get(x, x) for x in l], loc="upper center", ncol=4, frameon=True, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(os.path.join(OUT, "weekly_examples_2024_sat.png" if SAT else "weekly_examples_2024_v2.png"), dpi=130)
print("saved")

"""Wheat assimilation, pilot 2024: satellite weekly LAI vs crop model before (prior) and after (posterior) assimilation."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "pilot_2024")
s = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024_sat.csv"), parse_dates=["week"])
a = pd.read_csv(os.path.join(OUT, "assim_wheat_weekly_2024.csv"), parse_dates=["date"])
r = pd.read_csv(os.path.join(OUT, "assim_wheat_2024.csv"))
q = r["yield_t_ha"].quantile([0.1, 0.5, 0.9]).values
pick = [7493815] + [r.iloc[(r["yield_t_ha"] - v).abs().argsort().iloc[0]]["field_id"] for v in (q[0], q[2])]
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), layout="constrained")
for ax, fid in zip(axs.flat, pick):
    o, p, x = s[s.field_id == fid], a[a.field_id == fid], r[r.field_id == fid].iloc[0]
    ax.plot(p.date, p.lai_prior, "--", color="0.45", label="crop model alone (prior)")
    ax.fill_between(p.date, p.lai_p10, p.lai_p90, color="#9ecae1", alpha=0.6, label="crop model + satellite, 10-90 %")
    ax.plot(p.date, p.lai_post, color="#08519c", lw=1.8, label="crop model + satellite (posterior)")
    m = o[o.source == "measured"]
    ax.scatter(o.week, o.lai, s=12, color="#f46d43", alpha=0.6, label="satellite weekly (radar-filled)")
    ax.scatter(m.week, m.lai, s=26, color="#1a9850", zorder=3, label="satellite measured (used)")
    ax.set_title(f"wheat field {int(fid)}: yield {x.yield_t_ha:.1f} t/ha (10-90 %: {x.yield_t_ha_p10:.1f}-{x.yield_t_ha_p90:.1f}), "
                 f"ET {x.phys_eta_mm:.0f} mm", fontsize=11)
    ax.set_ylabel("green LAI (m²/m²)"); ax.set_ylim(0, 6); ax.grid(alpha=0.3)
    ax.set_xlim(pd.Timestamp("2023-10-01"), pd.Timestamp("2024-08-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax = axs.flat[3]
ax.hist(r["yield_t_ha"], bins=30, color="#08519c", alpha=0.8, label="crop model + satellite")
ax.hist(r["yield_t_ha_prior"], bins=30, color="0.6", alpha=0.6, label="crop model alone")
ax.set_xlabel("field yield, t/ha (scaled so fields average to the district model)"); ax.set_ylabel("number of fields")
ax.set_title(f"582 wheat fields: yield spread after assimilation", fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5, fontsize=10)
fig.savefig(os.path.join(OUT, "assim_wheat_examples_2024.png"), dpi=120)
print("saved", pick)

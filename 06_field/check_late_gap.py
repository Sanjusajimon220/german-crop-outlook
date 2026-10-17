"""Check (pre-registered docs/project_log.md 2026-10-15): late-gap re-analysis vs independent partly cloudy S2 dates."""
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy.ndimage import binary_dilation

from phase4_prosail import BANDS, MODEL, TARGETS, Net

P = os.path.join("data", "processed")
RAW = os.path.join("data", "raw", "field_v2", "pilot_2024", "s2")
OUT = os.path.join(P, "field_v2", "pilot_2024")
DATES = ["2024-05-16", "2024-06-15", "2024-06-30", "2024-07-10"]


def main():
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    upix = px[["row", "col"]].drop_duplicates().sort_values(["row", "col"]).reset_index(drop=True)
    rows, cols = upix["row"].values, upix["col"].values
    pw = px.merge(upix.reset_index().rename(columns={"index": "pi"}), on=["row", "col"])
    wtot = pw.groupby("field_id")["share"].sum()
    ck = torch.load(MODEL, weights_only=False)
    net = Net(len(BANDS) + 3, len(TARGETS)); net.load_state_dict(ck["state"])
    s = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024.csv"), parse_dates=["week"])
    z = np.load(os.path.join(OUT, "s2_pixels.npz"))
    s2d = pd.DatetimeIndex(z["dates"])
    recs = []
    for ds in DATES:
        d = pd.Timestamp(ds)
        with rasterio.open(os.path.join(RAW, f"openEO_{ds}Z.tif")) as r:
            a = r.read()
        bad = binary_dilation(np.isin(a[10], (3, 8, 9, 10, 11)), iterations=2)
        clear = (np.isin(a[10], (4, 5)) & ~bad & np.all(np.isfinite(a[:10]), axis=0))[rows, cols]
        refl = a[:10, rows, cols].T / 10000.0
        sza, vza = a[11, rows, cols], a[12, rows, cols]
        raa = np.abs(a[13, rows, cols] - a[14, rows, cols]) % 360
        raa = np.where(raa > 180, 360 - raa, raa)
        x = np.concatenate([refl, np.cos(np.radians(np.stack([sza, vza, raa], 1)))], 1).astype(np.float32)
        ok = clear & np.all(np.isfinite(x), axis=1)
        lai = np.full(len(rows), np.nan)
        with torch.no_grad():
            mu, _ = net(torch.tensor((x[ok] - ck["xm"]) / ck["xs"], dtype=torch.float32))
        lai[ok] = np.clip(mu.numpy()[:, 0] * ck["ys"][0] + ck["ym"][0], 0, None)
        v = lai[pw["pi"].values]; m = np.isfinite(v)
        g = pd.DataFrame({"field_id": pw["field_id"].values[m], "w": pw["share"].values[m], "v": v[m] * pw["share"].values[m]})
        g = g.groupby("field_id")[["w", "v"]].sum()
        g = g[g["w"] >= 0.5 * wtot.reindex(g.index)]
        g["obs"] = g["v"] / g["w"]
        print(f"{ds}: scene clear {100 * clear.mean():.0f} %, fields >= 50 % clear: {len(g)}", flush=True)
        # weekly series at date d (linear between weeks) and straight line between clear field anchors
        w0 = d - pd.Timedelta(days=d.dayofweek); w1 = w0 + pd.Timedelta(days=7); f = d.dayofweek / 7
        a0 = s[s["week"] == w0].set_index("field_id"); a1 = s[s["week"] == w1].set_index("field_id")
        g["fused"] = (1 - f) * a0["lai"].reindex(g.index) + f * a1["lai"].reindex(g.index)
        g["source"], g["crop"] = a0["source"].reindex(g.index), a0["crop"].reindex(g.index)
        recs.append(g.assign(date=d)[["date", "crop", "source", "obs", "fused"]].reset_index())
    j = pd.concat(recs).dropna()
    # straight line from the field-mean clear S2 dates (step 2 dates) around d
    fm = pd.read_csv(os.path.join(OUT, "weekly_field_series_2024.csv"), parse_dates=["week"])
    meas = fm[fm["source"] == "measured"]
    line = []
    for (fid, d), _ in j.groupby(["field_id", "date"]):
        q = meas[meas["field_id"] == fid].set_index("week")["lai"]
        b4, af = q[q.index <= d], q[q.index > d]
        line.append(np.nan if b4.empty or af.empty else
                    np.interp(d.value, [b4.index[-1].value, af.index[0].value], [b4.iloc[-1], af.iloc[0]]))
    j = j.sort_values(["field_id", "date"]); j["line"] = j.groupby(["field_id", "date"]).ngroup().map(dict(enumerate(line)))
    j = j[j["source"] == "re-analysis"].dropna()
    j.to_csv(os.path.join(OUT, "check_late_gap.csv"), index=False)
    def stats(x):
        return pd.Series({"n": len(x), "rmse_fused": np.sqrt(((x.fused - x.obs) ** 2).mean()),
                          "bias_fused": (x.fused - x.obs).mean(), "rmse_line": np.sqrt(((x.line - x.obs) ** 2).mean()),
                          "bias_line": (x.line - x.obs).mean(), "obs_mean": x.obs.mean()})
    print(j.groupby(["date", "crop"]).apply(stats, include_groups=False).query("n >= 10").round(3).to_string())
    print(j.groupby("date").apply(stats, include_groups=False).round(3).to_string())
    print("wheat 7493815:", j[j.field_id == 7493815].round(2).to_string())
    # (b) hidden-date test: wheat pixels hidden on 25 June
    c = pd.read_csv(os.path.join(OUT, "fusion3_check.csv.gz"))
    di = list(s2d).index(pd.Timestamp("2024-06-25"))
    h = c[(c.target == "lai") & (c.crop == "wheat") & (c.date_i == di)]
    for k in ("interp", "reanalysis"):
        print(f"hidden 25 June wheat (n {len(h)}): {k} bias {(h[k] - h.obs).mean():+.3f}, RMSE {np.sqrt(((h[k] - h.obs) ** 2).mean()):.3f}")


if __name__ == "__main__":
    main()

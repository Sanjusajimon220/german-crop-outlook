"""Weekly series v2 final (pre-registered docs/project_log.md 2026-10-15, paired test): per pixel-week
  measured   any clear S2 value within +-3 days, including the per-field-clear partial dates (v2)
  re-analysis / live: the version chosen per target and v1 gap class by the paired test (choice_v1_v2.json)
  interpolated (gap <= 15 d by v1 anchors): v1 straight line
Field value = area-weighted mean (main-field shares). LAI uncertainty = area mean of the pixel uncertainties
(measured 0.15, interpolated 0.319 = fusion3 line, re-analysis / live = paired RMSE of the chosen version).
Output weekly_field_series_2024_v2.csv (+ share of the field area that uses v2 values).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join("data", "processed", "field_v2", "pilot_2024")
SAT = "sat" in sys.argv          # satellite-only fusion (owner decision 2026-10-15): *_sat inputs, smoothing if adopted
SMOOTH_W = {"S1": [1, 2, 1], "S2": [1, 2, 3, 2, 1]}


def main():
    a = np.load(os.path.join(OUT, "weekly_pixel_series_2024_sat.npz" if SAT else "weekly_pixel_series_2024.npz"))
    b = np.load(os.path.join(OUT, "weekly_pixel_series_2024_v2_sat.npz" if SAT else "weekly_pixel_series_2024_v2.npz"))
    assert np.array_equal(a["rows"], b["rows"]) and np.array_equal(a["weeks"], b["weeks"])
    ch = json.load(open(os.path.join(OUT, "choice_v1_v2_nophys.json" if SAT else "choice_v1_v2.json")))
    sm = json.load(open(os.path.join(OUT, "smooth_choice_nophys.json"))) if SAT else {"lai": {"use": "none"}, "ndvi": {"use": "none"}}
    src1, src2, gap1 = a["lai_src"], b["lai_src"], a["lai_gap"]
    gc = np.where(gap1 <= 15, "<=15 d", np.where(gap1 <= 30, "16-30 d", ">30 d"))
    src = src1.copy()
    use2 = (src2 == 1) & (src1 != 1)                              # new measured values from partial dates
    src[use2] = 1
    unc = np.where(src == 1, 0.15, np.where(src == 2, 0.319, np.nan)).astype(np.float32)
    for k, m in ((3, "reanalysis"), (4, "live")):
        for c in ("16-30 d", ">30 d"):
            sel = (src == k) & (gc == c)
            c_ = ch[f"lai|{c}|{m}"]
            unc[sel] = c_["rmse"]
            if c_["use"] == "v2":
                use2 |= sel
    out = {}
    for t in ("lai", "ndvi"):
        v = np.where(use2, b[t], a[t])
        if t == "ndvi":                                          # NDVI may choose differently from LAI
            v = a[t].copy()
            v[(src2 == 1) & (src1 != 1)] = b[t][(src2 == 1) & (src1 != 1)]
            for k, m in ((3, "reanalysis"), (4, "live")):
                for c in ("16-30 d", ">30 d"):
                    sel = (src == k) & (gc == c)
                    if ch[f"ndvi|{c}|{m}"]["use"] == "v2":
                        v[sel] = b[t][sel]
        out[t] = v
    v2re = use2 & (src == 3) & (gc == ">30 d")                    # smoothing only on v2 long-gap re-analysis weeks
    for t in ("lai", "ndvi"):
        if sm[t]["use"] != "none":
            k = np.array(SMOOTH_W[sm[t]["use"]], float)
            h = len(k) // 2
            v = out[t]
            pad = np.concatenate([np.repeat(v[:1], h, 0), v, np.repeat(v[-1:], h, 0)])
            num = sum(k[o] * np.nan_to_num(pad[o:o + len(v)]) for o in range(len(k)))
            den = sum(k[o] * np.isfinite(pad[o:o + len(v)]) for o in range(len(k)))
            out[t] = np.where(v2re & (den > 0), num / np.maximum(den, 1e-9), v)
            if t == "lai":
                unc[v2re] = sm[t]["rmse"]
            print(f"{t}: smoothing {sm[t]['use']} on {v2re.mean():.2f} of pixel-weeks", flush=True)
    lai, ndvi = np.clip(out["lai"], 0, None), np.clip(out["ndvi"], -1, 1)
    if SAT:                                                      # final 10 m pixel series for the weekly maps
        np.savez_compressed(os.path.join(OUT, "weekly_pixel_final_2024_sat.npz"), weeks=a["weeks"], rows=a["rows"],
                            cols=a["cols"], lai=lai.astype(np.float32), ndvi=ndvi.astype(np.float32), src=src, lai_unc=unc)
    px = pd.read_csv(os.path.join(OUT, "pixels_nrw.csv.gz"))
    fl = pd.read_csv(os.path.join(OUT, "fields_nrw.csv"))
    mainf = px.sort_values("share").drop_duplicates(["row", "col"], keep="last").set_index(["row", "col"])
    key = pd.MultiIndex.from_arrays([a["rows"], a["cols"]])
    fid, share = mainf["field_id"].reindex(key).values, np.nan_to_num(mainf["share"].reindex(key).values)
    fids, inv = np.unique(fid, return_inverse=True)
    F = len(fids)
    names = {1: "measured", 2: "interpolated", 3: "re-analysis", 4: "live"}
    weeks = pd.DatetimeIndex(a["weeks"])
    recs = []
    for wi, wd in enumerate(weeks):
        ok = np.isfinite(lai[wi]) & (share > 0)
        w = np.where(ok, share, 0)
        sw = np.bincount(inv, weights=w, minlength=F)
        mean = lambda x: np.bincount(inv, weights=np.where(ok, np.nan_to_num(x), 0) * w, minlength=F) / np.where(sw > 0, sw, np.nan)
        okn = ok & np.isfinite(ndvi[wi])
        wn = np.where(okn, share, 0)
        swn = np.bincount(inv, weights=wn, minlength=F)
        f_ndvi = np.bincount(inv, weights=np.where(okn, ndvi[wi], 0) * wn, minlength=F) / np.where(swn > 0, swn, np.nan)
        area = np.stack([np.bincount(inv, weights=w * (src[wi] == k), minlength=F) for k in (1, 2, 3, 4)], 1)
        keep = sw > 0
        recs.append(pd.DataFrame({
            "field_id": fids[keep], "week": wd.date(), "lai": mean(lai[wi])[keep].round(3), "ndvi": f_ndvi[keep].round(3),
            "source": [names[k] for k in (area.argmax(1) + 1)[keep]], "gap_days": mean(gap1[wi])[keep].round(0),
            "days_since_clear": mean(b["lai_since"][wi])[keep].round(0), "lai_uncertainty": mean(unc[wi])[keep].round(3),
            "v2_share": mean(use2[wi].astype(float))[keep].round(2)}))
    res = pd.concat(recs, ignore_index=True)
    res["crop"] = res["field_id"].map(fl.set_index("field_id")["crop_group"])
    res.to_csv(os.path.join(OUT, "weekly_field_series_2024_sat.csv" if SAT else "weekly_field_series_2024_v2.csv"), index=False)
    print(f"weekly v2: {res['field_id'].nunique()} fields x {res['week'].nunique()} weeks; sources "
          f"{res['source'].value_counts(normalize=True).round(3).to_dict()}; mean v2 share {res['v2_share'].mean():.2f}; "
          f"median LAI uncertainty by source {res.groupby('source')['lai_uncertainty'].median().to_dict()}")


if __name__ == "__main__":
    main()

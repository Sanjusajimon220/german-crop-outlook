"""v10 priority 1: our in-season forecasts against the official in-season figures, same point in the season.

Re-slice of the frozen test forecasts (data/processed/forecast/forecast_<crop><tag>_leadNN.csv, test years
2018-2025); nothing is refitted. Pre-registered in docs/project_log.md (2026-10-08).

Truth      official final yields: Destatis 41241-01-03-4 (Germany 'DG' and states), grain maize (not in
           that table) from the Fachserie year-end issues; where neither exists, the area-weighted mean
           of the district yields (flagged 'district_mean').
Ours       national / state value = district forecast medians (over the 39 weather scenarios) weighted
           with the district crop area of the latest census year <= forecast year (areas only exist in
           census years). Forecast date = typical harvest day - 7 x lead weeks.
Destatis   harvest-reporter estimates / preliminary results (phase0_destatis_inseason.py). Dated twice:
           'pub' = publication date from the Fachserie schedule sheet ('Uebersicht', expected month):
                   jun ~1 Aug, jul_aug ~5 Sep, aug_sep ~15 Oct, sep (2005-09) ~5 Nov;
           'ref' = end of the survey month (stricter for us): jun 30 Jun, jul_aug 31 Aug, aug_sep 30 Sep.
MARS       JRC MARS Bulletin Germany forecasts (phase0_mars_bulletins.py; OCR of image tables phase0_mars_ocr.py), dated by their information
           cut-off (end of the period covered); only values consistent with the printed % change.
Persistence  last year's official yield; climatology  mean of the previous 5 official yields.
Matching   for every official figure (Destatis issue or MARS bulletin) our forecast is the LATEST of our
           forecast dates that is on or before the publication date (we never get extra time).
Scores     error = forecast - truth; RMSE and mean absolute error over years, nationally and per state.
Writes data/processed/benchmarks/benchmark_pairs.csv and benchmark_summary.csv.
"""
import os
import re

import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
B = os.path.join(P, "benchmarks")
CROPS = ("winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato")
LEADS = (12, 8, 6, 4, 2, 0)
GENESIS = {"Winterweizen": "winter_wheat", "Wintergerste": "winter_barley", "Silomais": "silage_maize",
           "Kartoffeln": "potato"}
STATE_CODES = {"010": "01", "020": "02", "030": "03", "040": "04", "050": "05", "060": "06", "070": "07",
               "080": "08", "090": "09", "100": "10", "110": "11", "120": "12", "130": "13", "140": "14",
               "150": "15", "160": "16"}
DESTATIS_DATE = {"pub": {"jun": "08-01", "jul_aug": "09-05", "aug_sep": "10-15", "sep": "11-05"},
                 "ref": {"jun": "06-30", "jul_aug": "08-31", "aug_sep": "09-30", "sep": "09-30"}}
MARS_CROP = {"SOFT WHEAT": "winter_wheat", "WINTER BARLEY": "winter_barley", "GRAIN MAIZE": "grain_maize",
             "GREEN MAIZE": "silage_maize", "POTATO": "potato", "POTATOES": "potato"}   # MARS 'green maize' = silage maize


def num(x):
    x = re.sub(r"\s", "", str(x)).replace(",", ".").replace("–", "-").replace("+", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def truth():
    d = pd.read_csv(os.path.join("data", "raw", "yields", "41241-01-03-4_flat.csv"), sep=";", dtype=str,
                    encoding="latin-1")
    d["crop"] = d["2_variable_attribute_label"].map(GENESIS)
    code = d["1_variable_attribute_code"]
    d["region"] = np.where(code == "DG", "DE", code.map(STATE_CODES))
    d["yield_t_ha"] = pd.to_numeric(d["value"], errors="coerce") / 10
    t = d.dropna(subset=["crop", "region", "yield_t_ha"])
    t = t.assign(year=t["time"].astype(int), source="destatis_41241")[["crop", "region", "year", "yield_t_ha", "source"]]
    fs = pd.read_csv(os.path.join(B, "destatis_inseason.csv"), dtype={"state": str})
    gm = fs[(fs["crop"] == "grain_maize") & (fs["stage"] == "final")]
    gm = gm.rename(columns={"state": "region"}).assign(source="fachserie_final")
    t = pd.concat([t, gm[["crop", "region", "year", "yield_t_ha", "source"]]])
    # fallback: area-weighted district mean (grain maize years without a year-end issue)
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    y = y[y["crop"].isin(CROPS)]
    y["area_w"] = census_area(y)
    rows = []
    for (c, yr), g in y.dropna(subset=["yield_t_ha", "area_w"]).groupby(["crop", "year"]):
        rows.append(dict(crop=c, region="DE", year=yr, source="district_mean",
                         yield_t_ha=np.average(g["yield_t_ha"], weights=g["area_w"])))
        for s, h in g.groupby(g["district"].str[:2]):
            rows.append(dict(crop=c, region=s, year=yr, source="district_mean",
                             yield_t_ha=np.average(h["yield_t_ha"], weights=h["area_w"])))
    t = pd.concat([t, pd.DataFrame(rows)])
    order = {"destatis_41241": 0, "fachserie_final": 1, "district_mean": 2}
    t = t.sort_values("source", key=lambda s: s.map(order)).drop_duplicates(["crop", "region", "year"])
    return t.set_index(["crop", "region", "year"])


def census_area(y):
    """Area of each district-year: the district's crop area in the latest census year <= that year
    (or the earliest census year after it)."""
    a = y[y["area_ha"] > 0][["district", "crop", "year", "area_ha"]].sort_values("year")
    out = pd.merge_asof(y[["district", "crop", "year"]].reset_index().sort_values("year"), a.rename(
        columns={"year": "census"}), left_on="year", right_on="census", by=["district", "crop"], direction="backward")
    miss = out["area_ha"].isna()
    if miss.any():
        fwd = pd.merge_asof(y[["district", "crop", "year"]].reset_index().sort_values("year"), a.rename(
            columns={"year": "census"}), left_on="year", right_on="census", by=["district", "crop"], direction="forward")
        out.loc[miss, "area_ha"] = fwd.set_index("index").loc[out.loc[miss, "index"], "area_ha"].values
    return out.set_index("index")["area_ha"].reindex(y.index).values


def ours(crop, tag):
    """National and state forecasts per lead (median over scenarios of the blend, then area-weighted)."""
    y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
    y = y[y["crop"] == crop].copy()
    y["area_w"] = census_area(y)
    harvest = int(pd.read_csv(os.path.join(P, f"master_{crop}.csv"), usecols=["doy_harvest"])["doy_harvest"].median())
    rows = []
    for lead in LEADS:
        f = pd.read_csv(os.path.join(P, "forecast", f"forecast_{crop}{tag}_lead{lead:02d}.csv"), dtype={"district": str})
        q = f.groupby(["district", "year"])[["blend", "boosting"]].median().reset_index()
        q = q.merge(y[["district", "year", "area_w"]], on=["district", "year"], how="left").dropna(subset=["area_w"])
        q["state"] = q["district"].str[:2]
        for yr, g in q.groupby("year"):
            date = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=harvest - 7 * lead - 1)
            for region, h in [("DE", g)] + list(g.groupby("state")):
                for model in ("blend", "boosting"):
                    rows.append(dict(crop=crop, region=region, year=yr, lead=lead, date=date, source=f"ours{tag}_{model}",
                                     value=np.average(h[model], weights=h["area_w"])))
    return pd.DataFrame(rows)


def official(t):
    fs = pd.read_csv(os.path.join(B, "destatis_inseason.csv"), dtype={"state": str})
    fs = fs[fs["stage"].isin(DESTATIS_DATE["pub"]) & fs["crop"].isin(CROPS)]
    out = []
    for kind, dates in DESTATIS_DATE.items():
        f = fs.assign(date=pd.to_datetime(fs["year"].astype(str) + "-" + fs["stage"].map(dates)),
                      source="destatis_" + fs["stage"] + "_" + kind, region=fs["state"], value=fs["yield_t_ha"])
        out.append(f[["crop", "region", "year", "date", "source", "value"]])
    files = [os.path.join(B, f) for f in ("mars_germany.csv", "mars_germany_ocr.csv", "mars_germany_ocr_old.csv")]
    if any(os.path.exists(f) for f in files):
        m = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)], ignore_index=True)
        m = m[m["crop"].notna()]
        m["crop"] = m["crop"].str.upper().str.strip().map(MARS_CROP)
        m = m[m["check_pct"] == True].dropna(subset=["crop", "info_date", "forecast"])   # value checked vs printed %
        m["date"] = pd.to_datetime(m["info_date"])
        m = m.assign(year=m["bulletin_year"].astype(int), region="DE",
                     source="mars_" + m["date"].dt.strftime("%m"), value=m["forecast"])
        m = m[m["date"].dt.year == m["year"]]
        # crop identity check: the row's printed 5-yr average must be within 12 % of the official 5-yr mean
        # (text order can attach a row to the wrong crop name; the % check cannot see that)
        new = m["columns"].fillna("").str.contains("forecast", case=False)
        vals = m["raw"].fillna("").str.split("|")
        m["avg5"] = [num(v[0] if n else (v[2] if len(v) > 2 else "")) for v, n in zip(vals, new)]
        y = t["yield_t_ha"]
        m["official_avg5"] = [np.nanmean([y.get((c, "DE", yr - k), np.nan) for k in range(1, 6)])
                              for c, yr in zip(m["crop"], m["year"])]
        m["identity_ok"] = (m["avg5"] / m["official_avg5"] - 1).abs() < 0.12
        m[["crop", "year", "date", "value", "avg5", "official_avg5", "identity_ok"]].to_csv(
            os.path.join(B, "mars_identity_check.csv"), index=False)
        m = m[m["identity_ok"]]
        m = m.sort_values("date")
        out.append(m[["crop", "region", "year", "date", "source", "value"]].drop_duplicates(["crop", "source", "year"], keep="last"))
    return pd.concat(out, ignore_index=True)


def main():
    t = truth()
    t.reset_index().to_csv(os.path.join(B, "truth_official.csv"), index=False)
    off = official(t)
    pairs = []
    for crop in CROPS:
        for tag in ("_v11", "_v10", "_v9", "_v8"):
            if not os.path.exists(os.path.join(P, "forecast", f"forecast_{crop}{tag}_lead00.csv")):
                continue
            o = ours(crop, tag)
            for _, r in off[(off["crop"] == crop) & off["year"].between(2018, 2025)].iterrows():
                cand = o[(o["region"] == r["region"]) & (o["year"] == r["year"]) & (o["date"] <= r["date"])]
                for model, g in cand.groupby("source"):
                    best = g.sort_values("date").iloc[-1]
                    pairs.append(dict(crop=crop, region=r["region"], year=r["year"], official=r["source"],
                                      official_date=r["date"].date(), official_value=r["value"], ours=model,
                                      ours_lead=best["lead"], ours_date=best["date"].date(), ours_value=best["value"]))
    p = pd.DataFrame(pairs)
    key = pd.MultiIndex.from_frame(p[["crop", "region", "year"]])
    p["truth"] = t["yield_t_ha"].reindex(key).values
    p["truth_source"] = t["source"].reindex(key).values
    prev = t["yield_t_ha"]
    p["persistence"] = prev.reindex(pd.MultiIndex.from_arrays([p["crop"], p["region"], p["year"] - 1])).values
    p["climatology5"] = np.nanmean([prev.reindex(pd.MultiIndex.from_arrays([p["crop"], p["region"], p["year"] - k])).values
                                    for k in range(1, 6)], axis=0)
    p.to_csv(os.path.join(B, "benchmark_pairs.csv"), index=False)
    rows = []
    for (crop, off_src, model, nat), g in p.dropna(subset=["truth"]).groupby(
            ["crop", "official", "ours", p["region"] == "DE"]):
        def sc(col):
            e = g[col] - g["truth"]
            return round(float(np.sqrt((e ** 2).mean())), 3), round(float(e.abs().mean()), 3)
        rows.append(dict(crop=crop, level="national" if nat else "states", official=off_src, ours=model,
                         n=len(g), years=f"{g['year'].min()}-{g['year'].max()}",
                         mean_lead_weeks=round(g["ours_lead"].mean(), 1),
                         rmse_official=sc("official_value")[0], rmse_ours=sc("ours_value")[0],
                         rmse_persistence=sc("persistence")[0], rmse_climatology5=sc("climatology5")[0],
                         mae_official=sc("official_value")[1], mae_ours=sc("ours_value")[1]))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(B, "benchmark_summary.csv"), index=False)
    with pd.option_context("display.width", 250):
        print(s[s["ours"].str.endswith("v9_blend")].to_string(index=False))


if __name__ == "__main__":
    main()

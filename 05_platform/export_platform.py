"""Platform v1 (phase 1, district results) data export -> D:\\vista\\platform\\data\\*.json

districts.geojson  VG250 districts (simplified ~250 m, WGS84), properties code, name, state
crops.json         per crop: label, unit, typical harvest day, forecast dates (lead -> day of year), years
fc_<crop>.json     v10 forecasts 2018-2025: per year -> lead -> {code: [median, low80, high80, official]} for
                   districts, states ('01'..'16') and Germany ('DE'); 2026: frozen end-of-season prediction
water_<crop>.json  simulated season water use / demand (mm) and water ratio per district-year 2018-2026 (v9 physics)
ndvi.json          MODIS cropland greenness per district, 16-day composites, 2018-2026
accuracy.json      test-year accuracy: v10 vs v9 by lead, v10 vs MARS (national), vs Destatis, ranges coverage
"""
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
OUT = os.path.join("docs", "data")      # served by GitHub Pages
CROPS = {"winter_wheat": "Winter wheat", "winter_barley": "Winter barley", "grain_maize": "Grain maize",
         "silage_maize": "Silage maize", "potato": "Potato"}
LEADS = (12, 8, 6, 4, 2, 0)
STATES = {"01": "Schleswig-Holstein", "02": "Hamburg", "03": "Niedersachsen", "04": "Bremen", "05": "Nordrhein-Westfalen",
          "06": "Hessen", "07": "Rheinland-Pfalz", "08": "Baden-Wuerttemberg", "09": "Bayern", "10": "Saarland",
          "11": "Berlin", "12": "Brandenburg", "13": "Mecklenburg-Vorpommern", "14": "Sachsen", "15": "Sachsen-Anhalt",
          "16": "Thueringen"}


def r2(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    krs = gpd.read_file("/vsizip/" + os.path.join("data", "raw", "boundaries", "vg250_12-31.utm32s.shape.ebenen.zip")
                        + "/vg250_ebenen_1231/VG250_KRS.shp", columns=["AGS", "GEN", "GF"])
    krs = krs[krs["GF"] == 4].dissolve("AGS", aggfunc="first").reset_index()
    krs["geometry"] = krs.geometry.simplify(250)
    krs = krs.to_crs(4326)
    krs = krs.rename(columns={"AGS": "code", "GEN": "name"})[["code", "name", "geometry"]]
    krs["state"] = krs["code"].str[:2]
    krs.to_file(os.path.join(OUT, "districts.geojson"), driver="GeoJSON", COORDINATE_PRECISION=4)
    names = dict(zip(krs["code"], krs["name"]))
    truth = pd.read_csv(os.path.join(P, "benchmarks", "truth_official.csv"), dtype={"region": str})
    truth = truth.set_index(["crop", "region", "year"])["yield_t_ha"]
    crops_meta = {}
    for crop, label in CROPS.items():
        harvest = int(pd.read_csv(os.path.join(P, f"master_{crop}.csv"), usecols=["doy_harvest"])["doy_harvest"].median())
        fc = {}
        for lead in LEADS:
            path = os.path.join(P, "ranges_v10_final", f"ranges_{crop}_lead{lead:02d}.csv")
            if not os.path.exists(path):
                continue
            r = pd.read_csv(path, dtype={"region": str})
            for _, x in r.iterrows():
                obs = x["obs"] if x["level"] == "district" else truth.get((crop, x["region"], int(x["year"])), np.nan)
                fc.setdefault(str(int(x["year"])), {}).setdefault(str(lead), {})[x["region"]] = \
                    [r2(x["median"]), r2(x["low80"]), r2(x["high80"]), r2(obs)]
        p26 = pd.read_csv(os.path.join(P, "check2026", f"pred2026_{crop}_v10.csv"), dtype={"district": str})
        s26 = pd.read_csv(os.path.join(P, "check2026", f"state_pred2026_{crop}_v10.csv"), dtype={"state": str})
        d26 = {d: [r2(v), None, None, None] for d, v in zip(p26["district"], p26["blend"])}
        for _, x in s26.iterrows():
            d26[str(x["state"]).zfill(2)] = [r2(x["blend"]), None, None, r2(truth.get((crop, str(x["state"]).zfill(2), 2026), np.nan))]
        y = pd.read_csv(os.path.join(P, "yields_long.csv"), dtype={"district": str})
        yc = y[(y["crop"] == crop) & (y["area_ha"] > 0)].sort_values("year")
        area = p26["district"].map(yc.groupby("district")["area_ha"].last())
        ok = area.notna()
        d26["DE"] = [r2(np.average(p26.loc[ok, "blend"], weights=area[ok])), None, None, r2(truth.get((crop, "DE", 2026), np.nan))]
        fc["2026"] = {"0": d26}
        json.dump(fc, open(os.path.join(OUT, f"fc_{crop}.json"), "w"), separators=(",", ":"))
        g = pd.read_csv(os.path.join(P, f"growth_predictions_{crop}_v9.csv"), dtype={"district": str})
        g = g[g["year"] >= 2018]
        water = {}
        for _, x in g.iterrows():
            water.setdefault(str(int(x["year"])), {})[x["district"]] = [r2(x["phys_eta_mm"]), r2(x["phys_etp_mm"]),
                                                                          r2(x["phys_water_ratio"])]
        json.dump(water, open(os.path.join(OUT, f"water_{crop}.json"), "w"), separators=(",", ":"))
        crops_meta[crop] = dict(label=label, unit="t/ha", harvest_doy=harvest,
                                leads={str(l): harvest - 7 * l for l in LEADS},
                                years=sorted(fc.keys()), districts=len(p26))
    json.dump(dict(crops=crops_meta, states=STATES, names=names), open(os.path.join(OUT, "meta.json"), "w"), separators=(",", ":"))
    ndvi = {}
    for yr in range(2018, 2027):
        f = os.path.join(P, "modis", f"modis_ndvi_{yr}.npz")
        if os.path.exists(f):
            d = np.load(f)
            ndvi[str(yr)] = {"dates": [str(x)[5:10] for x in d["dates"]],
                             "v": {c: [r2(v) for v in d["ndvi"][:, i]] for i, c in enumerate(d["codes"])}}
    json.dump(ndvi, open(os.path.join(OUT, "ndvi.json"), "w"), separators=(",", ":"))
    acc = {}
    s = pd.read_csv(os.path.join(P, "v10_district_score.csv"))
    acc["district_v10_vs_v9"] = s.to_dict("records")
    p = pd.read_csv(os.path.join(P, "benchmarks", "benchmark_pairs.csv"))
    p = p[p["ours"].eq("ours_v10_blend") & ~p["official"].str.endswith("_ref")].dropna(subset=["truth"])
    rows = []
    for (c, o, lvl), gg in p.groupby(["crop", "official", p["region"].eq("DE")]):
        t = gg["truth"].mean()
        f = lambda col: round(100 * float(np.sqrt(((gg[col] - gg["truth"]) ** 2).mean())) / t, 1)
        rows.append(dict(crop=c, official=o, level="national" if lvl else "states", n=len(gg), years=f"{gg['year'].min()}-{gg['year'].max()}",
                         official_err=f("official_value"), ours_err=f("ours_value"), lastyear_err=f("persistence"),
                         avg5_err=f("climatology5")))
    acc["benchmarks"] = rows
    json.dump(acc, open(os.path.join(OUT, "accuracy.json"), "w"), separators=(",", ":"))
    for fn in sorted(os.listdir(OUT)):
        print(f"{fn:28s} {os.path.getsize(os.path.join(OUT, fn)) / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()

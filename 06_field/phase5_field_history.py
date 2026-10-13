"""Field Outlook v1, item 1: crop history per wheat field (NRW subsidy records, all crops, Rur area, 2021-2026).

For every winter wheat field of season y (2022-2026), the field of season y-1 that covers the largest part of it
(>= 50 % of its area) gives the PREVIOUS CROP; the same again for y-2. Crop names are grouped (wheat, barley,
other cereal, maize, rapeseed, sugar beet, potato, legume, grass / fodder, other). Also the organic-farming flag.
No yields are used. Check (satellite only): is the field's green leaf signal relative to its district different
by previous crop? Agronomy expects wheat after wheat to be weaker (take-all, weeds, nitrogen) and wheat after
rapeseed, legumes or root crops to be stronger.
Outputs data/processed/field_outlook/field_history_wheat.csv and history columns merged into the Field Outlook.
"""
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

P = os.path.join("data", "processed")
OUT = os.path.join(P, "field_outlook")
GROUPS = [("wheat", r"weizen|dinkel"), ("barley", r"gerste"), ("other cereal", r"roggen|triticale|hafer|getreide"),
          ("maize", r"mais"), ("rapeseed", r"raps|r.bsen"), ("sugar beet", r"r.ben"), ("potato", r"kartoffel"),
          ("legume", r"erbse|bohne|lupine|soja|leguminos|klee|luzerne|wicke"),
          ("grass / fodder", r"gras|gr.nland|weide|wiese|ackerfutter"), ("fallow", r"brache|stilllegung|aus erzeugung"), ]


def group(txt):
    t = str(txt).lower()
    for name, pat in GROUPS:
        if re.search(pat, t):
            return name
    return "other"


def previous(cur, prev):
    """For each field in cur: the prev field with the largest overlap (share of cur's area)."""
    cur = cur[["ID", "geometry"]].copy()
    cur["a"] = cur.area
    ov = gpd.overlay(cur, prev[["ID", "CODE_TXT", "geometry"]].rename(columns={"ID": "ID_prev"}), how="intersection",
                     keep_geom_type=True)
    ov["share"] = ov.area / ov["a"]
    ov = ov.sort_values("share").drop_duplicates("ID", keep="last")
    ov = ov[ov["share"] >= 0.5]
    return ov.set_index("ID")[["CODE_TXT", "share"]]


def main():
    f = gpd.read_file(os.path.join(P, "fields_rur_allcrops.gpkg"))     # all crops, Rur area (NRW records)
    f["crop_group"] = f["CODE_TXT"].map(group)
    rows = []
    for y in range(2022, 2027):
        cur = f[(f["VALIDFROM"] == y) & (f["CODE"] == 115)]
        p1 = previous(cur, f[f["VALIDFROM"] == y - 1])
        p2 = previous(cur, f[f["VALIDFROM"] == y - 2]) if y - 2 >= 2021 else pd.DataFrame(columns=["CODE_TXT", "share"])
        h = pd.DataFrame({"field_id": cur["ID"].values, "season": y, "organic": cur["ORGANICFAR"].astype(str).str.lower().eq("true").values})
        h["prev_crop_text"] = h["field_id"].map(p1["CODE_TXT"])
        h["prev_crop"] = h["prev_crop_text"].map(group).where(h["prev_crop_text"].notna(), "unknown")
        h["prev2_crop"] = h["field_id"].map(p2["CODE_TXT"]).map(group).fillna("unknown")
        h["wheat_years_in_row"] = 1 + (h["prev_crop"] == "wheat") + ((h["prev_crop"] == "wheat") & (h["prev2_crop"] == "wheat"))
        rows.append(h)
        print(y, len(h), "wheat fields; previous crop found for", f"{100 * h['prev_crop'].ne('unknown').mean():.0f} %;",
              h["prev_crop"].value_counts(normalize=True).round(3).head(6).to_dict(), flush=True)
    hist = pd.concat(rows, ignore_index=True)
    hist.to_csv(os.path.join(OUT, "field_history_wheat.csv"), index=False)
    # satellite check (no yields): relative green signal by previous crop, within district-years
    out = []
    for y in range(2022, 2027):
        if not os.path.exists(os.path.join(OUT, f"field_outlook_wheat_{y}.csv")):
            continue
        fo = pd.read_csv(os.path.join(OUT, f"field_outlook_wheat_{y}.csv"), dtype={"district": str})
        fo = fo.drop(columns=[c for c in fo.columns if c in hist.columns and c != "field_id" or c.endswith(("_x", "_y"))])
        m = fo.merge(hist[hist["season"] == y].drop(columns="season"), on="field_id", how="left")
        for c in ("glad_after", "lai_peak"):
            m[c + "_rel"] = m[c] / m.groupby("district")[c].transform("median")
        g = m.groupby("prev_crop").agg(n=("field_id", "size"), glad_rel=("glad_after_rel", "median"),
                                       lai_peak_rel=("lai_peak_rel", "median"), water_supply=("water_supply", "median"))
        out.append(g.assign(season=y).reset_index())
        m.drop(columns=[c for c in m.columns if c.endswith("_rel")]).to_csv(
            os.path.join(OUT, f"field_outlook_wheat_{y}.csv"), index=False)
    o = pd.concat(out)
    s = o.groupby("prev_crop").apply(lambda g: pd.Series({"fields": g["n"].sum(),
        "glad_rel": np.average(g["glad_rel"].fillna(1), weights=g["n"]),
        "lai_peak_rel": np.average(g["lai_peak_rel"].fillna(1), weights=g["n"])}), include_groups=False)
    s = s.sort_values("fields", ascending=False).round(3)
    s.to_csv(os.path.join(OUT, "field_history_signal.csv"))
    print("\nrelative satellite signal of wheat fields by previous crop (2022-2025; 1.00 = district median):")
    print(s.to_string())


if __name__ == "__main__":
    main()

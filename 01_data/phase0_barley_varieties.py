"""Barley variety mix per year from the Bundessortenamt 'Beschreibende Sortenliste Getreide' (2007-2026,
data/raw/bundessortenamt, public). Each edition lists every winter barley variety with its registration
year ('zugelassen seit') and its seed-multiplication area (ha) for the last 3-4 years. Seed area is a
proxy for how much of each variety farmers grow one year later.

Per seed year Y (area taken from the newest edition that reports Y):
  vintage      seed-area-weighted mean registration year of the winter barley varieties
  age          Y - vintage (how old the varieties in multiplication are)
  share_new5   share of seed area in varieties registered within the last 5 years
  share_6row   share of seed area in multi-row (mehrzeilig) varieties
  hybrid_share share in hybrid varieties (footnote '1) Hybridsorte' in the overview tables)
The crop harvested in year H was mostly sown from seed multiplied in H-1, so the index for harvest H
is the seed year H-1. Writes data/processed/barley_variety_index.csv.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader

RAW = os.path.join("data", "raw", "bundessortenamt")


def parse_edition(path):
    ed = int(re.search(r"(\d{4})", os.path.basename(path)).group(1))
    rows, hybrids = [], set()
    r = PdfReader(path)
    for p in r.pages:
        t = p.extract_text() or ""
        if "Wintergerste" not in t:
            continue
        for line in t.splitlines():                       # hybrid flags in the overview tables
            m = re.match(r"^(?:neu\s+)?([A-Za-zÄÖÜäöüß][\w\s\-\.']*?)\s+1\)\s", line)
            if m and "Hybrid" in t:
                hybrids.add(m.group(1).strip())
        if "Saatgutvermehrungsfl" not in t:
            continue
        years = [int(y) for y in re.findall(r"\b(20\d\d)\b", t.split("Wintergerste")[0])][-4:]
        six = None
        for line in t.splitlines():
            if "Wintergerste" in line:
                six = "mehrzeilig" in line
                continue
            m = re.match(r"^(?:neu\s+)?(.+?)\s+GW\s*\d+\s+((?:19|20)\d\d)\s+(.*)$", line)
            if not m or six is None:
                continue
            name, reg, rest = m.group(1).strip(), int(m.group(2)), m.group(3).split()
            areas = rest[-len(years):] if len(years) else []
            if len(areas) != len(years) or not all(re.fullmatch(r"-|\d+(\.\d+)?", a) for a in areas):
                continue
            for y, a in zip(years, areas):
                rows.append(dict(edition=ed, name=name, registered=reg, six_row=six, seed_year=y,
                                 area_ha=0.0 if a == "-" else float(a.replace(".", ""))))
    return rows, hybrids


def main():
    allrows, hyb = [], set()
    for f in sorted(glob.glob(os.path.join(RAW, "bsl_getreide_*.pdf"))):
        rows, h = parse_edition(f)
        hyb |= h
        allrows += rows
        print(f"{os.path.basename(f)}: {len(rows)} variety-year area values", flush=True)
    d = pd.DataFrame(allrows)
    d = d[d["seed_year"] < d["edition"] + 1]
    # newest edition per seed year; last column of an edition is the provisional current-year registration
    d = d[d["seed_year"] < d["edition"]]
    d = d.sort_values("edition").groupby(["seed_year", "name"]).tail(1)
    d["hybrid"] = d["name"].isin(hyb)
    out = []
    for y, g in d.groupby("seed_year"):
        w = g["area_ha"]
        if w.sum() <= 0:
            continue
        out.append(dict(seed_year=y, harvest_year=y + 1, varieties=int((w > 0).sum()), seed_area_ha=w.sum(),
                        vintage=np.average(g["registered"], weights=w), share_new5=w[g["registered"] >= y - 4].sum() / w.sum(),
                        share_6row=w[g["six_row"]].sum() / w.sum(), hybrid_share=w[g["hybrid"]].sum() / w.sum()))
    o = pd.DataFrame(out)
    o["age"] = o["seed_year"] - o["vintage"]
    o.round(3).to_csv(os.path.join("data", "processed", "barley_variety_index.csv"), index=False)
    print(o.round(2).to_string(index=False))
    print("hybrid varieties found:", sorted(hyb)[:20])


if __name__ == "__main__":
    main()

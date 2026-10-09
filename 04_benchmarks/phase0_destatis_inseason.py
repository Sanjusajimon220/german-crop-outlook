"""Official in-season yield figures (benchmark): Destatis Fachserie 3 Reihe 3.2.1 'Wachstum und Ernte -
Feldfruechte' 2005-2022 (spreadsheets in data/raw/fachserie, index fachserie_index.json).

Each issue holds several tables; each table has a title saying what it is:
  'Schaetzung' (estimate by the harvest reporters - farmers estimating the coming yield) or
  'vorlaeufiges Ergebnis' (first / second preliminary result, mostly from the official sample harvests).
Every table lists Germany and the states, with rows for a multi-year average and the reported years,
and up to two crops side by side (area, yield dt/ha, production).
Extracted: issue identifier, number and season stage (apr, jun, jul_aug, aug_sep, sep, final), table kind, crop label, state, year, yield (t/ha) for the issue's
own year. Writes data/processed/benchmarks/destatis_inseason.csv.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw", "fachserie")
OUT = os.path.join("data", "processed", "benchmarks")
STATES = {"Deutschland": "DE", "Baden-Württemberg": "08", "Bayern": "09", "Berlin": "11", "Brandenburg": "12",
          "Bremen": "04", "Hamburg": "02", "Hessen": "06", "Mecklenburg-Vorpommern": "13", "Niedersachsen": "03",
          "Nordrhein-Westfalen": "05", "Rheinland-Pfalz": "07", "Saarland": "10", "Sachsen": "14",
          "Sachsen-Anhalt": "15", "Schleswig-Holstein": "01", "Thüringen": "16", "Berlin/Brandenburg": "12"}
CROPS = {"winterweizen": "winter_wheat", "wintergerste": "winter_barley", "körnermais": "grain_maize",
         "silomais": "silage_maize", "kartoffeln": "potato", "winterraps": "rapeseed"}


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def crop_of(label):
    l = label.lower().replace("\n", " ")
    if "ohne" in l or "einschl. körnermais" in l:     # cereal totals with/without grain maize
        return None
    for k, v in CROPS.items():
        if k in l and "sommer" not in l.split(k)[0][-8:]:
            return v
    return None


def issue_stage(year, number):
    """Season stage of an issue from its number within the year (titles are not always complete).
    2005 used other numbers than 2006 onwards."""
    if year == 2005:
        table = {1: "apr", 6: "jun", 11: "jul_aug", 13: "aug_sep", 15: "sep", 21: "final"}
    else:
        table = {1: "apr", 3: "jun", 5: "jul_aug", 9: "aug_sep", 11: "sep", 13: "final", 16: "final"}
    return table.get(number, f"n{number}")


def parse_sheet(d, year):
    title = norm(d.iloc[0, 0])
    kind = "estimate" if "schätzung" in title.lower() else "preliminary" if "vorläufig" in title.lower() else "other"
    rows = []
    hdr = d.iloc[:10].astype(str)
    for col in range(d.shape[1]):
        if "Ertrag" not in hdr.iloc[:10, col].str.cat(sep=" "):
            continue
        # crop label: the closest text from row 3 down to just above the 'Ertrag' header row,
        # at or left of this column within the block
        top = int(np.flatnonzero(hdr.iloc[:10, col].str.contains("Ertrag"))[0])
        label = ""
        for c in range(col, max(col - 4, -1), -1):
            txt = " ".join(norm(x) for x in d.iloc[3:top, c] if isinstance(x, str))
            if txt:
                label = txt
                break
        crop = crop_of(label)
        if crop is None:
            continue
        state = None
        for r in range(10, d.shape[0]):
            for c in (0, 1, 2):                   # state name column differs between issues
                s = d.iloc[r, c]
                if isinstance(s, str):
                    s = norm(re.sub(r"[.…:]+", " ", s))   # dot leaders 'Brandenburg ……'
                    if s in STATES:
                        state = s
            y = d.iloc[r, 3]
            try:
                y = int(str(y).strip()[:4])
            except ValueError:
                continue
            if state and y == year:
                v = pd.to_numeric(d.iloc[r, col], errors="coerce")
                if np.isfinite(v):
                    rows.append(dict(kind=kind, title=title, crop=crop, crop_label=label, state=STATES[state],
                                     year=y, yield_t_ha=round(v / 10, 3)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = {d["mods.identifier"][0]: d for d in json.load(open(os.path.join("data", "raw", "fachserie_index.json")))}
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.xls*"))):
        ident = os.path.basename(f).split("__")[0]
        meta = idx.get(ident, {})
        titles = meta.get("mods.title", [""])
        code = re.match(r"(\d{4}),(\d{2})", titles[0])
        if code is None:
            continue
        year, number = int(code.group(1)), int(code.group(2))
        stage = issue_stage(year, number)
        try:
            x = pd.ExcelFile(f)
        except Exception as e:
            print("cannot read", f, e)
            continue
        for s in x.sheet_names:
            if not re.match(r"^\d", s):
                continue
            d = pd.read_excel(x, s, header=None)
            if d.shape[0] < 12 or d.shape[1] < 8:
                continue
            for r in parse_sheet(d, year):
                if stage == "final":
                    r["kind"] = "final"
                r.update(issue=ident, issue_number=number, stage=stage, sheet=s)
                out.append(r)
    o = pd.DataFrame(out).drop_duplicates(["issue", "kind", "crop", "state", "year"])
    o.to_csv(os.path.join(OUT, "destatis_inseason.csv"), index=False)
    t = o[o["state"] == "DE"].pivot_table(index=["year", "stage", "kind"], columns="crop", values="yield_t_ha")
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()

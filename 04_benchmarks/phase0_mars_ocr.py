"""JRC MARS Bulletin Germany forecasts for issues whose yield tables are images (2021, 2023-2025) or whose
table text comes out in a jumbled order (2026). Owner approved the re-download (38 issues, ~814 MB).

Per bulletin (one PDF at a time, deleted afterwards):
  pages      every page of the crop-yield-forecast section (from the heading 'Crop yield forecast' to
             'Atlas', or any page whose text has '[t/ha]' / '(t/ha)')
  OCR        RapidOCR on the page rendered at 250 dpi; words with their boxes are kept in
             data/raw/mars/ocr/<id>.json and a grey 100 dpi JPEG of each page in data/raw/mars/ocr/img
  tables     each 'DE' word starts a row: the values are the words on the same line to its right (up to
             the next country code column); the crop is the nearest crop title above the row whose
             horizontal extent overlaps the row
  columns    2021-2025 '5-yr avg | last year | MARS forecast | % vs 5yr | % vs last year';
             2026 adds '% vs previous month'. Forecast = 3rd value; kept only if it agrees with the
             printed % change vs last year (tolerance 1.5 points)
Writes data/processed/benchmarks/mars_germany_ocr.csv (same columns as mars_germany.csv).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pymupdf
from rapidocr_onnxruntime import RapidOCR

from phase0_mars_bulletins import CODES, MONTHS, RAW, URL, info_date, num

OCR_DIR = os.path.join(RAW, "ocr")
OUT = os.path.join("data", "processed", "benchmarks")
YEARS = (2021, 2023, 2024, 2025, 2026)
OLD = "old" in sys.argv          # 2012-2014: several crop tables per page, layout 'last | this | avg'
if OLD:
    YEARS = (2012, 2013, 2014)
NAMES = ["soft wheat", "durum wheat", "total wheat", "winter barley", "spring barley", "total barley",
         "grain maize", "green maize", "potatoes", "potato", "rye", "triticale", "rape and turnip rape", "rapeseed",
         "sunflower", "sugar beets", "sugar beet", "soybeans", "soybean"]
CROP_RE = re.compile("(" + "|".join(n.replace(" ", r"\s*") for n in NAMES) + ")", re.I)   # OCR drops spaces
_ocr = None


def ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


def selected():
    b = pd.read_csv(os.path.join(RAW, "bulletins_index.csv"))
    return b[b["bulletin_year"].isin(YEARS) & b["no"].between(3, 10)].drop_duplicates("id")


def forecast_pages(doc):
    pages, inside = [], False
    for i, p in enumerate(doc):
        t = p.get_text()
        if re.search(r"Crop yield forecasts?\s*$", t, re.I | re.M) and i > 2:
            inside = True
        if inside and re.search(r"^\s*\d*\.?\s*Atlas\s*$", t, re.I | re.M):
            inside = False
        if (inside and (p.get_images() or "t/ha" in t)) or re.search(r"[\[(]t/ha[\])]", t):
            pages.append(i)
    return pages


def ocr_page(page, path_img):
    pix = page.get_pixmap(dpi=250)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    page.get_pixmap(dpi=100, colorspace=pymupdf.csGRAY).save(path_img, jpg_quality=60)
    res, _ = ocr()(img)
    return [dict(x0=float(min(p[0] for p in b)), x1=float(max(p[0] for p in b)), y0=float(min(p[1] for p in b)),
                 y1=float(max(p[1] for p in b)), text=t) for b, t, c in (res or [])]


def fetch(row):
    out = os.path.join(OCR_DIR, f"{row.id}.json")
    if os.path.exists(out):
        return False
    os.makedirs(os.path.join(OCR_DIR, "img"), exist_ok=True)
    pdf = os.path.join(RAW, "tmp", f"{row.id}.pdf")
    if not os.path.exists(pdf):
        url = URL.format(id=row.id, file=urllib.parse.quote(urllib.parse.unquote(row.file)))
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, pdf)
                break
            except Exception as e:
                print("retry", row.id, e, flush=True)
                time.sleep(10)
        else:
            return False
    try:
        doc = pymupdf.open(pdf)
        first = "\n".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        pages = {}
        for i in forecast_pages(doc):
            pages[i] = ocr_page(doc[i], os.path.join(OCR_DIR, "img", f"{row.id}_p{i}.jpg"))
        doc.close()
    finally:
        os.remove(pdf)
    json.dump(dict(first=first, pages=pages), open(out, "w", encoding="utf-8"))
    return True


def clean(t):
    t = re.sub(r"\s", "", t).replace("–", "-").replace("−", "-").replace("±", "").replace(",", ".")
    return t


def crop_name(t):
    t = re.sub(r"\s", "", t).lower()
    return next(n for n in NAMES if n.replace(" ", "") == t).upper()


def tables(words):
    """Germany rows with their crop title, from the OCR words of one page."""
    out = []
    titles = [w for w in words if CROP_RE.search(w["text"]) and len(w["text"]) < 40]
    for de in [w for w in words if w["text"].strip() == "DE"]:
        yc, h = (de["y0"] + de["y1"]) / 2, de["y1"] - de["y0"]
        line = sorted([w for w in words if abs((w["y0"] + w["y1"]) / 2 - yc) < 0.6 * h and w["x0"] > de["x1"]],
                      key=lambda w: w["x0"])
        vals, used = [], 0
        for w in line:
            if w["text"].strip() in CODES:
                break                      # next table to the right
            used += 1
        text = " ".join(w["text"] for w in line[:used])
        vals = [clean(x) for x in re.findall(r"[+\-–−±]?\s*\d+(?:[.,]\d+)?|—", text)]
        line = line[:used]
        x_end = line[-1]["x1"] if line else de["x1"] + 400
        above = [t for t in titles if t["y1"] < de["y0"] and t["x1"] > de["x0"] and t["x0"] < x_end]
        if not above:
            continue
        title = max(above, key=lambda t: t["y1"])
        out.append(dict(crop=crop_name(CROP_RE.search(title["text"]).group(1)), title=title["text"], vals=vals))
    return out


def check_old(vals):
    """Old layout (to 2016): last year | this year | 5-yr avg | % vs last | % vs avg; forecast = 2nd."""
    v = [num(x) for x in vals[:5]]
    if len(v) < 4 or None in v[:4] or not v[0] or not v[2]:
        return None
    calcs = [abs((v[1] / v[0] - 1) * 100), abs((v[1] / v[2] - 1) * 100)]
    pcts = [abs(p) for p in v[3:5] if p is not None]
    return bool(any(abs(c - p) < 1.5 for c in calcs for p in pcts))


def check(vals):
    """The forecast (3rd value) must agree with at least one printed % change (vs 5-yr avg or vs last
    year) in magnitude (OCR may drop a coloured sign), tolerance 1.5 points."""
    v = [num(x) for x in vals[:5]]
    if len(v) < 4 or None in v[:4] or not v[0] or not v[1]:
        return None
    calcs = [abs((v[2] / v[0] - 1) * 100), abs((v[2] / v[1] - 1) * 100)]
    pcts = [abs(p) for p in v[3:5] if p is not None]
    return bool(any(abs(c - p) < 1.5 for c in calcs for p in pcts))


def parse(row):
    d = json.load(open(os.path.join(OCR_DIR, f"{row.id}.json"), encoding="utf-8"))
    date = info_date(d["first"], row.bulletin_year, None)
    out = []
    for page, words in d["pages"].items():
        for t in tables(words):
            out.append(dict(id=row.id, vol=row.vol, no=row.no, bulletin_year=row.bulletin_year,
                            info_date=date.date() if date is not None else None, crop=t["crop"],
                            columns=("last year | this year | Avg 5yrs | % last | % 5yrs" if OLD else
                                     "Avg 5yrs | last year | MARS forecast | % 5yrs | % last year"),
                            raw=" | ".join(t["vals"]),
                            forecast=num(t["vals"][1 if OLD else 2]) if len(t["vals"]) > 2 else None,
                            check_pct=(check_old if OLD else check)(t["vals"]),
                            page=int(page), source="ocr"))
    return out


def main():
    b = selected()
    if "parse" not in sys.argv:
        print(f"{len(b)} bulletins, {b['size_mb'].sum():.0f} MB", flush=True)
        for n, row in enumerate(b.itertuples(), 1):
            t0 = time.time()
            if fetch(row):
                print(f"{n}/{len(b)} {row.id} {row.bulletin_year:.0f} no {row.no:.0f} ({row.size_mb} MB) "
                      f"{time.time() - t0:.0f} s", flush=True)
    rows = []
    for row in b.itertuples():
        if os.path.exists(os.path.join(OCR_DIR, f"{row.id}.json")):
            rows += parse(row)
    o = pd.DataFrame(rows).drop_duplicates(["id", "crop", "raw"])
    o.to_csv(os.path.join(OUT, "mars_germany_ocr_old.csv" if OLD else "mars_germany_ocr.csv"), index=False)
    print(o.groupby("bulletin_year").agg(issues=("id", "nunique"), tables=("crop", "size"),
                                         ok=("check_pct", lambda s: (s == True).sum())).to_string())


if __name__ == "__main__":
    main()

"""Official benchmark 2: JRC MARS Bulletin yield forecasts for Germany (country level), 2009-2026.

Source: JRC Publications Repository (index built from its search API -> data/raw/mars/bulletins_index.csv).
Issues: all issues 2009-2015 (numbering varied), issues 3-10 (March-October) from 2016.
Each PDF is downloaded, the text of every page holding a yield table ('(t/ha)') is kept in
data/raw/mars/pages/<id>.txt (so the numbers can be re-read without downloading again), and the PDF is
deleted at once. Resumable: bulletins with a pages file are skipped.

Parse: a table starts at a line 'CROP NAME (t/ha)', followed by its column labels, then for every country
a code line ('DE') and its values (5yr average, last year, MARS forecast, % changes).
Layouts differ: one or two crops per table ('CROP (t/ha)' or 'CROP t/ha'), columns either
'last year | this year | 5-yr avg | % | %' (to ~2016) or '5-yr avg | last year | MARS forecast | % | %';
the forecast column is found from the column labels. Date = information cut-off (end of 'Period covered').
Writes data/processed/benchmarks/mars_germany.csv (bulletin, info date, crop, columns, raw values, forecast).
"""
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd
import pymupdf

RAW = os.path.join("data", "raw", "mars")
PAGES = os.path.join(RAW, "pages")
OUT = os.path.join("data", "processed", "benchmarks")
URL = "https://publications.jrc.ec.europa.eu/repository/bitstream/{id}/{file}"
CODES = {"EU", "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "EL", "HR", "HU", "IE",
         "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "UK", "EU27", "EU28"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def selected():
    b = pd.read_csv(os.path.join(RAW, "bulletins_index.csv"))
    b = b[b["bulletin_year"].between(2009, 2026)]
    return b[(b["bulletin_year"] < 2016) | b["no"].between(3, 10)]


def fetch(row):
    out = os.path.join(PAGES, f"{row.id}.txt")
    if os.path.exists(out):
        return False
    pdf = os.path.join(RAW, "tmp", f"{row.id}.pdf")
    os.makedirs(os.path.dirname(pdf), exist_ok=True)
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
        keep = [f"=== page {i}\n{p.get_text()}" for i, p in enumerate(doc) if "t/ha" in p.get_text()]
        first = doc[0].get_text() if doc.page_count else ""
        doc.close()
    finally:
        os.remove(pdf)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"=== first\n{first[:3000]}\n" + "\n".join(keep))
    return True


def num(s):
    s = s.replace(",", ".").replace("+", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def info_date(text, year, month):
    """Information cut-off of the bulletin: end of 'Period covered / of analysis' (MARS uses weather up
    to that day); else the dated header 'Vol. X No Y - D Month YYYY'; else the 20th of the issue month."""
    p = re.search(r"(?:Period (?:covered|of analysis)|Covers the period)\s*:?\s*([^\n]{0,80})", text)
    if p:
        days = re.findall(r"(\d{1,2})(?:st|nd|rd|th)?\s*(" + MONTHS + r")", p.group(1))
        if days:
            d, m = days[-1]
            return pd.Timestamp(f"{year:.0f}-{m}-{d}")
    r = re.search(r"\d{1,2}(?:st|nd|rd|th)?\s+(?:" + MONTHS + r")\s+to\s+(\d{1,2})(?:st|nd|rd|th)?\s+(" + MONTHS
                  + r")\s+(20\d\d)", text)                      # 2009-10: '11th May to 10th June 2009'
    if r:
        return pd.Timestamp(f"{r.group(3)}-{r.group(2)}-{r.group(1)}")
    h = re.search(r"No\.?\s*\d+\s*[—–-]+\s*(\d{1,2})\s+(" + MONTHS + r")\s+(20\d\d)", text)
    if h:
        return pd.Timestamp(f"{h.group(3)}-{h.group(2)}-{h.group(1)}")
    return pd.Timestamp(f"{year:.0f}-{int(month):02d}-20") if month == month and month else None


def forecast_column(labels, year):
    """Index of the current-year forecast among the 5 values of one crop table: layout up to 2016
    'last year | this year | 5-yr avg | % | %' -> 1; from 2017 '5-yr avg | last year | MARS forecast' -> 2."""
    cols = re.findall(r"MARS\s*\d{4}\s*forecasts?|Avg\s*5\s*yrs|June\s*Bulletin|%\s*\d+\s*/\s*(?:5\s*yrs|\d+)"
                      r"|\b(?:19|20)\d\d\b", labels)
    return (2 if any("forecast" in c.lower() for c in cols) else 1), cols[:6]


def consistent(v, fc):
    """Check the forecast against the printed % change vs last year (tolerance 1.5 points)."""
    vals = [num(re.sub(r"\s", "", x).replace("–", "-").replace("−", "-").replace("±", "").rstrip("-")) for x in v]
    if len(vals) < 5 or None in vals[:5]:
        return None
    last, pct = (vals[0], vals[3]) if fc == 1 else (vals[1], vals[4])
    return bool(last and abs((vals[fc] / last - 1) * 100 - pct) < 1.5)


def parse(row):
    text = open(os.path.join(PAGES, f"{row.id}.txt"), encoding="utf-8").read()
    date = info_date(text, row.bulletin_year, row.month if hasattr(row, "month") else None)
    lines = [l.strip() for l in text.split("\n")]
    out, crops, labels, width = [], [], "", 5
    k = 0
    while k < len(lines):
        if re.search(r"yield forecasts for 20\d\d \[t/ha\]", lines[k], re.I):    # 2026 layout
            prev = [x for x in lines[max(0, k - 3):k] if x]
            if prev and re.fullmatch(r"[A-Za-z][A-Za-z ,/&-]+", prev[-1]):
                crops, width, labels = [prev[-1].upper()], 6, "Avg 5yrs | last year | MARS forecast"
            k += 1
            continue
        m = re.match(r"^([A-Za-z][A-Za-z ,/&-]+?)\s*\(?t/ha\)?$", lines[k])
        if m and m.group(1).strip() .upper() not in ("YIELD",):
            crops = [m.group(1).strip().upper()]
            j = k + 1
            while j < len(lines) and re.match(r"^([A-Za-z][A-Za-z ,/&-]+?)\s*\(?t/ha\)?$", lines[j]):
                crops.append(re.match(r"^([A-Za-z][A-Za-z ,/&-]+?)\s*\(?t/ha\)?$", lines[j]).group(1).strip().upper())
                j += 1
            lab = []
            while j < len(lines) and lines[j] not in CODES and len(lab) < 40:
                lab.append(lines[j])
                j += 1
            labels, width = " ".join(lab), 5
            k = j
            continue
        if crops and lines[k] == "DE":
            vals = lines[k + 1:k + 1 + width * len(crops)]
            fc, cols = forecast_column(labels, row.bulletin_year)
            for n, crop in enumerate(crops):
                v = vals[width * n:width * n + width]
                out.append(dict(id=row.id, vol=row.vol, no=row.no, bulletin_year=row.bulletin_year,
                                info_date=date.date() if date is not None else None, crop=crop,
                                columns=" | ".join(cols), raw=" | ".join(v),
                                forecast=num(v[fc]) if fc < len(v) else None, check_pct=consistent(v, fc)))
            crops = []
        k += 1
    return out


def main():
    os.makedirs(PAGES, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    b = selected()
    if "parse" not in sys.argv:
        print(f"{len(b)} bulletins, {b['size_mb'].sum():.0f} MB", flush=True)
        for n, row in enumerate(b.itertuples(), 1):
            if fetch(row):
                print(f"{n}/{len(b)} {row.id} {row.bulletin_year:.0f} no {row.no:.0f} ({row.size_mb} MB)", flush=True)
    rows = []
    for row in b.itertuples():
        if os.path.exists(os.path.join(PAGES, f"{row.id}.txt")):
            rows += parse(row)
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(OUT, "mars_germany.csv"), index=False)
    print(o.groupby("bulletin_year").agg(issues=("id", "nunique"), tables=("crop", "size")).to_string())


if __name__ == "__main__":
    main()

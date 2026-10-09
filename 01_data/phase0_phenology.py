"""Phase 0: turn the DWD crop-stage (phenology) records into clean tables and an overview.

Input  data/raw/dwd_phenology/   files downloaded from opendata.dwd.de
Output data/processed/phenology_<crop>.csv with one row per observation:
       station, latitude, longitude, state, year, phase id, phase name, day of year
and a printed overview: how many observations per stage and decade, and when each
stage happens on average.

Run `python phase0_phenology.py`.
"""
import csv
import glob
import os
from collections import defaultdict

RAW = os.path.join("data", "raw", "dwd_phenology")
OUT = os.path.join("data", "processed")
# crop: (file name parts, DWD plant codes). Each crop has a historical file (*_hist.txt, up to
# about 2023) and a recent file (*_akt.txt, the newest years); both are read, and where a
# station reports the same stage in the same year in both, the historical value is kept.
CROPS = {"winter_wheat": ("Jahresmelder_Landwirtschaft_Kulturpflanze_Winterweizen", (202,)),
         "winter_barley": ("Jahresmelder_Landwirtschaft_Kulturpflanze_Wintergerste", (204,)),
         "maize": ("Jahresmelder_Landwirtschaft_Kulturpflanze_Mais", (215,)),
         "potato": ("Sofortmelder_Landwirtschaft_Kulturpflanze_Kartoffel", (231, 232, 233, 234))}


def read_semicolon(path):
    """DWD text files: ';'-separated, padded with spaces, latin-1, last column 'eor'."""
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter=";")
        header = [h.strip() for h in next(reader)]
        for row in reader:
            if len(row) >= len(header) - 1:
                yield dict(zip(header, (c.strip() for c in row)))


def stations():
    out = {}
    for kind in ("Jahresmelder", "Sofortmelder"):
        for r in read_semicolon(os.path.join(RAW, f"PH_Beschreibung_Phaenologie_Stationen_{kind}.txt")):
            try:
                out[int(r["Stations_id"])] = (float(r["geograph.Breite"]), float(r["geograph.Laenge"]),
                                              r["Bundesland"])
            except (KeyError, ValueError):
                continue
    return out


def phase_names():
    return {int(r["Phase_ID"]): r["Phase_englisch"] for r in read_semicolon(
        os.path.join(RAW, "PH_Beschreibung_Phase.txt")) if r.get("Phase_ID", "").isdigit()}


def main():
    os.makedirs(OUT, exist_ok=True)
    where, names = stations(), phase_names()
    for crop, (pattern, objects) in CROPS.items():
        paths = (sorted(glob.glob(os.path.join(RAW, f"PH_{pattern}_[0-9]*_hist.txt")))[-1:]
                 + glob.glob(os.path.join(RAW, f"PH_{pattern}_akt.txt")))
        seen = set()
        count = defaultdict(lambda: defaultdict(int))     # phase -> decade -> observations
        days = defaultdict(list)                           # phase -> day of year (1991-2024)
        kept = skipped = 0
        with open(os.path.join(OUT, f"phenology_{crop}.csv"), "w", newline="",
                  encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(["station", "lat", "lon", "state", "year", "phase_id", "phase", "doy"])
            for r in (row for p in paths for row in read_semicolon(p)):
                try:
                    obj, phase, year = int(r["Objekt_id"]), int(r["Phase_id"]), int(r["Referenzjahr"])
                    station, doy = int(r["Stations_id"]), int(r["Jultag"])
                except (KeyError, ValueError):
                    skipped += 1
                    continue
                if obj not in objects or station not in where or (station, year, phase) in seen:
                    skipped += 1
                    continue
                seen.add((station, year, phase))
                lat, lon, state = where[station]
                out.writerow([station, lat, lon, state, year, phase, names.get(phase, ""), doy])
                kept += 1
                count[phase][year // 10 * 10] += 1
                if year >= 1991:
                    days[phase].append(doy)

        decades = sorted({d for c in count.values() for d in c})
        print(f"\n{crop}: {kept} observations kept ({skipped} skipped or duplicate) from "
              f"{', '.join(os.path.basename(p) for p in paths)}, "
              f"{decades[0]}s to {decades[-1]}s")
        print(f"  {'stage':38s} {'1991-2024 mean day':>18s} " + "".join(f"{d}s".rjust(8) for d in decades[-5:]))
        for phase in sorted(count, key=lambda p: sorted(days[p])[len(days[p]) // 2] if days[p] else 999):
            d = sorted(days[phase])
            median = f"{d[len(d) // 2]:>5d}" if d else "    -"
            print(f"  {phase:3d} {names.get(phase, '?')[:34]:34s} {median:>18s} "
                  + "".join(f"{count[phase].get(dec, 0):8d}" for dec in decades[-5:]))


if __name__ == "__main__":
    main()

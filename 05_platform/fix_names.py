"""Repairs district names that were read with the wrong encoding (VG250 is UTF-8 but was read as cp1252, so 'Düren'
became 'DÃ¼ren'). Only names are changed; every number stays as it is. Run `python fix_names.py [docs/data]`."""
import json
import os
import sys


def repair(s):
    """'DÃ¼ren' -> 'Düren'; names that are already correct are returned unchanged."""
    if "Ã" not in s and "Â" not in s:
        return s
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fix_files(data_dir):
    m = os.path.join(data_dir, "meta.json")
    meta = json.load(open(m, encoding="utf-8"))
    meta["names"] = {k: repair(v) for k, v in meta["names"].items()}
    json.dump(meta, open(m, "w", encoding="utf-8"), separators=(",", ":"))
    g = os.path.join(data_dir, "districts.geojson")
    geo = json.load(open(g, encoding="utf-8"))
    for f in geo["features"]:
        if isinstance(f["properties"].get("name"), str):
            f["properties"]["name"] = repair(f["properties"]["name"])
    json.dump(geo, open(g, "w", encoding="utf-8"), separators=(",", ":"))


if __name__ == "__main__":
    fix_files(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "docs", "data"))

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_platform"))
import fix_names as N

DATA = Path(__file__).resolve().parents[2] / "docs" / "data"


def test_repair_double_encoded_names():
    assert N.repair("D\u00c3\u00bcren") == "Düren"
    assert N.repair("Gro\u00c3\u0178-Gerau") == "Groß-Gerau"
    assert N.repair("M\u00c3\u00a4rkischer Kreis") == "Märkischer Kreis"
    assert N.repair("Kiel") == "Kiel" and N.repair("Düren") == "Düren"      # correct names stay


def test_district_data_has_no_garbled_names():
    for f in ("meta.json", "districts.geojson"):
        assert "\u00c3" not in json.dumps(json.load(open(DATA / f, encoding="utf-8")), ensure_ascii=False), f

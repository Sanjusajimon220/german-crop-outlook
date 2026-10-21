import numpy as np
import pandas as pd

from fv2_zones_export import field_meta, zone_png_values


def test_field_meta_stability_spread_and_terrain():
    tests = pd.DataFrame({"area": "pilot", "field": [1, 1, 1, 2, 2, 2], "score": [1.0, 0.5, -0.5, 1.0, 1.0, np.nan],
                          "spread": [0.08, 0.06, 0.07, 0.02, 0.03, 0.04], "n_seasons": [5, 5, 5, 3, 3, 3]})
    checks = pd.DataFrame({"area": "pilot", "field": [1, 1, 1, 1], "var": ["slope_deg", "twi", "nfk_mm", "elev_dev_m"],
                           "weak_minus_strong": [0.8, -0.6, -12.0, 0.3]})          # height is not shown
    m = field_meta(tests, checks, "pilot")
    assert m["1"] == {"held": 2, "tests": 3, "spread": 0.07, "seasons": 5, "slope": 0.8, "twi": -0.6, "nfk": -12.0}
    assert m["2"]["held"] == 2 and m["2"]["tests"] == 2 and m["2"]["spread"] == 0.03 and "slope" not in m["2"]


def test_zone_png_values_spread_classes_over_grey_levels():
    z = np.array([[0, 1], [2, 3]], np.int8)
    v = zone_png_values(z)
    assert v.tolist() == [[0, 85], [170, 255]]


def test_field_meta_writes_valid_json_when_spread_is_missing():
    import json
    tests = pd.DataFrame({"area": "pilot", "field": [3, 3], "score": [np.nan, np.nan], "spread": [np.nan, np.nan], "n_seasons": [3, 3]})
    m = field_meta(tests, pd.DataFrame(columns=["area", "field", "var", "weak_minus_strong"]), "pilot")
    assert m["3"]["spread"] is None and m["3"]["tests"] == 0
    json.loads(json.dumps(m, allow_nan=False))

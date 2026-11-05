from pyproj import Transformer

from areas import AREAS, SEASONS, TEST_SEASONS, TEST_SITES, bounds


def test_tower_sites_are_centred_in_their_8km_box():
    tr = Transformer.from_crs(4326, 32632, always_xy=True)
    for area, (lat, lon) in {"dekli": (50.893042, 13.5223), "degeb": (51.09973, 10.91463)}.items():
        x, y = tr.transform(lon, lat)
        x0, y0, x1, y1 = bounds(area)
        assert x0 + 3000 < x < x1 - 3000 and y0 + 3000 < y < y1 - 3000


def test_test_sites_have_only_the_judged_crop_seasons_and_are_not_platform_areas():
    assert TEST_SEASONS["dekli"] == [2018, 2021, 2022, 2023, 2025] and TEST_SEASONS["degeb"] == [2020, 2021, 2022, 2024]
    assert TEST_SITES == {"dekli", "degeb"}
    assert set(AREAS) == set(SEASONS) == {"pilot", "selhausen", "north", "south"}   # platform / training never see them

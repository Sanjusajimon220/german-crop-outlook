from pyproj import Transformer

from areas import AREAS, SEASONS, TEST_SEASONS, TEST_SITES, bounds


def test_tower_sites_are_centred_in_their_8km_box():
    tr = Transformer.from_crs(4326, 32632, always_xy=True)
    for area, (lat, lon) in {"dekli": (50.893042, 13.5223), "degeb": (51.09973, 10.91463)}.items():
        x, y = tr.transform(lon, lat)
        x0, y0, x1, y1 = bounds(area)
        assert x0 + 3000 < x < x1 - 3000 and y0 + 3000 < y < y1 - 3000


def test_test_sites_have_only_the_judged_crop_seasons_and_are_not_platform_areas():
    assert TEST_SEASONS["dekli"] == [2017, 2018, 2021, 2022, 2023, 2025] and TEST_SEASONS["degeb"] == [2020, 2021, 2022, 2024]
    assert TEST_SEASONS["belon"] == [2017, 2018, 2019, 2020, 2022, 2023, 2024]
    assert TEST_SEASONS["frgri"] == [2018, 2019, 2020, 2022, 2023, 2024, 2025]
    assert TEST_SITES == {"dekli", "degeb", "belon", "frgri"}
    assert set(AREAS) == set(SEASONS) == {"pilot", "selhausen", "north", "south"}   # platform / training never see them


def test_heldout_towers_sit_in_a_3km_box():
    tr = Transformer.from_crs(4326, 32632, always_xy=True)
    for area, (lat, lon) in {"belon": (50.5516198, 4.7462339), "frgri": (48.84422, 1.95191)}.items():
        x, y = tr.transform(lon, lat)
        x0, y0, x1, y1 = bounds(area)
        assert x1 - x0 == 3000 and y1 - y0 == 3000
        assert x0 + 1000 < x < x1 - 1000 and y0 + 1000 < y < y1 - 1000

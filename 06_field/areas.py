"""Study areas (8 x 8 km, EPSG:25832 lower-left corner) and seasons for field v2 (project log 2026-10-16)."""
import os

AREAS = {"pilot": (304000, 5648000), "selhausen": (316000, 5634000), "north": (288000, 5656000), "south": (328000, 5608000)}
SIZE = 8000
SEASONS = {"pilot": [2021, 2022, 2023, 2024, 2025], "selhausen": [2021, 2022, 2023, 2024, 2025],
           "north": [2021, 2022, 2023, 2024, 2025], "south": [2021, 2022, 2023, 2024, 2025]}
# Flux-tower TEST sites outside NRW (thermal water method, project log 2026-11-05): 8 x 8 km around the ICOS tower,
# only the seasons with district-model crops (DE-Kli: maize 2018, wheat 2021 / 2025, spring barley 2022, winter barley
# 2023; DE-Geb: potato 2020, wheat 2021 / 2022 / 2024). Used for testing only, never shown on the platform.
# Kept OUT of AREAS / SEASONS so platform export, production training and the area test never loop over them.
# Held-out towers for the water v5 test (project log 2026-11-06): ICOS BE-Lon (Lonzee) and FR-Gri (Grignon), small
# fields -> 3 x 3 km box around the tower (Sentinel-2 only). FR-Gri lies west of UTM zone 32 (x < 0; ~0.3 % scale error).
TEST_AREAS = {"dekli": (814000, 5644000), "degeb": (630000, 5659000), "belon": (197200, 5607110), "frgri": (-18440, 5432630)}
TEST_SIZE = {"belon": 3000, "frgri": 3000}
TEST_SEASONS = {"dekli": [2017, 2018, 2021, 2022, 2023, 2025], "degeb": [2020, 2021, 2022, 2024],
                "belon": [2017, 2018, 2019, 2020, 2022, 2023, 2024], "frgri": [2018, 2019, 2020, 2022, 2023, 2024, 2025]}
TEST_SITES = set(TEST_AREAS)


def bounds(area):
    x0, y0 = AREAS[area] if area in AREAS else TEST_AREAS[area]
    n = TEST_SIZE.get(area, SIZE)
    return x0, y0, x0 + n, y0 + n


def period(year):
    """Season = 1 Oct of the previous year to 31 Aug."""
    return [f"{year - 1}-10-01", f"{year}-08-31"]


def raw_dir(area, year):
    return os.path.join("data", "raw", "field_v2", f"{area}_{year}")


def out_dir(area, year):
    return os.path.join("data", "processed", "field_v2", f"{area}_{year}")

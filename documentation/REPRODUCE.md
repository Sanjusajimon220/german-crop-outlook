# How to reproduce

All commands from the repository root: `./run.sh <script> [arguments]` (sets the module path; Python 3.12,
`pip install -r requirements.txt`). Outputs go to `data/processed/`. Steps in order; each script's header explains
its inputs, method and outputs. The project log (`project_log.md`) records why each step exists.

## 1. Data (`01_data`)
1. Weather: `weather_districts.py` (DWD HYRAS-DE daily, district means), `phase0_stations.py` (wind),
   `phase0_sarah3.py` + `phase0_sarah_districts.py` (CM SAF SARAH-3 sunlight; ordered from CM SAF),
   `phase0_solar_check.py` (check against pyranometers), `phase0_weather_2026.py` (current season).
2. Soil: `phase0_soil.py`. Yields: `phase0_yields.py`, `phase0_yields_long.py` (Destatis 41241 + Duden et al.).
3. Phenology: `phase0_phenology.py` (DWD). Fertiliser: `phase0_nitrogen.py`. Barley varieties: `phase0_barley_varieties.py`.
4. District table: `phase0_master.py`.
5. Satellite: `phase0_modis_districts.py`, `phase0_modis_features.py` (MODIS NDVI, Planetary Computer),
   `phase0_modis_lst.py` (land surface temperature), `phase0_dwd_soilmoisture.py` (DWD AMBAV),
   `phase0_s2_district_sample.py <state> <year> 25 cheap` (Sentinel-2 crop-specific district greenness, openEO
   account on Copernicus Data Space needed). `newdata_features.py` turns soil moisture / LST into model inputs.

## 2. District model (`02_district_model`)
1. `phase1_split.py` (train 1979-2017, test 2018-2025, held-out states 01 / 08 / 12), `phase1_baselines.py`,
   `phase1_wofost.py` (benchmarks).
2. `phase2_phenology.py <crop>` (crop calendar), `phase2_growth.py <crop>` (growth model + correction + blend);
   forward validation: `phase2_growth.py <crop> forward 2008 save`.
3. Set `VISTA_WEATHER=v8 VISTA_MODIS=1` for the current version. In-season forecasts: `phase2_forecast.py <crop>
   <blend weight>` (weights: wheat / barley 0.8, grain maize 0.2, silage 0.6, potato 0.4).
4. Barley refits: `VISTA_REFIT=2020` / `2023` with `phase2_growth.py winter_barley refit` and `phase2_forecast.py`.
5. v10: `build_v10.py`; 2026: `phase2_predict2026.py <crop>` (with `VISTA_WEATHER=v8_2026`), `make_pred2026_v10.py`.

## 3. Validation (`03_validation`)
Training-year method choices: `test_district_offsets.py`, `test_level_update.py`, `test_refit.py`,
`eval_ranges_v10.py`, `ranges_adaptive.py`, `stress_matrix.py`, `eval_step3.py`, `eval_select.py`,
`diagnose_errors.py`, `fusion_train.py` (needs `VISTA_FORWARD=2008` forecasts from `phase2_forecast.py`).
Test-year scoring (one look each): `score_v10_district.py`, `score_ranges_v10b.py`; final ranges:
`ranges_v11.py _v10`.

## 4. Benchmarks (`04_benchmarks`)
`phase0_destatis_inseason.py` (Fachserie 3.2.1 spreadsheets), `phase0_mars_bulletins.py` (MARS PDF text),
`phase0_mars_ocr.py` (+ `old` for 2012-2014; image tables, RapidOCR), `phase5_benchmarks.py` (comparison).

## 5. Platform (`05_platform`)
`export_platform.py` writes `docs/data/*.json`; `docs/index.html` is the page.

## 6. Field level (`06_field`), NRW
Field weather / soil, Sentinel-2 + PROSAIL LAI, particle-filter assimilation, YieldSAT tests,
`phase4_relative_maps.py` (within-field relative yield maps).

### Field v2: weekly 10 m green LAI and NDVI (4 NRW areas x 2021-2025)
Run from the data folder (`D:ista`) with Python 3.12; long runs detached (no console).
1. `areas.py` - the four 8 x 8 km areas (pilot, selhausen, north, south) and the seasons (1 Oct - 31 Aug).
2. `phase10_fields.py` - NRW field register per area-season, each 10 m pixel's area share inside its field.
3. `phase10_download.py` - Sentinel-2 L2A and Sentinel-1 (both orbits) from the Copernicus Data Space (openEO; ~17 credits
   per area-season); `phase10_landsat.py` - Landsat 8/9 C2 L2 from Microsoft Planetary Computer (free);
   `phase10_landsat_net.py` - PROSAIL network for Landsat.
4. `phase11_retrieval.py <area> <year> ...` (uses `fv2.py`) - green LAI, chlorophyll, NDVI per clear pixel-date.
5. `phase11_fusion_test.py` - season tests (leave one season out, whole dates hidden); `phase12_area_test.py rows | test |
   summary` - the deciding area test (new area, new year). Rules and results: `documentation/project_log.md`.
6. `fv2_produce.py train` - uncertainty table (`uncertainty_V2.json`) and production models (LAI, NDVI; live + re-analysis).
7. `fv2_weekly.py` - weekly pixel series per area-season (`weekly_pixels_V2.npz`): line for gaps <= 30 days, radar model
   for longer gaps, each value with its uncertainty and source.
8. `fv2_platform.py` - platform export to `docs/field/` (fields, weekly series, pixel blocks, map images, Sentinel-2
   true colour, `index.json`, `districts.json`). `run_platform_data.py` runs steps 6-8; every step resumes after a stop.

### Platform
`docs/index.html` holds the district and field level (`docs/js/field_view.js`, `my_fields.js`, `isoxml.js`,
`field_core.js`). `05_platform/build_site.py [src] [site]` copies `docs/` into the site's `crop-outlook/` after a dated
backup. Tests: `python -m pytest tests/field` and `node --test tests/js`.

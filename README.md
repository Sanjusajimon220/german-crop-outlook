# German Crop Outlook

Open, validated yield forecasts for German districts: winter wheat, winter barley, grain maize, silage maize and
potato, from twelve weeks before the harvest to the harvest, with 80 % ranges. A
[Groundtruth-earth](https://groundtruth-earth.netlify.app/) project: open data only, data gaps shown honestly,
reproducible code.

**Live platform:** https://sanjusajimon220.github.io/german-crop-outlook/ (served from `docs/`)

## What it does
- A daily crop growth model (light, temperature, soil water, nitrogen, heat, wetness, breeding progress) runs for
  every district from the sowing date; a learned correction and a gradient-boosting model refine it; satellite
  greenness (MODIS) enters from 2000.
- During the season the weather after the forecast date is replaced by 39 past seasons, which gives the forecast
  distribution.
- Model v10 adds an annual district level update learned from the model's own past errors, a national update for
  grain maize and a barley model refitted every three years. Ranges combine weather uncertainty with year-wide and
  local error; for barley and grain maize they adapt to recent misses.

## How good is it (test years 2018-2025, never used for fitting)
- District error v10 vs v9: wheat -11 to -18 %, barley -5 to -12 %, grain maize -4 to -7 %, silage maize -13 to
  -22 %, potato -7 to -17 % (known districts / three held-out states).
- Germany vs the JRC MARS forecast at the same date: v10 better in 15 of 22 month-crop comparisons (wheat May-July,
  potato May-September, both maize crops from August); behind for barley, wheat in August, maize in July,
  potato in October.
- vs the Destatis harvest reporters: better nationally for wheat (June estimate) and silage maize; the reporters
  remain better at state level.
- Full numbers, method and limitations: the platform's Accuracy and Method pages and
  [`documentation/model_card_v10.md`](documentation/model_card_v10.md).

## Repository layout (pipeline order)
| Folder | Content |
|---|---|
| `01_data` | download and preparation: DWD HYRAS weather, stations, CM SAF SARAH-3, soils, Destatis yields, phenology, fertiliser, MODIS greenness and land surface temperature, DWD soil moisture, Sentinel-2 district samples |
| `02_district_model` | train/test split, baselines, crop calendar, growth model, in-season forecasts, v10 / v11 assembly, 2026 predictions |
| `03_validation` | method choices on training years (ranges, stress tests, district offsets, refit schedule, fusion) and test-year scoring |
| `04_benchmarks` | Destatis Fachserie and JRC MARS Bulletin extraction (text and OCR) and the comparison |
| `05_platform` | data export for the web platform |
| `06_field` | field level (phase 2): field weather and soil, Sentinel-2 / PROSAIL LAI, assimilation, YieldSAT tests, within-field relative yield maps |
| `docs` | the web platform (GitHub Pages) |
| `documentation` | project log (every decision, pre-registration and result in order), model card, v11 plan, how to reproduce |
| `runs` | the shell scripts as they were executed during development (paths as on the development machine) |

Run any script from the repository root with `./run.sh <folder>/<script>.py [arguments]`; see
[`documentation/REPRODUCE.md`](documentation/REPRODUCE.md) for the order and the data sources.

## Data
No data are stored in this repository (sizes and licences). All inputs are open and are downloaded by the scripts in
`01_data` and `04_benchmarks`: DWD Climate Data Center, CM SAF, BGR, Destatis (regional database 41241, Fachserie
3.2.1), Duden et al. 2024 (OpenAgrar), JRC MARS Bulletins (JRC Publications Repository), NASA MODIS via Microsoft
Planetary Computer, ESA WorldCover, Copernicus Sentinel-2 / CLMS crop types via Copernicus Data Space (openEO).
Field-level tests used YieldSAT (CC BY-NC-ND, research only), which is not redistributed here.

## Validation discipline
Every design choice was made on the training years 1979-2017 (mostly in two separate forward periods), written into
the project log before the result was seen, frozen with file fingerprints, and then tested once on 2018-2025. The
frozen 2026 predictions will be checked against the official 2026 district yields in spring 2027.

## Licence
Code: MIT. Data remain under their providers' licences.

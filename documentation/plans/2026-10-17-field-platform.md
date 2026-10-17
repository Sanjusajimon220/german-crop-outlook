# Crop Outlook field level: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the NRW field level (weekly 10 m green LAI / NDVI with uncertainty, Sentinel-2 images, zones, exports) to the existing German Crop Outlook app, with every district feature unchanged.

**Architecture:** A Python pipeline (run from `D:\vista`, code kept in the repo under `06_field/`) turns the 20 area-seasons into weekly pixel series with the tested design (line <= 30 d, radar V2 > 30 d) and exports static platform files (field boundaries, weekly series, per-field pixel arrays, PNG overlays). The existing single-page app (`docs/index.html`, d3 v7, SVG map) gains a field view and a MY FIELDS tab; pure logic lives in an ES module (`docs/js/field_core.mjs`) tested with `node --test`.

**Tech Stack:** Python 3.12 (numpy, pandas, scikit-learn HistGradientBoosting, rasterio, geopandas, matplotlib, pytest), JavaScript (d3 7.8.5 from cdnjs, JSZip 3.10.1 from cdnjs for ISO-XML zips, node 22 for tests).

**Spec:** `documentation/specs/2026-10-17-field-platform-design.md` (approved 2026-10-17).

## Global Constraints

- Python: always `C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe`, run from `D:\vista` (data paths are relative `data/...`). Pipeline modules live in `D:\german-crop-outlook\06_field\` and are copied to `D:\vista` before running (as for all earlier steps).
- NRW only: areas `pilot`, `selhausen`, `north`, `south` (8 x 8 km), seasons 2021-2025 (past seasons, re-analysis). No live 2027 mode, no downloads.
- Every displayed value has an uncertainty and a source label (measured / straight line / radar-filled / after last clear image).
- No field yield anywhere; the field panel shows "Field yield: not shown yet (needs field yield data)".
- Fields are shown by register ID, crop and area only; no farm or owner names; no accounts; picked fields and zone rates are stored in the browser (`localStorage`, key `cropOutlook.myFields.v1`).
- District features unchanged and from the original files in `docs/data/` (5 crops, all map layers, panel, charts, ACCURACY, METHOD).
- Attribution on every field view and export: "Contains modified Copernicus Sentinel data [year]" and "Field boundaries: Land NRW (dl-de/by-2-0)".
- Large field data (`docs/field/`) is not committed (added to `.gitignore`); it is copied into the site build only.
- Deploy target: `D:\groundtruth-earth-site` (`crop-outlook/` + story page), deployed by the owner to Netlify.
- Commits only as "Sanju Sajimon <sanjusajimon76@gmail.com>", no co-author lines; the owner pushes.

## Review Focus

- A week with no value for a field (before the first clear image, or a field with no clear image all season) should show "no data" with the reason, never 0 or a blank chart → Task 2 test `test_no_clear_value_gives_missing`.
- Fields that straddle the 8 x 8 km area edge should not appear with a partial-area value as if complete → Task 3 test `test_edge_fields_flagged_partial`.
- Weeks before sowing / after harvest (previous crop or stubble) must be labelled, not presented as the crop → Task 3 test `test_off_season_label`.
- A browser that has never picked fields, or has corrupt `localStorage`, should open MY FIELDS with an empty list and a hint, not crash → Task 5 test `loadMyFields handles missing and corrupt storage`.
- Zone terciles on a field with very few or identical pixel values (tiny field, uniform LAI) must still give valid zones (fewer classes, message) → Task 5 test `zones with uniform values`.

---

### Task 1: Production fusion models and the uncertainty table

**Files:**
- Create: `06_field/fv2_produce.py`
- Modify: `06_field/phase11_fusion_test.py` (add `hide_share` to `Season.__init__`, default 0.3)
- Test: `tests/field/test_fv2_produce.py`

**Interfaces:**
- Consumes: `phase11_fusion_test.Season(area, year, target, landsat=False, date_hide=True, hide_share=0.3)`, `phase11_fusion_test.fit(X, y, rng, v)`, `phase11_fusion_test.pred(model, X)`, area-test parts in `data/processed/field_v2/area_parts/`, `area_test_summary.csv`.
- Produces:
  - `train_production(target: str) -> tuple[model_live, model_re]`: V2 models trained on all clear values (`hide_share=0`) of all 20 area-seasons, <= 250,000 rows per area-season and model; saved with joblib to `data/processed/field_v2/models/{target}_{live|re}_V2.joblib`.
  - `uncertainty_table() -> dict[str, dict[str, float]]`: keys `"lai"` / `"ndvi"`, inner keys `"line_le15"`, `"line_16_30"`, `"radar_gt30"`, `"live_le30"`, `"live_gt30"`; values = pooled RMSE over all 20 area-season parts (whole-date hiding). Saved as `data/processed/field_v2/uncertainty_V2.json`.
  - Constant `NDVI_MEASURED_SD = 0.02` (assumed sensor noise; labelled as assumed in the app).
  - `design_adopted(target: str) -> bool`: reads the RESULT lines of `area_test_summary_log.txt` ("DESIGN ADOPTED" / "NOT adopted"); if False, Task 2 uses the straight line for all gaps of that target and the table's `line` values.

- [ ] **Step 1: Write the failing tests** (plus `test_design_adopted_reads_result_lines` on a temporary log file with one ADOPTED and one NOT line)

```python
def test_hide_share_zero_hides_nothing():
    S = F.Season("pilot", 2024, "lai", landsat=False, date_hide=True, hide_share=0.0)
    assert not S.hide.any()

def test_uncertainty_table_keys_and_order():
    u = P.uncertainty_table()
    for t in ("lai", "ndvi"):
        assert set(u[t]) == {"line_le15", "line_16_30", "radar_gt30", "live_le30", "live_gt30"}
        assert all(0 < v < (2.0 if t == "lai" else 0.5) for v in u[t].values())
```

- [ ] **Step 2: Run** `python -m pytest tests/field/test_fv2_produce.py -v` from `D:\vista` with `PYTHONPATH=D:\german-crop-outlook\06_field`. Expected: FAIL (`hide_share` unknown, module missing).
- [ ] **Step 3: Implement** `hide_share` (in the date-hide branch: number of hidden dates = `round(hide_share * T2)`, 0 → no hiding), then `uncertainty_table()` (read all `area_parts/*.csv.gz`, RMSE of `reanalysis` for gaps > 30 d, of `line` for <= 15 and 16-30 d, of `live` split at 30 d) and `train_production()`.
- [ ] **Step 4: Run** the tests. Expected: PASS. Then run `python fv2_produce.py train` (after the area test has finished) and check the two joblib files and `uncertainty_V2.json` exist.
- [ ] **Step 5: Commit** `06_field/fv2_produce.py`, `06_field/phase11_fusion_test.py`, `tests/field/test_fv2_produce.py`.

### Task 2: Weekly pixel series per area-season (the design)

**Files:**
- Create: `06_field/fv2_weekly.py`
- Test: `tests/field/test_fv2_weekly.py`

**Interfaces:**
- Consumes: Task 1 models and `uncertainty_table()`, `Season` (with `hide_share=0`), `s2_pixels_fv2.npz` (`lai_sd` for measured LAI).
- Produces: `weekly_pixels(area: str, year: int) -> dict` with keys `weeks` (datetime64[D], Mondays 1 Oct (prev year) - 31 Aug), `lai`, `ndvi` (float32 [weeks x pixels], NaN = no data), `lai_unc`, `ndvi_unc` (float32), `src` (int8: 0 none, 1 measured, 2 line, 3 radar, 4 after last clear image), `gap` (int16 days), `rows`, `cols`; saved as `data/processed/field_v2/{area}_{year}/weekly_pixels_V2.npz`. Rules: measured = clear S2 value within ±3 days of the Monday (nearest); gap <= 30 d → line; gap > 30 d → radar V2 re-analysis (line instead when `design_adopted(target)` is False); only a previous clear value → live V2 (`src` 4); LAI clipped >= 0, NDVI to [-1, 1].

- [ ] **Step 1: Write the failing tests** `test_measured_week_keeps_s2_value` (a Monday 2 days from a clear date returns that pixel's value, `src == 1`, `lai_unc == lai_sd`), `test_gap_classes` (a 20-day gap → `src == 2` and value on the straight line; a 40-day gap → `src == 3`), `test_no_clear_value_gives_missing` (pixel with no clear value → all NaN, `src == 0`), using small synthetic arrays passed to a pure helper `assign_week(days, values, sd, week_day, models=None) -> (value, unc, src, gap)`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `assign_week` (pure) and `weekly_pixels` (vectorised over pixels per anchor pair, as in `phase8d_weekly_series.py`).
- [ ] **Step 4: Run** the tests: PASS. Run `python fv2_weekly.py all` (20 area-seasons, resumable: skip existing files; start outside the app via WMI as the other long runs). Check 20 files exist and the source shares are printed per area-season.
- [ ] **Step 5: Commit** `06_field/fv2_weekly.py` and its test.

### Task 3: Platform files for the field level

**Files:**
- Create: `06_field/fv2_platform.py`
- Modify: `.gitignore` (add `docs/field/`)
- Test: `tests/field/test_fv2_platform.py`

**Interfaces:**
- Consumes: Task 2 npz, `pixels_nrw.csv.gz`, `fields_nrw.csv`, `fields_rur_allcrops.gpkg`, raw S2 GeoTIFFs, `docs/data/meta.json` (district names).
- Produces (under `docs/field/`):
  - `index.json`: `{areas: {area: {bounds_wgs84: [w,s,e,n], seasons: [2021..2025], districts: [codes]}}}`
  - `{area}_{year}/fields.geojson` (WGS84, 1 m simplification): properties `id`, `crop` (NRW text), `group`, `ha`, `partial` (bool: < 99 % of the field inside the area), `district`.
  - `{area}_{year}/series.json`: `{weeks: ["YYYY-MM-DD"...], fields: {id: {lai:[], ndvi:[], lai_unc:[], ndvi_unc:[], src:[], season:[0|1], alert:[0|1|2]}}}` (field = area-weighted pixel mean; `season` 0 = before sowing / after harvest; `alert` 0 none, 1 below crop median, 2 falling while median rises).
  - `{area}_{year}/px/{id}.json`: `{rows, cols, w (share), lai: [weeks][pixels], ndvi: [weeks][pixels]}` rounded to 2 decimals.
  - `{area}_{year}/lai/{YYYY-MM-DD}.png`, `ndvi/{YYYY-MM-DD}.png`: 800 x 800 RGBA overlays (LAI colour scale `d3.interpolateYlGn`-equivalent 0-5, NDVI 0-1; transparent outside fields); `rgb/{YYYY-MM-DD}.png` true colour for every S2 date with >= 20 % clear field pixels, and `rgb.json` listing those dates.
  - `districts.json`: `{year: {code: {weeks: [...], n_fields: [...], median_lai: {group: [...]}}}}` for the districts touched by the areas.
- Rules: alerts as in the spec (difference to the area median of the same crop group > the field's uncertainty for 2 consecutive weeks), only for arable groups (wheat, barley, other cereal, rapeseed, maize, potato, sugar beet); grassland and other land always `alert` 0; `season` = 0 for weeks before the first week the field's LAI rises above 0.5 after its minimum in autumn for winter crops / before 1 April for spring crops, and after the last week above 0.5 at the season end.

- [ ] **Step 1: Write the failing tests** `test_edge_fields_flagged_partial`, `test_off_season_label`, `test_alert_needs_two_weeks_beyond_uncertainty` (synthetic series: one week below by more than uncertainty → no alert; two weeks → alert 1), `test_series_matches_pixel_mean` (field value = share-weighted mean of its pixels).
- [ ] **Step 2: Run**: FAIL.
- [ ] **Step 3: Implement** pure helpers `alerts(field_vals, field_unc, median_vals) -> list[int]`, `season_mask(lai_series, group) -> list[int]`, then the exporter `export(area, year)` and `main()` (all 20, then `index.json` and `districts.json`).
- [ ] **Step 4: Run** tests: PASS. Run the export; check `docs/field/` size (target < 2 GB) and spot-check one field's series against `weekly_pixels_V2.npz`.
- [ ] **Step 5: Commit** `06_field/fv2_platform.py`, `.gitignore`, test.

### Task 4: Field view in EXPLORER (zoom from districts to fields)

**Files:**
- Create: `docs/js/field_core.mjs`, `docs/js/field_view.js`, `tests/js/field_core.test.mjs`
- Modify: `docs/index.html` (map container: d3.zoom; layer menu: field layers; panel switch; one district-panel line)

**Interfaces:**
- Consumes: Task 3 files.
- Produces (`field_core.mjs`, pure): `weekIndex(weeks: string[], date: string) -> number` (nearest Monday <= date), `sourceLabel(src: number) -> string` ("measured" / "straight line" / "radar-filled" / "after last clear image" / "no data"), `withinNoise(diff: number, unc: number) -> boolean` (|diff| < unc), `nearestRgbDate(dates: string[], week: string) -> string|null` (closest clear image within ±10 days), `fmtUnc(v, u, d) -> string` ("3.0 ± 0.4").
- Behaviour: one date slider for both levels: in district view it keeps "weeks before harvest" (`LEADS`), in field view it steps through the season's calendar weeks (label "Week of 27 May 2024 · re-analysis (past season)"); uncertainty shown as "± u (typical error of a 10 m value)" and NDVI on measured weeks as "± 0.02 (assumed sensor noise)"; districts and all existing layers unchanged at zoom 1; the four study areas drawn as outlined boxes; zoom > 4 over an area loads its `fields.geojson`, `series.json` and the week's overlay PNG (positioned by `bounds_wgs84` through the same `geoMercator`); field click → panel (LAI and NDVI `fmtUnc`, source label, weekly curve in the existing `fanChart` style with the band, vs median field, district context from `rowFor(district)`, true-colour image of `nearestRgbDate` with its date, "Field yield: not shown yet (needs field yield data)", attribution). District panel gains: "Field greenness this week: N fields, median green LAI x" (only districts in `districts.json`).

- [ ] **Step 1: Write failing node tests** for each `field_core.mjs` function with exact values (e.g. `sourceLabel(3) === "radar-filled"`, `withinNoise(0.2, 0.4) === true`, `nearestRgbDate(["2024-05-01","2024-06-25"], "2024-05-27") === null`, `... "2024-05-06" ) === "2024-05-01"`).
- [ ] **Step 2: Run** `node --test tests/js/`: FAIL.
- [ ] **Step 3: Implement** `field_core.mjs`, then `field_view.js` and the `index.html` changes.
- [ ] **Step 4: Run** node tests: PASS. Serve `docs/` with a local server (`.claude/launch.json`, `python -m http.server 8090 --directory docs`), check in the browser pane: every district layer for every crop still renders as before (screenshot each layer once for wheat); zoom into the pilot area, click field 7493815 in 2024 week 2024-05-27 → LAI 3.0 ± 0.4 (radar-filled), image of 2024-05-01; no console errors.
- [ ] **Step 5: Commit** the JS, tests and `index.html`.

### Task 5: MY FIELDS tab (pick on map, list + map, zones, exports)

**Files:**
- Modify: `docs/js/field_core.mjs`, `docs/index.html` (new tab)
- Create: `docs/js/my_fields.js`, `docs/js/isoxml.mjs`
- Test: `tests/js/my_fields.test.mjs`, `tests/js/isoxml.test.mjs`

**Interfaces:**
- Consumes: Task 3 files, Task 4 `field_core.mjs`.
- Produces: `loadMyFields(storage) -> {fields: [{area, year, id}], rates: {id: [weak, mid, strong]}}` (empty object on missing or corrupt JSON), `saveMyFields(storage, state)`, `zones(values: number[], weights: number[]) -> {classes: Int8Array, cuts: [number, number], n: 1|2|3}` (weighted terciles of the period mean; fewer classes when values are identical or < 9 pixels), `buildTaskData({field, zones, rates, grid}) -> {xml: string, bin: Uint8Array}` (ISO 11783-10 `TASKDATA.XML` with one task, one product, treatment zones, grid type 2, rates in kg/ha as integer mg/m² per the standard's DDI 6 unit) in `isoxml.mjs`; shapefile export = GeoJSON → zip via JSZip containing `.geojson` + `.prj` (WGS84) [shapefile writer left to the builder if a small vetted library is available on cdnjs]; weekly report = print stylesheet (`@media print`) of the field panel.
- Behaviour: zones, rates and exports only for arable fields (grassland: list row and LAI only, message "zones not offered for grassland"); "Add to my fields" button in the field panel; MY FIELDS lists picked fields sorted by alert then ID (columns: field ID, crop, green LAI ± u, change since last week with "within noise", vs median, source, alert); selecting a field shows true-colour image, 10 m LAI for the week (canvas from `px/{id}.json`), zones for a period (default 1 Apr - 30 Jun, two date inputs), editable rates per zone (default empty; export disabled until all rates are set), buttons: Export ISO-XML (zip), Export shapefile, Print report.

- [ ] **Step 1: Write failing tests**: `loadMyFields handles missing and corrupt storage`, `zones with uniform values` (n === 1), `zones terciles` (9 values 1..9 → cuts [3.67, 6.33] ± 0.01 with equal weights), `buildTaskData` (xml contains `<TSK`, `<PDT`, `<GRD` with `J="2"`, bin length = rows × cols × 4).
- [ ] **Step 2: Run** `node --test tests/js/`: FAIL.
- [ ] **Step 3: Implement** the functions and the tab.
- [ ] **Step 4: Run** tests: PASS. Browser check: pick 3 fields, reload (still picked), set rates, export ISO-XML and open `TASKDATA.XML` in an ISO-XML viewer (e.g. the open-source "ISOXML Viewer" web app) without errors; clear `localStorage` → empty list with hint.
- [ ] **Step 5: Commit**.

### Task 6: Old field page, build into the Groundtruth-earth site, deploy

**Files:**
- Modify: `docs/fields/index.html` (banner "Field yields in this page are not validated; superseded by the field level in the main app" and hide t/ha values) or delete the page and its link (owner chooses at review; default: banner + hide).
- Create: `05_platform/build_site.py`: `build_site(src="docs", site="D:/groundtruth-earth-site") -> dict` copies `docs/` incl. `docs/field/` into `site/crop-outlook/` (replacing it, after a dated backup to `D:/groundtruth-earth-site-backup-YYYYMMDD/crop-outlook/`), writes `site/_headers` rules for long cache on `/crop-outlook/field/*`, returns file count and size.
- Modify: `D:/groundtruth-earth-site/crop-outlook-germany.html` (one new section "Field level (NRW pilot)": what it shows, honest limits, link into the app's field view).
- Create: `documentation/REPRODUCE.md` section "Field level".
- Test: `tests/field/test_build_site.py` (on a temporary site folder: `crop-outlook/index.html`, `crop-outlook/data/meta.json`, `crop-outlook/field/index.json` copied; backup made; `_headers` contains `/crop-outlook/field/*`).

- [ ] **Step 1: Write the failing test**. **Step 2: Run**: FAIL. **Step 3: Implement** `build_site`; if the field data exceeds ~200 MB or ~5,000 files, pack the per-field pixel files into one file per area-season (index + byte ranges) so the drag-and-drop deploy stays practical.
- [ ] **Step 4: Run**: PASS; build into the real site folder; open `D:/groundtruth-earth-site/crop-outlook/index.html` through a local server and repeat the Task 4/5 browser checks; the owner deploys the site (drag and drop or Netlify CLI) and checks it on a phone and a desktop.
- [ ] **Step 5: Commit**; update `documentation/project_log.md` (via `D:ista\docs\project_log.md` copy) with the release entry.

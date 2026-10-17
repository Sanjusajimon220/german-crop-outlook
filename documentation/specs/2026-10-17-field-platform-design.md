# Crop Outlook with field level: platform design

Date: 2026-10-17 · Owner: Sanju Sajimon · Status: draft for owner approval

## Intent and the why
The German Crop Outlook (district yield forecasts, live at
`groundtruth-earth.netlify.app/crop-outlook/`) is tested and honest, but it speaks to readers, not to farmers.
The owner's ultimate aim is the field level: a professional product that farmers and agronomists in NRW use to
decide **fertiliser and spraying** (when, where in the field, how much), competitive with xarvio FIELD MANAGER,
Planet Crop Biomass, EOS, OneSoil (market check: Claude Doc "Satellite crop monitoring: market and licence check").

One platform, two zoom levels:
- **District level (exists):** yield forecasts per district, 5 crops, 12 weeks before harvest to harvest, 80 % ranges,
  accuracy against JRC MARS and Destatis.
- **Field level (new):** weekly green LAI and NDVI for every 10 m pixel of every registered field, each value with a
  tested uncertainty and its source (measured / straight line / radar-filled), zones for fertiliser, exports.
- **Field yield and water: later**, only once field yield data validates them. Until then the platform shows
  "field yield: not shown yet".

Our edge (what the market does not offer): a physical green LAI (m2/m2), not a relative index; an uncertainty on
every value, measured on unseen seasons and areas; free inputs only.

## Users
- Farmers in NRW (one farm, 5-50 fields).
- Agronomists / advisors (many farms).
Their decision: fertiliser and spraying. The district outlook gives regional context.

## Goals (version 1)
1. Extend the existing Crop Outlook app, keeping its look, controls, EXPLORER / ACCURACY / METHOD tabs and the
   district map unchanged (approved sketch: `D:\vista\.superpowers\brainstorm\1487-1792226390\content\platform-v3.html`).
2. EXPLORER: zooming in on NRW switches from districts to fields (exact registered boundaries) and then to 10 m
   pixels. New map layers: green LAI, NDVI, change since last week, data source, uncertainty, zones. Field panel:
   LAI and NDVI with uncertainty, source label, weekly curve with band, comparison with the median field of the same
   crop in the area, district forecast as context, "field yield: not shown yet".
3. District panel gains one line: field greenness in this district this week (fields monitored, median green LAI).
4. MY FIELDS tab (list + map, owner's choice; fields picked on the map, no accounts): ranked field list with alerts (below the area median for the crop,
   falling LAI; see calls below), selected field with Sentinel-2 image, 10 m LAI and 3 zones,
   editable rate per zone, exports: ISO-XML (task data for terminals), shapefile, weekly PDF report.
5. Real data, NRW only: the four study areas (pilot, selhausen, north, south; 8 x 8 km each), seasons 2021-2025
   (historic, re-analysis), all crops in the field register.
6. Every value carries the uncertainty measured in the fair tests (whole dates hidden; area test for the final
   numbers), per target, crop group where available, and gap class.

## Non-goals (version 1)
- Field yield, water use, soil moisture, nitrogen advice in kg (the platform supplies zones; rates are entered by
  the user).
- Areas outside the four study areas, states other than NRW, Germany-wide processing.
- Live in-season operation for the 2027 season (planned as version 2; see constraints).
- Native mobile apps; payments; multi-language (English first).

## Anti-goals (failure even if it works)
- Showing a number without its uncertainty or source.
- Showing field yields that are not validated (the current `docs/fields` "Rur Field Outlook" page shows field
  t/ha from Field Outlook v1; it must be removed or clearly labelled before the new platform goes public).
- Changing district results or the district test protocol.
- Using data whose licence forbids a paid product (YieldSAT, Google Earth Engine, Planet) in the product.

## Method decisions already made (from tests; see docs/project_log.md, 16-17 Oct 2026)
- Retrieval: Sentinel-2 green LAI from our PROSAIL network (10 bands + angles), NDVI from B04/B08; cloud mask SCL 4/5
  with 2-pixel buffer; a field counts as measured on a mostly cloudy image if >= 50 % of it is clear.
- Gap filling ("the design"): straight line between clear values for gaps <= 30 days; radar re-analysis model V2
  (Sentinel-1 both orbits, speckle filter, field-level radar change vs both anchors, anchors, gap, position, crop;
  no calendar, heading or rain) for gaps > 30 days. Physics is not used in the fusion.
- Landsat: not used in version 1 (cross-check passed for LAI, but it did not improve the fused series in every
  season; a filtered variant may be tested later).
- No smoothing (tested: less accurate).
- Weekly Monday grid; field value = area-weighted mean of its 10 m pixels (exact registered boundary shares).
- Uncertainty per value = RMSE from the fair tests for its source and gap class (final numbers from the area test).
- Zones: 3 classes (terciles within the field) of the mean green LAI over a chosen period (default April-June);
  user can change the period.

## Constraints
- Free data only: Copernicus Sentinel (attribution "Contains modified Copernicus Sentinel data [year]"), DWD
  (CC BY 4.0), NRW field register (Datenlizenz Deutschland - Namensnennung 2.0, to be confirmed; credit "Land NRW").
- Compute: one Windows laptop (7.7 GB RAM, no GPU); long runs must be resumable and started outside the app.
- Copernicus Data Space credits: 342 used of the free allowance; any new download needs the owner's OK
  (file, source, size, credits).
- Hosting today: static site on Netlify (district app) and GitHub Pages (`docs/`). Field data must be served as
  static files (per area-season, per week) (v1 has no backend; see owner decisions).
- Commits in the repo only as "Sanju Sajimon"; the owner pushes.

## Owner decisions (2026-10-17)
- **Approved** by the owner on 2026-10-17.
- **Click behaviour:** clicking a district shows its results exactly as in today's app (forecast, 80 % range,
  through-the-season chart, official yield); clicking a field shows its real data (green LAI and NDVI with
  uncertainty and source, weekly curve) and its real Sentinel-2 true-colour image for the chosen week (nearest clear
  image, with its date).
- **District features unchanged:** every existing district capability stays exactly as today, from the original
  data files in `docs/data/` (all 5 crops, all map layers: forecast, official, forecast vs official, range width,
  water supply / use / demand, NDVI; the district panel, through-the-season chart, ACCURACY and METHOD tabs). The
  field level reuses the same visual format (panel layout, chart style, colour scales).
- **My Fields without accounts:** the user picks fields on the map in the app; the selection (and rates per zone) is
  kept in the user's browser. Fields are shown by their register name / ID, crop and area only; no farm or owner
  names. Accounts, log-in and server-side storage come later (the LOG IN tab is not built in v1).
- **Time period:** past seasons 2021-2025 only (re-analysis), as planned; no live 2027 operation in v1.
- **Code:** our git repository `german-crop-outlook` is the master; the Netlify site is deployed from it.
- **Deployment target:** the owner's Groundtruth-earth site (`D:\groundtruth-earth-site`, Netlify): the combined app
  replaces `crop-outlook/` (same address `groundtruth-earth.netlify.app/crop-outlook/`), and the story page
  `crop-outlook-germany.html` gains a short field-level section linking into it.

## Calls made by Claude (owner may override)
- **Release:** built and tested as a private preview; it goes public only after the NRW field-register licence is
  confirmed (contact: Landwirtschaftskammer NRW, opendata@lwk.nrw.de). The old `docs/fields` page with field t/ha is
  removed or labelled "not validated" before that.
- **Uncertainty:** each pixel value carries the pixel-level RMSE from the fair tests for its source and gap class
  (one pooled table over areas and seasons, per target; final numbers from the area test); clear-day values carry the
  PROSAIL network's own per-pixel standard deviation. The field value shows the same pixel-level number, labelled
  "typical error of a 10 m value" (the field mean is at most this uncertain). No per-crop split in v1.
- **Derived numbers** (change since last week, difference to the median field, district-panel median): shown with
  the text "within noise" when smaller than the combined uncertainty.
- **Alerts:** a field is flagged only when it differs from the median of fields of the same crop in its study area by
  more than its uncertainty for 2 consecutive weeks (below median; falling while the median rises). "Ageing early" is
  left out of v1 (no tested measure).
- **Crops:** arable crops (wheat, barley, rapeseed, maize, potato, sugar beet, other cereals) get LAI, NDVI, alerts and
  zones; grassland and other land show LAI / NDVI only; weeks before sowing / after harvest are labelled.
- **Time control in the field view:** calendar weeks of the chosen season; the district view keeps "weeks before
  harvest"; both follow the same date slider. Field values are labelled "re-analysis (past season)".
- **Data storage:** field data is published as static files per area-season; large binary layers are kept out of the
  git history (published with the site build, not committed), so the repository stays small.
- **ISO-XML:** ISO 11783-10 task data with one treatment-zone grid (one product, kg/ha), checked with an open
  ISO-XML viewer; a test on a real terminal by the owner later.
- **If the area test fails its rule** for a target, that target uses the straight line for all gaps, and the
  uncertainty table is the line's.

## Left to the builder
- Web map technology for 10 m pixels (e.g. pre-rendered PNG tiles per week vs vector field polygons + small rasters),
  file formats and compression, as long as the first view loads in < 3 s on a normal connection.
- Code structure of the new pipeline step that turns the weekly pixel series into platform files.
- Chart styling within the existing app's look.
- ISO-XML details (ISO 11783-10 task data with a treatment zone grid) and PDF layout.

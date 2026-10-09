# v11 plan (draft pre-registration, 2026-10-09; written after the v10 test look, before any v11 result)

Goal (unchanged): before harvest, at every forecast date, crop and level (national, state, district), more
accurate on average than JRC MARS, the Destatis harvest reporters and the simple rules, never clearly worse,
with 80 % ranges that hold.

Open gaps after v10 (test 2018-2025): barley nationally (May, Jul, Aug), wheat in August, both maize crops
in July, potato in October, state level vs the reporters, ranges at state / national level.

Important: the v10 test years 2018-2025 have now been looked at. v11 choices must therefore be made on
training years (<= 2017) where possible; where a component only exists from 2017 (Sentinel-2), the
evaluation is rolling (each year predicted only from earlier years) and the clean test is 2026 (spring
2027) and later seasons. Every result is reported with this caveat.

## A. Fusion with published official forecasts
- When a MARS bulletin or a Destatis estimate is published, combine it with our forecast of the same date:
  combined = w * ours + (1 - w) * official, per crop and month (national) or per crop and issue (state).
- Weights from training years only: our in-season forward forecasts 2009-2017 (models fitted <= 2008,
  same scenario method; to be produced) against MARS 2009-2017 and Destatis 2009-2017 at matching dates;
  w = inverse-error-variance weights with the error covariance (Bates & Granger), shrunk towards 0.5.
- District level: the national / state combination shifts the district forecasts proportionally
  (district shape from our model, level from the combination).
- Adopted if it lowers training-year error in both halves (2009-2012, 2013-2017) for the crop-month.

## B. Crop-specific Sentinel-2 greenness (2017+)
- Data: district x crop x dekad NDVI from sampled interior cells (CLMS crop types, 25 cells, Apr-Sep).
- Route 1 (bridge): S2 crop NDVI -> MODIS cropland NDVI mapping per district from overlap years
  (satellite vs satellite only, no yields) so the model can run on S2 when MODIS ends.
- Route 2 (new information): features per crop (peak, peak date, area under curve, decline rate, anomaly)
  up to the forecast date; evaluated by rolling origin 2019-2025 (train on 2017..t-1) and held-out states.
- Missing crop maps 2024-2026: DLR CropTypes (2024); own crop classification from S2 time series (2025+).

## C. Adaptive ranges
- Ranges calibrated online: for year t, scale the year-wide and local error parts so that the coverage of
  the previous k years (k = 5, known by t) is 80 % (conformal-style); evaluated by rolling origin on
  2018-2025 (each year only from earlier years) and on 2026+.

## D. Possibly v10.1: MODIS canopy temperature (if it passes the two-period rule on training years).

## Order
1. Produce training-year in-season forecasts 2009-2017 (needed for A and C). 2. Fusion weights (A).
3. S2 Germany 2019-2023 (running) -> features, bridge (B). 4. Adaptive ranges (C). 5. Combined v11,
freeze, evaluation as above, report.

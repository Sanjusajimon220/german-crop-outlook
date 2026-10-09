# Model card: German district crop yield model, v10 (frozen 2026-10-09)

## What it does
Forecasts the yield (t/ha) of winter wheat, winter barley, grain maize, silage maize and potato for every
German district (Landkreis), with state and national aggregates, at six dates in the season (12, 8, 6, 4, 2,
0 weeks before the typical harvest). Each forecast comes with an 80 % range and the probability of a yield
below the district's previous 5-year average.

## How it works (short)
1. **Crop calendar** (phenology model fitted to DWD phenology observations) gives sowing, emergence,
   flowering / heading and maturity per district and year.
2. **Growth model** (daily, physics-based): light interception, temperature, soil water from the BK/BUEK
   soil and HYRAS weather (v8: SARAH-3 sunlight from 2021), nitrogen supply, heat and wetness effects,
   breeding progress.
3. **Learned correction** (gradient boosting on the log ratio observed / simulated) using weather,
   simulated quantities and MODIS satellite greenness (2000+), blended with a purely data-driven boosting
   model (crop-specific weights chosen by forward validation).
4. **In-season**: weather after the forecast date is replaced by the 39 past seasons 1979-2017 (scenarios).
5. **v10 additions**: annual district level update (offsets learned from the model's own errors of
   years up to t-2, shrunk), national 3-year update for grain maize, barley model refitted every 3 years;
   ranges with a shared year-wide error learned from 27 forward years plus a local error.

## Data
DWD HYRAS-DE (temperature, rain, humidity), DWD stations (wind), CM SAF SARAH-3 (sunlight), BGR soils,
DWD phenology, Destatis district yields (regional database 41241) and Duden et al. 2024 (1979-2021),
fertiliser statistics, MODIS MOD13Q1 NDVI over ESA WorldCover cropland.

## How it was tested
- **Split**: training 1979-2017; test 2018-2025 (new years) and three held-out states (Schleswig-Holstein,
  Baden-Wuerttemberg, Brandenburg: new regions).
- **Every design choice** made on training years only, by forward validation (fit up to a year, predict the
  following years), most v10 choices in two separate periods (2003-2008 and 2009-2017); rules written into
  the project log before results were seen; one look at the test years after freezing (md5 fingerprints).
- **Benchmarks**: JRC MARS Bulletin forecasts and Destatis harvest-reporter estimates at the same date,
  last year's yield, 5-year average.
- **Final clean check**: frozen 2026 predictions vs official 2026 district yields (spring 2027).

## Results (test years 2018-2025, one look 2026-10-09)
**District accuracy vs v9** (typical error, all six forecast dates; known districts / held-out states):
wheat -11 to -13 % / -16 to -18 %; barley -5 to -12 % / -8 to -10 %; grain maize -4 to -6 % / -4 to -7 %;
silage maize -13 to -20 % / -15 to -22 %; potato -7 to -9 % / -14 to -17 %. Better in every cell.

**National, vs JRC MARS** (error in % of mean yield, v10 / MARS): wins 15 of 22 month-crop cells: wheat
May-July (e.g. June 3.9 / 7.0), potato May-September, grain and silage maize from August. Behind: barley
May, July, August (June 7.7 / 7.5), wheat August, both maize crops July, potato October.

**vs Destatis harvest reporters:** national wins for wheat (June estimate 4.2 / 6.0 %) and silage maize;
state level the reporters are better (wheat 8.1 / 7.8 %, silage, potato).

**vs simple rules** (last year, 5-year mean): better at every crop, level and date.

**80 % ranges:** district coverage 58-81 % (silage maize meets 80 %), state and national 40-75 %: too
narrow. Probability of a below-normal yield: clearly better than the base rate nationally for wheat and
both maize crops.

**2026:** frozen predictions (data/processed/check2026/*_v10.csv, md5 in frozen_2026_v10_md5.txt) are
checked against official 2026 district yields in spring 2027.

## Known limitations
- Extreme drought years: the model is still somewhat too optimistic in dry, hot flowering periods,
  above all for maize; soil moisture (DWD AMBAV) and drought / heat physics did not give robust gains.
- Barley before harvest: MARS was better with v9; v10 adds refits for fast breeding progress.
- State level: Destatis harvest reporters (observing the crop) are better than v9 at state level.
- Ranges: district and state ranges too narrow in v9; v10 recalibrated, see results.
- Grain maize district yields only to 2021 in our data (fewer test years).
- MODIS satellites will retire; a Sentinel-2 bridge is in development.
- Silage maize from 2010 and potato statistics are noisier.

## Not suitable for
Single fields (see the field-level work), crops not listed, regions outside Germany, years without
current weather data.

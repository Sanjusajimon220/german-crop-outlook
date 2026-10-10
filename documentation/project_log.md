# Project log: decisions, results and lessons

Newest phase last. Numbers are test-data errors unless stated otherwise.

## Background and prototype (Sep-Oct 2026)
- PROMET: Vista GmbH / LMU Munich process model (hourly, 1 km, water + energy balance, crop
  growth). Vista's satellite retrieval is SLC. Code is closed.
- Prototype (deleted from D:\vista at the fresh start): land model, PROSAIL-trained Sentinel-2
  retrieval network with clumping, SL2P reimplementation (matched all 41,452 ESA test cases),
  ICOS validation at Selhausen (DE-RuS) and Gebesee (DE-Geb). Retrieval reached parity with SL2P
  on 20 test points (0.94 vs 0.97 LAI RMSE). Lessons: ground PAI after heading is not green LAI;
  sugar beet clumping; too little data to prove anything -> restart at national scale.

## Literature: PROMET's published accuracy
- Hank et al. 2015, Remote Sensing 7:3934 - wheat, field/pixel level with satellite assimilation
  (ensemble, best-matching run), yield RMSE 1.15 t/ha avg (1.29 in 2010, 1.59 in 2011).
- Brandenburg, Reckleben & Griepentrog 2024, Precision Agriculture - Schleswig-Holstein farms
  2019-2022: whole-field grain R2 ~0.91; pixel wheat R2 0.38, RMSE 2.4 t/ha; silage maize
  R2 0.66, RMSE 4.8 t/ha.
- No published PROMET district-level, multi-year, unseen-year or drought test found.

## Phase 0: data (done)
- Yields: Regionaldatenbank 41241-01-03-4 (1999-2025) + Duden, Nacke & Offermann 2024 Sci Data
  11:95 (1979-2021, harmonised to 2020 districts, includes grain maize). 99-100 % identical in
  overlap. Combined: data/processed/yields_long.csv (147,892 values, 11 crops).
- Weather: HYRAS 1 km daily (temp, humidity, precip; radiation 5 km to 2020) averaged per
  district; station wind and sunshine (Angstrom a=0.211 b=0.547, daily RMSE 10 W/m2 vs HYRAS);
  FAO-56 ET0 per district latitude. weather_daily.npz: 17,167 days x 400 districts, complete.
- Soils: DWD AMBAV field capacity / wilting point, 20 layers of 10 cm = 0-2 m. Layer thickness
  checked 2026-10-04: DWD's AMBAV soil-moisture product reports exactly these 10 cm layers
  (BFGL01..06 = 0-10 ... 50-60 cm; dataset description v2, 2024); the grids show the plough layer
  as 3 identical layers (0-30 cm); median plant-available water 186 mm in 1 m (grid), which is
  typical for German soils. Not stated in the file itself, so 'confirmed by consistency'.
- Crop stages: DWD phenology (historical + recent files), 1.8 M observations; autumn stages
  shifted to harvest year; stages after day 250 of winter crops dropped.
- Fields: NRW Chamber of Agriculture subsidy records 2021-2026, 1.29 M field-years of our crops;
  field areas match official crop areas within ~5 %.
- Findings: wheat yields stagnant since ~2000 (straight-line trend over 1979-2017 overshoots);
  2018 drought -11..-18 % for all crops.

## Phase 1: benchmarks (done) - bar to beat, RMSE t/ha (new years / new regions / drought years)
- Winter wheat 0.96 / 1.15 / 1.01 (gradient boosting)
- Winter barley 0.97 / 1.03 / 0.89 (WOFOST + weather, boosting)
- Grain maize 1.66 / 1.29 / 2.06
- Silage maize 5.59 / 6.71 / 6.96
- Potato 8.09 / 5.42 / 8.17
- WOFOST (PCSE, Wofost72_WLP_CWB, development speed calibrated) helps mainly for barley and
  maize in droughts; its year-to-year signal correlates only 0.18-0.34 with real deviations.
- Lesson: pure physics loses to machine learning at district level -> hybrid design.

## Phase 2: crop calendar
- PyTorch model: temperature response (Wang-Engel), vernalisation, day length, separate settings
  before/after heading, soft (differentiable) stage crossing; fitted on <= 5,000 training seasons.
- Version 3 adds a variety trend (thresholds x exp(trend x decades since 2000)). Wheat:
  -3.8 %/decade to heading, -3.1 % after (breeding: earlier varieties).
- Wheat results (days, new years): heading 5.8 (WOFOST 6.0, simple forecast 13.0),
  yellow ripeness 7.0 (WOFOST 8.9). Barley beat WOFOST by 1.5-2.5 days (version 2).
- Version 3 for maize (flowering, new years): grain maize 5.6 days (WOFOST 7.6), silage maize 6.6
  (WOFOST 6.9). Maize variety trend is positive (+3.0 / +1.7 %/decade): later-maturing hybrids
  chosen as summers warmed - the opposite of wheat/barley breeding for earliness.
- Boosting benchmark for stages (only weather of months before the stage): wheat heading 5.6,
  ripeness 6.7; average of boosting + calendar best or tied almost everywhere (heading 5.5,
  ripeness 6.5) -> keep physics calendar inside the crop model, add learned correction on top.

## Plans and ideas
- Growth/yield module: light capture, biomass (RUE first, two-leaf later), partitioning by stage,
  leaf ageing, layered soil water with root growth, heat at flowering, water demand outputs.
- Field level: rebuild PROSAIL retrieval (green LAI, brown fraction, chlorophyll, water), S1
  Water Cloud Model, ensemble Kalman filter; test region Rur (Dueren, Aachen, Euskirchen,
  Heinsberg, Rhein-Erft); field yield data (yield maps) still needed - ask KIT/partners.
- CropPulse (owner's notebook): GEE-based S1->NDVI gap filling, SL2P LAI, classification with
  spatial CV. To adapt: per-field/year CV, per-field fill of lags, +-1-2 day matching, rain and
  incidence angle features, target green LAI with uncertainty.

## Phase 2b: growth, water and yield module (started 2026-10-04)
- phase2_growth.py (wheat first): daily PyTorch model driven by the frozen crop calendar; RUE
  biomass, juvenile (temperature-driven) leaf growth then specific leaf area, partitioning to
  grain after heading + stem reserves, leaf senescence faster under drought, 10 soil layers of
  20 cm with root growth to heading, transpiration = ET0 x Kc x absorbed fraction cut by root-zone
  water, heat around flowering, saturating technology trend; then boosting on log(obs/sim).
- Field yield data: owner has none; options = state variety trials (LSV), long-term field
  experiments (BonaRes), and the consistency test (fields summed to district = official yield).
- Wheat v1 result (RMSE t/ha, identical rows; boosting bar in brackets): new years hybrid 1.01
  (0.96), physics only 1.17; new regions 1.12 (1.15); both 1.27 (1.33); drought 0.99 (1.01).
  Physics alone beats WOFOST+weather everywhere. Fitted: RUE 2.39 g/MJ PAR, k 0.52, root 115 cm,
  Kc 1.17, stress below 52 % available water, heat >32.5 C, tech tau 20 y. Main miss: wet years
  2023 (+1.1) and 2024 (+0.9) over-predicted; post-2017 yields below trend (fertiliser rules).
  Next: excess-water/disease terms, correction without raw year, CV-chosen hybrid+boosting average.
- Wheat v2 (2026-10-05): + wet-season term (c_wet 0.008/wet day), correction without year,
  CV blend (0.9 hybrid / 0.1 boosting). Test RMSE: new years 1.02 (bar 0.96), regions 1.11 (1.15),
  both 1.27 (1.33), drought 0.96 (1.01). CV 1979-2017: hybrid 0.81 vs boosting 0.91. Test years:
  both biased high (blend +0.43, boosting +0.32), scatter equal (0.92 vs 0.90). 2023 +1.2 (harvest
  rain?), 2024 +0.85. Next: forward validation inside training to set tech-level rule; harvest rain.
- Forward validation (fit <=2008, score 2009-2017 training rows; test untouched): boosting 0.77,
  hybrid tech-extrapolated 0.80 (bias +0.25), hybrid tech-frozen 0.78 (+0.11), blend frozen 0.77.
  Rule adopted: technology level held at last training year. Wheat v3 test: new years 0.97 (bar
  0.96, tie), regions 1.11 (1.15), both 1.24 (1.33), drought 0.95 (1.01).
- Nitrogen data (2026-10-05): downloaded NRW nitrate red areas 2021-03, 2022-01, 2024-01
  (data/raw/nitrate_nrw, 80 MB) and UBA Texte 131/2019 report (data/raw/nitrogen): it has only
  maps/state tables, no district time series (data on request from Uni Giessen, Bach). Destatis
  GENESIS 42321 and Regionaldatenbank livestock need a registered account -> owner downloads.
  Open alternative: Batool et al. 2022 Sci Data, European soil N surplus 1850-2019, gridded.
- Nitrogen data built (phase0_nitrogen.py -> nitrogen_districts.csv): Batool et al. 2022 N surplus
  (16 methods, mean/sd, 1979-2019) per district; Destatis 42321-0001 (Germany, harvest 2011-2025)
  and 42321-0010 (states, harvest 2017-2025) mineral N sales, -40 % from 2010-16 to 2021-23.
  n_supply = surplus to 2019, then + 100 kg/ha x (sales index change). Median: 2019 39, 2022 29,
  2023 17, 2024 19, 2025 29 kg/ha (2023/2024 = the two most over-predicted wheat years).
- Wheat v4 = v3 + N factor on yield exp(-c_n softplus((N_ref - N)/10)); forward validation running.
  Limitation: forward validation (<=2017) only sees the surplus decline, not the post-2019 sales drop.
- v4 forward validation: hybrid frozen 0.79 (v3 0.78), blend 0.78 (0.77) = neutral within noise.
  Fitted N effect: c_n 0.027, N_ref 33 kg/ha -> e.g. 2023/24 (N ~17-19) about -5 % yield.
  Kept (physically motivated; forward period cannot test the post-2019 sales drop). Final v4 fit
  on 1979-2017 + single test scoring running.
- Nitrogen v2 (coherent, 2026-10-05): n_supply = national mineral N per ha cropland 1979-2025
  (FAOSTAT agricultural N use = Destatis sales; Destatis replaces FAO 2010+; harvest year = marketing
  year + 1) x state factor (state vs German sales change since 2017, 3-yr smoothed, half weight;
  city states + Saarland = 1) + fixed district manure term (district minus state mean Batool surplus
  2010-2019, sd 16 kg/ha). Median: 1979 136, 1990 146, 2010 103, 2017 111, 2019 90, 2022 76,
  2023 72, 2024 73, 2025 79 kg/ha. Old Destatis Fachserie PDFs (state data before 2017) are scans
  with OCR errors - not used. Wheat v5 = N on new scale (N_ref 30-200); forward validation running.
- v5 forward (sales-based N): hybrid frozen 0.80 (v3 0.78) -> sales N not used. Waiting for crop-
  specific N (figshare 25435432 Cropland_Maps.zip 6.15 GB, CC0; Adalibieke Zenodo 7408852 is
  restricted - owner may request access).
- Barley forward (no N, fit <=2008, score 2009-2017): boosting 0.92, hybrid extrapolated 0.86,
  frozen 0.92 -> technology rule per crop by forward validation (wheat frozen, barley extrapolated).
  Final barley fit running.
- Crop-specific N (2026-10-05): figshare 25435432 Cropland_Maps.zip (6.15 GB, CC0), maps = kg N per
  ha of grid cell per crop group. Extracted N maps for Wheat, Other Cereals, Maize, Roots and tubers
  1979-2019 (n_maps, 588 MB). Rate per ha of crop = state sum(map x cell area) / state crop area
  (census years interpolated; maize = grain maize only). District rates too noisy (wheat p10-p90
  108-377) -> state level. Median wheat: 1979 194, 2000 214, 2019 155, 2023 117 (extended by
  national change); cereals 197/137/120/91; maize 140/158/147/111; potato 293/164/152/114.
- Barley final (no N): new years 1.11 (bar 1.00), regions 0.97 (1.03), both 1.38 (1.31), drought
  1.06 (1.01); test bias +0.29, 2023 +0.75, 2024 +0.99. Wheat v6 (crop N) forward running.
- Fertiliser maps kept (Germany-only crops, 5 arcmin, LZW): n_maps = N for Wheat, Other Cereals,
  Maize, Roots and tubers, Sugar crops, Other Oilseeds 1979-2019 (12 MB); pk_maps = P2O5 and K2O for
  the four crop groups (16 MB). Cropping verified identical to the original. Cropland_Maps.zip
  (6.15 GB) deleted 2026-10-05; re-download: figshare 10.6084/m9.figshare.25435432 (file 50123172).
- Wheat v6 forward (crop N): blend extrapolated 0.73, frozen 0.74, hybrid frozen 0.75 (v3 0.78),
  boosting 0.77. Fitted c_n 0.03, N_ref 125 kg/ha. Wheat rule -> extrapolated. Final v6 fit running.
- Wheat v6 final (crop N in physics + correction, tech extrapolated): new years 1.12 (v3 0.97, bar
  0.96), regions 1.04 (1.15), both 1.43, drought 1.09; test bias +0.60 (2023 +1.57). Cause: test-year
  wheat N 117-159 kg/ha below training range (p1 133); trees cannot extrapolate; physics N effect
  unidentified (c_n ~0.03). Lesson: inputs leaving the training range must not enter the tree
  correction - handle them in physics. Test set looked at 6 times (v1-v6), all logged; proposed:
  v7 = last wheat version, decided by principle; untouched check = harvest 2026 yields (spring 2027).
- Copernicus Data Space: openEO login (device code) OK 2026-10-05; phase4_s2_fields.py (S2 L2A field
  means, SCL 4/5 only, 10 m inner buffer, bands + angles). Test job 20 wheat fields 2023 running.
- S2 test job (20 wheat fields, Oct 2022-Aug 2023): 5:41 min incl. queue, 858 cpu-s, 230 Mpx input,
  84 acquisitions, 68 with >=1 valid field; cloud-free obs per field median 20 (14-43). Median NDVI:
  Oct-Nov 0.35, Mar 0.67, Apr 0.81, May 0.88, Jun 0.86, Jul 0.30, Aug 0.19 (plausible wheat cycle).
  Reflectance as 0-10000 DN. Next: scale to all Rur wheat fields per year (check credit use first).
- Wheat v7 forward (drifting inputs physics-only): blend extrapolated 0.77 (-0.08), hybrid 0.78,
  frozen 0.80-0.82; boosting 0.77. Rule: extrapolated. Final v7 = last wheat test score; running.
- S2 500 wheat fields 2023: 9 min, 4091 Mpx, 3276 cpu-s, 13 credits (balance 9994 -> 9981); median 19
  cloud-free obs/field (min 12). Full run started: all Rur wheat fields >= 3 ha (~6,000-6,600 per
  year), harvest 2021-2026, expected ~170 credits/year (~1,000 total). Output data/processed/s2/.
- S2 field selection changed (owner: include small farms): min 0.3 ha (was 3 ha); shrink 10 m, or
  5 m if <0.1 ha remains; skip if <0.05 ha (5 px). 2021 wheat: 16,736 of 17,202 fields kept (496 at
  5 m), median inner area 1.5 ha. Per field: shrink + inner_ha saved (quality weight). Chunks of
  1,500 fields (~12 jobs/season). First >=3 ha attempt: 413 (geometry too large) - fixed by 2 m
  simplification + 6-decimal coordinates; the interrupted 3 ha job was cancelled on the server.
- WHEAT FINAL (v7, last test look, 2026-10-06): RMSE t/ha blend (boosting bar): new years 0.94 (0.96),
  new regions 1.11 (1.15), both 1.25 (1.33), drought 0.97 (1.01); hybrid 0.96/1.11/1.23/0.97; physics
  alone 1.13/1.30/1.32/1.22 (WOFOST+weather 1.27/1.29/1.67/1.28). Blend weight 0.8 hybrid. Bootstrap
  of the difference vs boosting running. Versions on test: v1 1.01, v2 1.02, v3 0.97, v4 stopped,
  v5 not scored, v6 1.12, v7 0.94 (new years, blend).
- S2 download: chunk 0 of 2021 finished server-side in 15 min (7,062 cpu-s, 8,666 Mpx) but the
  computer slept; script now submits all chunks of a season at once, reuses server jobs, resumable.
- Bootstrap wheat v7 blend minus boosting (95% CI): new years -0.014 (-0.095, +0.067) tie; new
  regions -0.041 (-0.059, -0.022); both -0.086 (-0.142, -0.021); drought -0.040 (-0.065, -0.014).
  Report section 8b + milestone log updated.
- Forecast mode (phase2_forecast.py, 2026-10-06): weather observed to forecast date F, then each
  training year 1979-2017 as scenario (39); calendar, growth, correction, blend and Phase 1 boosting
  run per scenario; not-yet-observed stage dates replaced by calendar prediction; median = forecast,
  10-90 % = range. Leads 12/8/6/4/2/0 weeks before typical harvest (wheat day 218). Scored on
  test_years + test_both: RMSE, bias, 80 % coverage. 9 s per scenario; wheat run started.
- Barley v7 forward (with cereal N): blend 0.95, hybrid 0.96 vs boosting 0.93 (earlier no-N config
  0.86). Running barley v7 without N to separate the N effect from the tech-feature change.
- 2026-10-06: previous session ended mid-run. Saved: wheat forecasts lead 12/8/6; S2 2021 parts
  0,1,2,5-10 (part 11 finished on server, reused). Parts 3/4 failed repeatedly: GeometryCollection
  from fields split by the 10 m inward shrink -> converted to MultiPolygon (117 fields); retries
  capped at 2. Both jobs restarted.
- Barley v7 forward without N: blend 0.95 (same as with N) -> N is not the cause; the drop vs the
  earlier no-N config (0.86) comes from removing phys_tech from the correction (DRIFTING rule).
  Barley yields rose steeply 2009-2017; open question how to carry barley's trend. To decide later.
- Wheat forecast skill (test_years RMSE t/ha, blend vs boosting): 12 wk 1.02/1.06, 8 wk 0.97/1.04,
  6 wk 0.93/1.00, 4 wk 0.91/0.98, 2 wk 0.91/0.97, 0 wk (6 Aug) 0.90/0.97; both-new 8 wk 1.32/1.44,
  0 wk 1.19/1.34. Bias +0.14..+0.38 (both models positive). Weather-only 80 % ranges covered 22-41 %.
- Calibrated ranges (phase2_intervals.py): scenario predictions + training out-of-sample residuals
  (all combinations). New years: residuals from leaving out year blocks (sd 0.81) -> coverage
  76-77 %, width ~2.1-2.3 t/ha. Both-new: residuals from leaving out a state AND a year block
  (sd 0.96) -> 69-72 %, width ~2.4-2.6 (leave-state-only gave 60-66 %). No test data used to build.
- Barley forward with technology level in the correction: 0.87 (with N), 0.87 (no N), boosting 0.93,
  v7 0.95. Final barley config: N in physics, tech in physics + correction (TECH_IN_CORRECTION),
  extrapolated trend. Final barley fit (one test score) running.
- BARLEY FINAL (v7 + tech in correction, test scored once; earlier noN v5 also scored): blend vs
  boosting RMSE: new years 1.07/1.00 (diff +0.07, CI -0.005..+0.15, tie/slightly worse), new regions
  0.99/1.03 (-0.04, CI -0.056..-0.022, better), both 1.29/1.31 (tie), drought 1.09/1.01 (+0.085,
  CI +0.05..+0.12, worse). Year bias: 2022 -0.62 (drought year, barley yields held up - early
  harvest escapes summer drought; model's drought penalty too strong for barley), 2025 -1.08
  (record year under-predicted), 2018 +0.51, 2024 +0.50. Open: barley drought response (test only
  with forward validation on training drought years, e.g. 2003, 2015) - after Phase 4 start.
- PROSAIL retrieval (phase4_prosail.py): 70,000 simulated cereal canopies (angles from real S2
  data), Gaussian S2 band responses, 3 % + 0.005 noise, network -> LAI, Cab, CCC, Cbrown with sd.
- PROSAIL network trained (70,000 simulations, 114 s; 60 epochs). Unseen simulated canopies: LAI RMSE
  0.86 (R2 0.82), Cab 8.2 ug/cm2 (0.66), CCC 39.6 (0.86), Cbrown 0.19 (0.78); 90 % ranges hold the
  truth 89-93 % (honest uncertainty). Applied to 2021 wheat fields (retrieval_115_2021.csv, 54 MB):
  median LAI Oct-Dec 0.25-0.34, Mar 0.86, Apr 2.37, May 3.59, Jun 3.56, Jul 0.75, Aug 0.42; Cab
  May-Jun ~56. Peak LAI 3.6 matches the district growth model (median simulated LAI max 3.6).
  Odd: Jan 2021 Cab 10.7 (low sun / residual snow?) - check later.

## 2026-10-06 - SL2P comparison (independent check of our PROSAIL LAI), wheat 2021, Rur
- Downloaded ESA SNAP Biophysical Processor weights v2_1 (LAI, LAI_Cab; S2A network; 0.4 MB) from
  github.com/senbox-org/s2tbx to data/raw/sl2p. Code: phase4_sl2p.py.
- Verification: LAI reproduces ESA's 100 test cases (max diff 0.0002). LAI_Cab: all valid test
  cases reproduced (max diff 0.005); ESA's test file marks every value > 90 as invalid although the
  SNAP output limit is 600 - a test-file convention, we keep SNAP's limits.
- 381,945 field-dates; 80 % inside ESA's definition domain; SL2P LAI invalid 2.4 %.
- LAI ours vs SL2P: r 0.95, RMSD 0.73, mean diff -0.19. Oct-Apr and Jul-Aug: close (diff < 0.1,
  RMSD 0.2-0.5). Peak (May-June): SL2P higher - median 4.7-4.8 vs ours 3.6 (90th pct 6.3 vs 5.0);
  only 50-57 % of SL2P values inside our 90 % range. Per-field peak ranking still agrees (r 0.85).
- Both are trained on PROSAIL simulations, and reflectance saturates at high LAI, so each method
  falls back on its own prior there (ours: cereal prior centred on 3.5). Which one is right at peak
  can only be settled with ground LAI measurements -> look for open wheat LAI field data in the
  Rur region (TR32 / TERENO Selhausen, Merzenhausen) before the assimilation weights peak LAI.
- January (308 obs, frost/snow) unreliable in both -> drop Dec-Jan dates with low sun in assimilation.
- CCC: SL2P much higher at peak (340 vs 195) - same saturation issue plus different leaf Cab prior.

## 2026-10-06 - TR32 field data (Rur catchment ground truth)
- Downloaded reichenau_et_al_crop_and_ancillary_data_v1.0.zip (31 MB, 79 files) to data/raw/tr32
  and unpacked. Licence CC BY 4.0; user accepted the TR32DB download terms. Citation (mandatory):
  Reichenau, T. G. et al. (2020): A comprehensive dataset of vegetation states, fluxes of matter and
  energy, weather, agricultural management, and soil properties from intensively monitored crop
  sites in Western Germany. ESSD 12, 2333-2353; data DOI 10.5880/TR32DB.39. Contact the creators
  before publishing results that use it.
- Content per site (Selhausen, Merken, Merzenhausen, Huertgenwald): vegetation (green/brown LAI,
  biomass by organ, BBCH, height, N/C, GPS per sample), management (sowing, every N dose, yield),
  site weather (half-hourly), soil profiles, eddy-covariance fluxes (ET) for 11 field-years.
- Winter wheat / barley field-seasons: 20 (2008-2017). Measured yields for 9 of them (WW 8.3-10.5,
  WB 7.97-9.7 t/ha), with sowing date and total N (85-196 kg N/ha).
- Sentinel-2 overlap is small: S2A data start July 2015, so usable seasons are 2016 (SEF05WW16,
  SEF01WB16, SEF06WB16) and 2017 (MEF01WW17): about 4 field-seasons with ~7 LAI dates each.
- Measured peak green LAI of wheat 4.4-7.6 (single-date means) - above both our PROSAIL peak (3.6)
  and SL2P (4.8). Destructive leaf LAI is "true" LAI; satellites see effective LAI (clumping), so
  part of the gap is expected; to be quantified with the S2 matchups.
- EC water-use years with wheat: Merzenhausen 2011, 2012, 2017; Selhausen 2015 (WW), 2016 (WB);
  Merken 2009 -> direct check of the model's actual evapotranspiration.

## 2026-10-06 - Satellite LAI vs measured LAI (TR32 fields, 2016-2017)
- phase4_tr32.py: 20 m circles around the TR32 sampling points, one openEO job (Jun 2015-Aug 2017,
  same bands/mask as the field extraction; 252 cpu-s, well under 1 credit). 2015 seasons have no
  usable in-season image (S2A data start after harvest) -> 4 field-seasons left, 7-9 cloud-free
  in-season dates each.
- Matchups (satellite within 3 days, or interpolated between dates <= 21 days apart): only 7.
  Ground LAI itself is uncertain (3 samples per date, sd 0.2-1.8).
  ours (PROSAIL net): mean diff +0.03, RMSE 0.70, r 0.79; ground LAI >= 3: -0.11
  ESA SL2P:           mean diff +0.58, RMSE 1.15, r 0.84; ground LAI >= 3: +0.69
- Reading: SL2P's higher peak (4.8 vs 3.6 in 2021) looks like over-estimation, not our network
  under-estimating; keep our PROSAIL retrieval as the main LAI. Sample is far too small to be
  conclusive -> in the assimilation, satellite LAI error = network sd plus about 0.7.

## 2026-10-06 - Field growth model + satellite assimilation (wheat, Rur), first version
- Fields mapped to districts by centroid (VG250 KRS): data/processed/field/fields_district.csv. Most
  wheat fields in Dueren, Heinsberg, Rhein-Erft, Euskirchen, Aachen, plus Neuss, Rhein-Sieg, MG.
- phase2_growth.py: forward(record=True) returns daily LAI; load_growth keeps the development clock
  (dev, stage thresholds); optional field-only terms (frost leaf loss, senescence onset g0) are off
  by default -> district model v7 unchanged. Backup: data/processed/field/phase2_growth_backup.py.
- phase4_field_assim.py: per district a shared prior ensemble (3,000 members, Latin hypercube) of
  field unknowns (RUE, initial LAI, early leaf growth, SLA, leaf ageing, soil water x0.5-1.5,
  N 60-240 kg/ha, development speed +-6 %); per field importance weights from satellite LAI
  (Student-t, error sqrt(sd_net^2 + 0.5^2), tempered to ESS >= 50); Dec/Jan and post-maturity
  dates dropped. Field yield = district blend x weighted physics yield / district prior mean.
- Finding: the yield-fitted district model has the wrong LAI shape (winter LAI ~1.9 vs satellite
  0.6; leaves age 1-2 weeks early). Fitted LAI-shape settings to the 2022 district median satellite
  LAI (no yields): rgr 0.009 -> 0.0027, sla 0.028, r_sen 0.056, senescence starts at 31 % of grain
  filling, frost loss small. growth_lai_shape_winter_wheat.json. District median RMSE 0.88 -> 0.41.
- Ensemble ranges and error tuned on the 2022 hidden-date check, confirmed on 2021 (unseen):
  hidden every other date: no satellite 0.99 (v7 shape) -> 0.77 (new shape); with satellite 0.58;
  line between neighbouring dates 0.63; 90 % range holds 96 % (a bit wide).
  look-ahead (satellite to 1 June, predict June-July): with satellite 0.90, district model 0.98,
  last value carried forward 1.35.
- Full 2021 run (47 s): 15,981 fields, median 14 satellite dates, ESS 472. Field yield 10/50/90 %:
  7.3 / 8.4 / 9.2 t/ha; peak LAI 3.0-4.3; water use 353-387 mm (demand 366-399 mm).
  N and soil water are NOT identified by LAI in 2021 (posterior = prior, wet year) -> needs
  stress years, Sentinel-1 / chlorophyll (CCC) for N.
- Not yet done (frozen comparison later): field sums vs official district yields. Note: while
  reading growth_predictions I printed the official 2021 yields of the Rur districts (already used
  in the v7 district scores); nothing in the field method was chosen with them.
- S2 download: 2023 complete, 2024 4/12 parts; stopped by the 2-h background limit.

## 2026-10-06 - FIELD METHOD FROZEN (before any comparison with official yields)
- Frozen: phase4_field_assim.py (3,000 members, ranges as in file, error 0.5, Student-t nu 4,
  ESS >= 50, Dec/Jan dropped), phase2_growth.py, growth_lai_shape_winter_wheat.json (fit on 2022
  satellite LAI), growth_model_winter_wheat_v7.json. md5 in data/processed/field/frozen_method_md5.txt.
- Comparison defined in advance: per district and year, field yields averaged weighted by field
  area (area_ha) = "fields + satellite"; compared with official district yields and with the
  district model alone ("blend", same as v7). Districts: the 5 core Rur districts (05334 Aachen,
  05358 Dueren, 05362 Rhein-Erft, 05366 Euskirchen, 05370 Heinsberg) as the main result; all
  districts with >= 100 fields as secondary. Years 2021-2025 (each scored once when its satellite
  data is complete; same frozen method). Metrics: RMSE, mean error, per year.
- 2026 stays untouched (official yields spring 2027).
- Bug fix after freeze (no effect on results): fields outside all district borders (4) dropped in observations() - crashed 2023. New md5 appended.

## 2026-10-06 - Frozen field method vs official yields, 2021-2023 (first and only look for these years)
- 5 core Rur districts (15 district-years): district model alone RMSE 0.89 (mean error -0.30);
  fields + satellite RMSE 1.17 (mean error -0.58). Per year: 2021 0.50 vs 0.47 (slightly better);
  2022 1.38 vs 1.88 (worse); 2023 0.50 vs 0.58 (worse). All 22 district-years with >= 100 fields:
  0.87 vs 1.17. -> The frozen field method does NOT improve district yields; it makes them worse.
- Cause (diagnosed without yields): field yield / district model has median 0.96 (2021), 0.92
  (2022), 0.95 (2023). The prior ensemble was widened upwards (2022 tuning for LAI), so its mean is
  above the typical field; dividing by the prior mean pulls every field down. In 2022 (dry, sunny)
  satellite LAI was low but yields high (9-10.6 t/ha): lower LAI did not mean lower yield.
- Lesson: LAI alone is a weak yield signal at district scale; the ratio must be anchored to the
  typical field, not the prior mean. Proposed v2, to be fixed BEFORE 2024/2025 are scored:
  field yields rescaled so their area-weighted district mean equals the district model (satellite
  only distributes yield between fields); frozen v1 and v2 both scored once on 2024 and 2025.

## 2026-10-06 - Satellite yield signals per field + v2 (FROZEN before 2024/2025 are scored)
- phase4_field_features.py (no yields): per field LAI at heading, green leaf area duration after
  heading (0-45 d), stay-green days (LAI > 1.5), CCC and Cab around heading (+-10 d), LAI peak.
  Heading/maturity from the frozen district calendar. Coverage 70-100 % of fields (2021 had gaps).
- Within-district correlation with the model's field yield: green leaf duration 0.87-0.96, LAI
  peak 0.76-0.92, CCC 0.73-0.86, Cab 0.64-0.75, stay-green 0.51-0.54 (saturates: most fields stay
  above 1.5 for the whole 45-day window -> use a longer window later). The model's field yield is
  essentially a green-leaf-duration index; chlorophyll / N carries partly independent information
  that can only be weighted with real field yields (YieldSAT).
- v2 frozen: v1 field yields x (district model / area-weighted mean of v1 field yields) - district
  mean equals the district model by construction, so v2's district score = district model; v2 can
  only be judged on real field yields. Files field_results_v2_winter_wheat_<year>.csv; md5 appended.
- BK50 soil map NRW: ISBK50_EPSG25832_GeoPackage.zip, 641 MB, opengeodata.nrw.de, dl-de/by-2-0,
  updated 2026-03-12 (awaiting permission).
- BK50 downloaded (641 MB) and extracted to data/raw/soil/bk50: BK50_A.gpkg (1.7 GB, layer BK50,
  155,128 soil units, NRW), method PDFs (NFK, GW, KAP, TIE, Liesmich), column legend. Zip deleted
  (owner's OK). phase0_bk50_fields.py builds data/processed/field/fields_soil_bk50.csv (one row per
  field record, area-weighted soil values) - running.
- 2026-10-06: owner submitted the YieldSAT access form (188 German wheat fields 2016-2022, combine
  yield maps 10 m, CC BY-NC-ND 4.0: research only, no commercial use).
- fields_soil_bk50.csv done (211,835 field records, 135,462 distinct outlines, 31 MB; mapped share
  median 1.0). Wheat fields: plant-available water 98 / 160 / 225 mm (10/50/90 %), rooting depth
  ~11 dm, groundwater in reach for ~10 % of fields, soil value (Bodenwertzahl) 48 / 71 / 80.
- Sanity check (no yields): within districts, field NFK correlates with green leaf duration after
  heading 0.16 (2021, wet), 0.28 (2022, dry), 0.24 (2023); soil value similar. Soil water matters
  more in the dry year - physically consistent; next: field NFK into the field model (new version,
  frozen before scoring).

## 2026-10-06 - Field model v3: own soil water per field (BK50)
- phase4_field_assim_v3.py: soil prior on the members' soil-water multiplier (log-normal around
  field NFK / district median NFK, sd 0.15), then satellite likelihood as v1; yields rescaled as v2.
- Posterior soil multiplier follows the map (r 0.98). Field yields v3 vs v2: r 0.95 (2021, wet),
  0.78 (2022, dry), 0.84 (2023); mean change 0.19 / 0.57 / 0.49 t/ha. Field yield vs soil water
  r 0.32 / 0.79 / 0.63 (v2: 0.09 / 0.30 / 0.18).
- LAI look-ahead check 2022 (to 1 June): v1 0.716, v3 0.724 - the satellite neither confirms nor
  contradicts the soil effect on LAI. Whether v3's strong soil effect in dry years is right can
  only be judged on field yields (BK50 covers NRW only; YieldSAT fields elsewhere need a national
  soil map).
- HYRAS 1 km, 2020-2026, 5 variables: 35 files, 2.64 GB download, Rur cells kept, raw deleted
  (awaiting permission). Field sunlight: CM SAF SARAH-3 daily SIS 0.05 deg (needs free CM SAF /
  EUMETSAT account - owner).

## 2026-10-06 - Field weather (HYRAS 1 km) + field model v4
- phase0_hyras_fields.py: HYRAS v6-1 2020-2026 (tas, tasmax, tasmin, hurs, pr), 3,872 1 km cells
  holding Rur fields kept (~7 MB/year), raw grids deleted. Season rain differs strongly between
  cells (2023: 503-1,399 mm; Eifel vs lowland).
- phase4_field_assim_v4.py: v3 + own weather: up to 20 weather groups per district (k-means on
  season rain, spring rain, temperature); per group tmin/tmax/rain drive the calendar (recomputed),
  growth, water and heat; PAR and ET0 still district. Look-ahead LAI check 2022 (to 1 June):
  v1 0.716, v3 0.724, v4 0.670 -> field weather improves LAI prediction (no yields used).
- Field sunlight: SARAH-3 order via CM SAF WUI by owner (account needed).
- Report updated (rev 33): new section 9b 'Field level: built and tested' with chart (district comparison v1), LAI checks table, v2-v4, milestone rows. Sentinel-1 test (500 wheat fields, 2023, ascending, sigma0-ellipsoid) running: phase4_s1_fields.py.
- Report rev 34: section 9a 'How the field level relates to the district model' (training/test logic, timeline, tests, data per level).
- S1 test 1 failed: local_incidence_angle not supported for SENTINEL1_GRD on CDSE (4 credits, 95 min mostly queue). Removed, rerun.
- S1 test 2 (no incidence angle): finished but EMPTY output (header only) - 33 credits, 2.4 Gpx input,
  34 GB read. Cause unknown (suspect orbit-state filter value or masking). Cost scale: ~33 credits
  per 500 fields per season -> all Rur wheat 2021-2026 would be ~6,900 credits (too much with the
  10,000/month budget). Next: debug on 20 fields x 1 month (~1 credit) before any larger run.

## 2026-10-06 - SARAH-3 and YieldSAT arrived
- SARAH-3 order (CM SAF ORD70178/70179): 7 tar files, 25 GB, daily SIS from 2015-09-01 (CDR to
  2020, ICDR from 2021) in data/raw/sarah3; phase0_sarah3.py extracting Rur + Germany boxes (fix:
  SIS stored as int16 -> cast before NaN fill). Raw tars kept until the owner agrees to delete.
- YieldSAT Germany (Germany.zip 2.2 GB -> Germany_raw.zip + Germany_preprocessed.zip): 299 fields,
  188 wheat. Per field: metadata (sowing/harvest dates from the farmer for 184, field-mean yield
  'yield_ground_truth' at 15 % moisture, yield-map quality Good 94 / Average 81 / Bad 13), yield
  maps (mean/std/count rasters), ~40 Sentinel-2 L2A clips + SCL masks, SoilGrids 0-200 cm, DEM,
  daily weather (ERA5-type). Location: farm2 (124 wheat fields) in southern Lower Saxony
  (51.6-51.8 N, 9.7-9.9 E); farm1/5 Mecklenburg-Vorpommern; farm3/4/6 Saxony-Anhalt - none in the
  district test states. Wheat fields per year 2016-2022: 27/29/30/49/16/20/17.
- Note: while surveying, per-year mean of the field yields was printed (8.0/8.6/7.7/8.6/8.6/7.9/
  9.6 t/ha). Evaluation design to be fixed before any field-level comparison.
- SARAH-3 extracted: sarah3_rur.npz (24 x 34 cells, 3 MB) and weather/sarah3_germany.npz (158 x 186,
  float16, 87 MB), 4,051 days 2015-09-01 - 2026-10-03, no gaps. Check vs district radiation Dueren:
  r 0.985 (2016-2020, HYRAS) / 0.979 (2021+, station estimate); SARAH ~9 % higher (product level).
  Use in models: field/district differences and daily variation from SARAH, level kept from the
  district radiation the models were calibrated on. 25 GB tar files deleted (owner's OK).
- YieldSAT S2 (phase4_yieldsat_s2.py): 4,106 clear field-dates for all 188 wheat fields (median 24
  per field); +1000 offset removed on 50 clips from 2022; monthly LAI like Rur (May-June ~3.2).
- YieldSAT SoilGrids layout: 12 bands = 6 depths (0-5,5-15,15-30,30-60,60-100,100-200 cm) x
  (mean, uncertainty); odd bands = mean (sand+silt+clay ~980 g/kg).

## 2026-10-06 - Sunlight check against measured radiation (DWD pyranometer stations)
- Downloaded all DWD daily solar station files (57 files, 6.1 MB, data/raw/dwd_solar; open data).
  phase0_solar_check.py: measured daily global radiation vs SARAH-3 (nearest 0.05 deg cell) and vs
  our district radiation (district containing the station), 2015-09 - 2026-08.
- 2015-2020 (30 stations, 52,511 days): SARAH bias -0.1 %, RMSE 14.6 W/m2, r 0.989; district
  (HYRAS grid) bias -3.2 %, RMSE 13.4, r 0.992 (HYRAS is built from these stations - not independent).
- 2021-2026 (41 stations, 73,803 days): SARAH bias -0.8 %, RMSE 14.0, r 0.990; district (sunshine-
  hour estimate) bias -4.2 %, RMSE 18.4, r 0.985. Aachen-Orsbach: SARAH -0.5 W/m2, district -5.5.
- Conclusion: SARAH-3 matches the measurements; our district radiation is 3-4 % low (estimate
  after 2020 also noisier). The district model's fitted RUE absorbs a constant level offset, so
  v7 stays as is; field models use SARAH's relative pattern (consistent). Candidate for a future
  district version: SARAH-3 radiation for all years (needs SARAH 1983-2015 for Germany).

## 2026-10-06 - PRE-REGISTRATION: YieldSAT field-level test (written before any field-yield comparison)
- Model: field model v5 as in phase4_yieldsat_run.py (frozen when the run finishes; md5 appended to
  data/processed/field/frozen_method_md5.txt before scoring). R = 0.962 fixed from Rur v4 (no yields).
- Truth: YieldSAT 'yield_ground_truth' (field-mean yield, 15 % moisture), converted to our 14 %
  (x 0.85/0.86). All 188 wheat fields; no field removed afterwards (map quality only as a
  secondary breakdown).
- MAIN result: harvests 2018-2022 (132 fields; district model out-of-sample there).
  Secondary: 2016-2017 (56 fields; district anchor in-sample - flagged).
- Metrics: RMSE, mean error, r over fields; and the field-level skill that matters most:
  within district-year correlation (does the model rank fields correctly inside a district-year,
  district-years with >= 4 fields).
- Baselines (computed in the same script, same fields):
  B1 district model alone (blend for the field's district-year);
  B2 satellite index: linear regression of yield on green leaf duration after heading (GLAD),
     leave-one-year-out (it learns from YieldSAT yields, so fit only on other years);
  B3 official district yield of that district-year (uses the true district yield - an oracle
     reference for the district part, not a competitor).
- Claim rule: v5 'better than B1/B2' only if RMSE lower AND the bootstrap (resampling district-
  years, 2,000 draws) 90 % interval of the RMSE difference excludes 0. One look; any later version
  needs new data (2026 Rur harvest / farmers' yield maps) for its own test.
- YieldSAT v5 run 1: 38 fields without result - September clips had no view angles (Rur angle table
  lacks September) -> NaN LAI -> NaN weights. Fix: NaN observations dropped in the run; angle table
  interpolated over months in phase4_yieldsat_s2.py. Calendar check (no yields): predicted heading
  day 147 vs DWD 147.5, maturity 193 vs DWD yellow ripeness 192.5; farmers harvest 27 days after
  yellow ripeness (8 days later than DWD harvest) - calendar correct, no change. Rerun.
- v5 YieldSAT run complete (188/188 fields, median 20 satellite dates, ESS >= 51) and FROZEN: md5 in frozen_method_md5.txt. Scoring next (one look).

## 2026-10-06 - RESULT: YieldSAT field-level test (pre-registered, one look)
- MAIN 2018-2022 (132 fields, 19 district-years, truth mean 8.32 t/ha at 14 %):
  v5 field model          RMSE 1.38, mean error -0.24, r 0.38, within district-year r 0.24 (7 groups)
  B1 district model       RMSE 1.28, mean error -0.33, r 0.33 (same value for all fields of a district-year)
  B2 satellite index LOYO RMSE 1.00, mean error -0.11, r 0.68, within district-year r 0.56
  B3 official district    RMSE 1.13 (oracle)
  v5 - B1: 90 % interval -0.24 .. +0.45 (tie, not better); v5 - B2: +0.13 .. +0.65 (B2 better).
- Secondary 2016-2017 (56 fields, district anchor in-sample): v5 1.74, B1 1.50, B2 1.75, B3 1.37.
- By map quality (main): Average n 59 v5 1.25 / B1 1.23; Good n 73 v5 1.47 / B1 1.33.
- (Script prints 'within district-year r 0.00' for B1/B3: constant within a district-year -> not
  defined; read as 'no ranking'.)
- Verdict: the physics field model v5 does NOT beat the district model and is clearly beaten by a
  simple learned satellite index (green leaf duration after heading, fitted on other years), which
  even beats the true district yield (1.00 vs 1.13). The satellite carries real field information;
  our physics ensemble turns it into yield poorly (field ranking r 0.24 vs 0.56).
- Lessons: (1) the yield of the ensemble is driven by LAI shape through a district-calibrated RUE/
  partitioning - field differences in yield per unit leaf are not captured; (2) a learned satellite
  -> yield link is strong; next version should be hybrid at field level (physics + satellite
  features + learned correction), validated by leave-one-farm-out and leave-one-year-out on
  YieldSAT - and its final claim needs NEW field data (YieldSAT test is now used).

## 2026-10-07 - Field hybrid v6 on YieldSAT (district recipe at field scale; nested CV, not a fresh test)
- phase4_yieldsat_v6.py: 16 features (physics from frozen v5, satellite, SoilGrids, HYRAS weather,
  SARAH sunlight); ridge models with alpha by inner leave-one-year-out; outer LOYO and LOFO.
- LOYO, 2018-2022 (RMSE / within district-year r): M0 district 1.28 / - ; M1 satellite index
  1.00 / 0.56; M2 hybrid v5 x correction 1.12 / 0.45; M3 direct ridge 1.12 / 0.53; M4 blend M2+M1
  0.98 / 0.54. Best minus M1: -0.11 .. +0.08 (tie). All years: M4 1.26, M1 1.27, M0 1.35.
- LOFO (new farm), 2018-2022: M0 1.28; M1 1.16 (bias -0.50); M2 1.71 (bias -1.08); M3 1.38; M4 1.23.
  All models lose and get biased low when a farm is unseen - farm-level differences (management,
  variety) are the hard part; with 6 farms (one with 124 fields) nothing learns them.
- Verdict: with 188 fields the field-scale hybrid does not beat a one-feature satellite index; the
  physics adds little beyond the satellite at field scale. Ranking fields inside a district-year
  works with satellite features (r 0.4-0.56); the yield LEVEL across farms is best left to the
  district model. Practical design: district model for the level + satellite (learned) for the
  pattern between fields. More field yields are the limiting factor; YieldSAT 10 m maps (pixels)
  could give far more training examples for the satellite -> yield link (validate by farm/year).
- Pure field-level model (ridge on 12 field features: satellite, SoilGrids, HYRAS, SARAH; no district input): new year 2018-2022 RMSE 1.13 (all 1.33), new farm 1.47 (bias -0.89); within district-year r 0.49-0.51. Worse than district-anchored options (district model 1.28 robust for new farms; satellite index 1.00/1.16).

## 2026-10-07 - Search for more field-level yield data (owner's request)
- Open and directly usable: none beyond YieldSAT with field yields + Germany + Sentinel-2 era.
  * ETH Zurich 'Pixel to practice' (Sci Data 2024, DOI 10.3929/ethz-b-000662770, CC BY 4.0):
    9 winter wheat fields, 8 farms, central Switzerland, 2020; yield maps for several fields,
    management records, Sentinel-2 - small, but open.
  * Bundessortenamt multi-environment trials (Sci Data 2025, s41597-024-04332-7): 6 German sites,
    2015-2020, 228 cultivars x 9 management scenarios, yield + heading/maturity + diseases +
    fertiliser - plot level (not satellite), good for growth-model / N-response testing.
  * ZALF N-fertilisation wheat yields 1958-2015 (BonaRes DOI 10.20387/BONARES-YG6F-K61B): plots.
- On request only (authors / institutes): Perich et al. 2023 (Agroscope/ETH; large farm in
  western Switzerland, combine yield maps 2017-2021); SLU Sweden (18 + 8 fields); Cranfield/NIAB
  UK whole-farm 10-year yield maps; ZALF patchCROP (Brandenburg, 70 ha, patch yields from 2020).
- Farm-level (not field): German FADN / Testbetriebsnetz farm wheat yields 1995-2019 (423,815 farm-
  year-crop values) - research access via Thuenen/BMEL application.
- Dead ends: DFKI 2023 paper (= YieldSAT data); OpenAgrar/Zenodo 'wheat yield Germany' records are
  district yields (Duden et al., already used).

## 2026-10-07 - District model: barley forecast + intervals (step 1 of finishing the district level)
- Found: barley's final rule (technology level allowed in the correction, TECH_IN_CORRECTION) was
  only applied when phase2_growth.py ran directly; forecast / interval scripts imported the wheat
  rule. Fix: phase2_growth.drifting(crop) used in phase2_forecast.py and phase2_intervals.py
  (wheat unchanged). Barley forecast running (blend weight 0.8, harvest day from master median),
  then calibrated intervals.
- 2026-10-07: rapeseed postponed by owner (district level finished with wheat, barley, maize x2, potato first).
- Barley trend test (forward validation, saturating x linear technology): hybrid extrapolated 0.85
  (v7 0.87), bias -0.28 (v7 -0.33); 2017 still -0.9. Gain 0.02 t/ha = not clearly better -> keep
  barley v7, no new test look. Barley's recent rise (all models incl. boosting miss 2013-2017,
  plausibly hybrid-barley adoption) stays an open issue for a later version.

## 2026-10-07 - Barley forecast + calibrated intervals (barley steps 9-10)
- RMSE new years, blend / boosting: 12 wk 1.10/1.11, 8 wk 1.09/1.09, 6 wk 1.03/1.04, 4 wk 1.03/1.04,
  2 wk 1.09/1.02, at harvest 1.09/1.02. Blend better or tied up to 4 weeks before harvest; at the end
  boosting is better (as in the final barley test, 1.07 vs 1.00). Blend bias -0.22 at 0-2 wk.
- 80 % ranges: coverage only 60-66 % new years, 58-61 % both new (wheat 76-77 %). Cause: barley's
  year-level misses (2022 -0.71, 2025 -1.23, 2018 +0.48 t/ha, all districts move together) are
  larger than the training-years residuals suggest (the recent rise problem). Not fixed on the test
  data; candidate for a later version: residual pool from forward-validation years (2009-2017
  predicted from <= 2008), which contain such drift - to be checked on 2026.
- Open barley issues (owner informed, no change now): recent rise since ~2012 (hybrid barley?) ->
  needs variety / hybrid-share data (seed multiplication statistics, Beschreibende Sortenliste,
  LSV old-vs-new); ranges from forward-validation residual pool; blend weight toward boosting near
  harvest (forward validation). Revisit with 2026 check.

## 2026-10-07 - Grain maize step 6 (forward validation, fit <= 2008, predict 2009-2017)
- Maize settings (CROP_SETTINGS): rue 3.56, k 0.68, tb 5.4, topt 27.8, t_heat 31.0, fg 0.52, remob 0.38.
- Boosting 1.13 (bias -0.19); physics extrapolated 1.60 (-0.69); hybrid extrapolated 1.42 (-0.95);
  blend 0.9 1.36 (-0.87); frozen worse. All physics-based versions ~1 t/ha too low every year:
  the saturating technology curve flattens while maize breeding progress continued. Testing
  saturating x linear technology (techlin).

## 2026-10-07 - Step 11: standard district outputs (wheat, barley)
- phase2_outputs.py -> data/processed/outputs/district_outputs_<crop>.csv, all district-years
  1979-2025: predicted yield + 80 % range (out-of-fold training errors), normal yield (previous 5
  years official mean) and deviation, predicted ripening and harvest date (frozen calendar + district
  typical gap), water demand / use / ratio, official yield.
- Plausibility (test rows, medians): wheat water demand 370-434 mm, use 356-389 mm, ratio 0.85
  (2020) - 0.99 (2024); barley demand 325-384 mm. Predicted harvest moved ~3-4 weeks earlier
  since 1980 (wheat day ~228 -> ~205, barley ~205 -> ~184) - consistent with warming.
- Grain maize forward with saturating x linear technology (tech_lin 0.029 per decade): physics 1.53
  (-0.47), hybrid 1.30-1.32, blend 0.9 1.26-1.28; blend weights: 0.6 1.17, 0.4 1.13, 0.2 1.12;
  boosting 1.13. Decision (forward only): techlin, technology extrapolated, blend weight 0.2.
  Physics adds little for maize (wheat-built structure) -> maize physics to improve later
  (grain filling to physiological maturity, irrigation). Final grain maize model next (one test look).

## 2026-10-07 - Grain maize final model (steps 7-8, one test look)
- Settings: rue 3.55, tb 5.4, topt 27.8, t_heat 30.7, fg 0.51, remob 0.38, tech 0.47, tau 27,
  tech_lin 0.036/decade; technology extrapolated; blend weight 0.2 (forward). CV blend 0.94 (boosting 0.97).
- Test RMSE (blend / boosting / trend+boosting / WOFOST+weather): new years 1.69 / 1.71 / 1.66 / 1.85
  (n 357 - few districts publish grain maize in 2018-2025); new regions 1.30 / 1.32 / 1.29 / 1.32;
  both 2.25 / 2.27 / 2.61 / 2.72; drought 2.04 / 2.06 / 2.21 / 2.43.
- Verdict: tie with the best data models everywhere (differences 0.02), clearly better than WOFOST;
  the physics adds little for maize (weight 0.2). Errors large in test years (maize very weather-
  sensitive; small samples). Next: grain maize forecast + intervals (weight 0.2), outputs.

## 2026-10-07 - Silage maize step 6 (forward validation, fresh mass t/ha, mean ~44)
- Saturating technology: boosting 5.99; physics 7.60 (+2.2); hybrid 5.98; blend 0.6 5.89, 0.4 5.90,
  0.2 5.93. Saturating x linear: no gain (hybrid 6.01, best blend 5.91). Decision: saturating,
  extrapolated, blend weight 0.6 (forward). Final silage model + forecast + intervals + outputs next.
- Potato configured: tubers fill from flowering to the district's typical harvest day (training years) on the development clock; dry matter 0.22; ranges rue 1.5-4.5, tb 0-8, topt 12-26, t_heat 24-34. Forward validation running. phase2_outputs: potato maturity column = flowering (harvest = flowering + typical gap).

## 2026-10-07 - Grain maize steps 9-11 (forecast, ranges, outputs)
- Forecast RMSE new years blend / boosting: 12 wk 1.97/1.98, 8 wk 1.75/1.74, 6 wk 1.73/1.74,
  4 wk 1.68/1.71, 2 wk 1.67/1.70, harvest 1.68/1.71 - tie throughout.
- 80 % ranges: coverage 52-60 % new years, 38-45 % both new (target 80 %). Training out-of-fold
  errors (sd 0.96, 10/90 % -1.18/+1.12) are much smaller than the test errors (RMSE ~1.7): the few
  district-years with grain maize statistics after 2018 behave differently (volatile years,
  changing reporting districts). Not tuned on test; same open issue as barley (forward-validation
  residual pool as candidate).
- Outputs: data/processed/outputs/district_outputs_grain_maize.csv (range -1.18/+1.12 t/ha).

## 2026-10-07 - Silage maize final (steps 7-8, one test look) and potato step 6
- Silage test RMSE (t/ha fresh; blend / boosting / trend+boosting / WOFOST+weather): new years
  6.07 / 6.00 / 5.59 / 6.15; new regions 6.35 / 6.71 / 8.40 / 9.02 (hybrid 6.19); both 8.59 / 8.81 /
  9.84 / 10.42; drought 6.91 / 6.96 / 7.23 / 8.03. Better on new regions, both and drought; on new
  years trend+boosting is best (5.59). Simulated LAI max 8.7 (high), biomass 15.1 t/ha DM (plausible).
- Potato forward (fresh t/ha): boosting 6.83; physics 9.74 (bias -2.3); hybrid 7.26; blend 0.4 6.68,
  0.2 6.70, 0.6 6.77; techlin no gain. Decision: saturating, extrapolated, blend weight 0.4.
  Physics weak for potato (tuber phase approximated) - later improvement. Final potato chain running.
- Potato final (one test look; fresh t/ha; blend / boosting / trend+boosting / WOFOST+weather):
  new years 8.21 / 8.09 / 9.00 / 9.51; new regions 5.33 / 5.42 / 6.16 / 6.73; both 10.25 / 10.50 /
  14.34 / 14.38; drought 8.24 / 8.17 / 9.73 / 11.15. Tie with boosting (slightly better on new
  regions and both, slightly worse on new years); clearly better than WOFOST and trend models.
- Silage maize steps 9-11: forecast RMSE new years blend / boosting: 12 wk 7.63/7.70, 8 wk 6.37/6.37,
  6 wk 6.09/6.14, 4 wk 5.89/5.89, 2 wk 5.96/5.91, harvest 6.01/5.96 - tie. 80 % ranges new years
  76-80 % (well calibrated, like wheat). Outputs written.

## 2026-10-07 - DISTRICT LEVEL FINISHED (5 crops) - report section 8c (rev 35-36)
- Potato steps 9-11: forecast new years blend / boosting 12 wk 9.60/9.64, 4 wk 9.47/9.52, harvest
  8.32/8.28; 80 % ranges 59-61 % (too narrow). Outputs written.
- Summary files: data/processed/district_summary_test.csv, district_summary_forecast.csv.
- Ours better than boosting in 15 of 20 crop x test combinations (worse: barley new years +7 %,
  barley drought +9 %, silage new years +1.2 %, potato new years +1.5 %, potato drought +0.9 %).
  Barley: WOFOST+weather best on new years (0.96) and drought (0.89).
- Ranges calibrated for wheat (77 %) and silage maize (76-80 %); too narrow for barley (61-66 %),
  grain maize (52-60 %), potato (59-61 %).

## 2026-10-07 - District improvements (v8 work) started - owner: district improvements first
- Policy: every change decided by forward validation; one v8 test look per crop at the end
  (logged); 2026 harvest = final judge. 'Beat all tests' is the aim, not something tuned on test.
- Range method: forward predictions saved per district-year (forward(..., save)) for cut 2002
  (learn: errors 2003-2008) and cut 2008 (check: coverage 2009-2017), all 5 crops - running.
- SARAH sunlight: weather_daily_v8.npz (phase0_sarah_districts.py) - district SARAH means from 2021,
  scaled by HYRAS/SARAH 2016-2020 (median 0.951), ET0 recomputed. Pyranometers 2021+: RMSE 18.4 ->
  15.8 W/m2, r 0.985 -> 0.990, level unchanged (-4.1 %). Selected by env VISTA_WEATHER=v8
  (phase2_phenology.WEATHER_FILE); v7 default unchanged. Affects only years >= 2021 (test years) -
  no forward-validation check possible for it; it is an input-data correction justified by the
  pyranometer check alone.

## 2026-10-07 - District improvements (v8 work) started - owner: district improvements first
- Policy: every change decided by forward validation; one v8 test look per crop at the end
  (logged); 2026 harvest = final judge. 'Beat all tests' is the aim, not something tuned on test.
- Range method: forward predictions saved per district-year (forward(..., save)) for cut 2002
  (learn: errors 2003-2008) and cut 2008 (check: coverage 2009-2017), all 5 crops - running.
- SARAH sunlight: weather_daily_v8.npz (phase0_sarah_districts.py) - district SARAH means from 2021,
  scaled by HYRAS/SARAH 2016-2020 (median 0.951), ET0 recomputed. Pyranometers 2021+: RMSE 18.4 ->
  15.8 W/m2, r 0.985 -> 0.990, level unchanged (-4.1 %). Selected by env VISTA_WEATHER=v8
  (phase2_phenology.WEATHER_FILE); v7 default unchanged. Affects only years >= 2021 (test years):
  justified by the pyranometer check alone, no forward check possible.
- Physics variants running (forward 2008): grain maize grain filling to typical harvest
  (physvar=maize_harvest), potato tuber bulking from canopy closure (physvar=potato_canopy).
- Barley variety search: hybrid winter barley approved from 2008; share of German winter barley
  area (special harvest surveys, LLG Sachsen-Anhalt): 1.4 % 2009, 2.7 % 2010, 3.4 % 2011, 5.4 % 2012,
  6.8 % 2013, 3.2 % 2014 - too small to explain a 0.5-1 t/ha rise alone (hybrid gain ~5-10 %).
  Bundessortenamt publishes yearly seed-multiplication areas per variety (P1_gemFlaechen_LW_<year>.pdf;
  hybrid share of multiplication area 11.8 % 2012, ~18 % 2013; multi-row varieties 73.5 %) - a yearly
  variety-mix series could be built from these PDFs.
- Owner: try Bundessortenamt variety (seed-multiplication) data for barley later, after the other improvements.
- Physics variants (forward 2008, blend at the crop weight): grain maize to typical harvest: physics
  1.50 vs 1.53, hybrid 1.31 vs 1.30, blend 0.2 1.12 vs 1.12 -> no gain, NOT adopted (keep dough
  ripeness). Potato tubers from canopy closure: hybrid 7.05 vs 7.29, blend 0.4 6.63 vs 6.72 (boosting
  6.83) -> ADOPTED for potato v8 (GROWTH potato heading = canopy_closed).
- Range method check (eval_range_method.py, training data only; 80 % ranges, check on 2009-2017 fitted
  <= 2008): A cross-validation errors vs B forward errors (fitted <= 2002, 2003-2008): wheat 80.5 / 84.0 %,
  barley 71.7 / 77.4, grain maize 70.5 / 77.0, silage 79.5 / 76.4, potato 70.9 / 78.3; mean 74.6 / 78.6 %.
  Decision: method B for all crops (v8); final pool = forward errors 2009-2017 (fitted <= 2008);
  both-new pool widened by the CV spread ratio (both / years). phase2_intervals.py `forward` option,
  output forecast_intervals_<crop>_v8.csv.

## 2026-10-07 - DISTRICT v8 FROZEN (before its one test look) - md5 in data/processed/frozen_v8_md5.txt
- v8 = v7 + (1) SARAH-3 sunlight from 2021 in daily weather and in the monthly features (input data,
  pyranometer-checked), (2) range method B (forward errors 2009-2017; both-new widened), (3) potato
  tubers from canopy closure (forward 6.63 vs 6.72, physics refitted on <= 2017). All other physics,
  corrections and blend weights unchanged (fitted on training years only). Grain maize to-harvest
  variant and barley rising-trend term rejected by forward validation.
- Run with VISTA_WEATHER=v8; files *_v8. One test look per crop, logged next.
- Fix before the range step ran: potato v8 forward pool from the canopy-closure forward run (md5 appended).

## 2026-10-07 - v8 test look (one look per crop) - final models
- RMSE blend v7 -> v8 (boosting): wheat new years 0.944 -> 0.938 (0.958), new regions 1.113 -> 1.119
  (1.154), both 1.248 -> 1.237 (1.334), drought 0.969 -> 0.957 (1.009); barley 1.069 -> 1.051 (1.001),
  0.993 -> 0.998 (1.033), 1.292 -> 1.300 (1.310), 1.090 -> 1.078 (1.005); grain maize 1.686 -> 1.688
  (1.712), 1.299 -> 1.300 (1.320), 2.247 -> 2.238 (2.270), 2.038 -> 2.036 (2.058); silage 6.066 ->
  6.019 (5.998), 6.346 -> 6.323 (6.711), 8.591 -> 8.411 (8.815), 6.907 -> 6.809 (6.956).
- Small gains from SARAH sunlight in 2021+ (13 of 16 better or equal); win/lose pattern vs boosting
  unchanged except silage new years now nearly tied (6.02 vs 6.00). Potato refit pending.
- Potato v8 test look (blend v7 -> v8, boosting): new years 8.21 -> 8.22 (8.09), new regions 5.33 -> 5.30 (5.42), both 10.25 -> 10.20 (10.50), drought 8.24 -> 8.25 (8.17) - essentially unchanged.
- v8 forecasts (new years RMSE ours/boosting): wheat 12w 1.02/1.06, 8w 0.98/1.03, 4w 0.92/0.98, 0w 0.91/0.97;
  barley 1.09/1.11, 1.08/1.09, 1.01/1.03, 1.07/1.01; grain maize 1.97/1.98, 1.75/1.74, 1.68/1.71, 1.68/1.71;
  silage 7.63/7.68, 6.38/6.38, 5.90/5.89, 6.01/5.97; potato 9.57/9.64, 9.78/9.81, 9.47/9.52, 8.33/8.26.
- v8 80 % range coverage new years (v7 -> v8, over lead times): wheat 76-77 -> 75-77 %; barley 60-66 ->
  67-69 %; grain maize 52-60 -> 58-64 %; silage 76-80 -> 74-79 %; potato 58-61 -> 66-71 %. Both new:
  wheat 69-72 -> 71-74; barley 58-61 -> 64-66; grain maize 38-45 -> 47-56; silage 60-65 -> 61-65;
  potato 58-59 -> 65-69. Better for the three badly calibrated crops, still short of 80 % (test years
  2018-2025 more volatile than 2009-2017). district_summary_forecast_v8.csv.
- v8 standard outputs written for all 5 crops (data/processed/outputs/district_outputs_<crop>_v8.csv).
- Fix: v8 outputs now use the v8 range method (forward_pool); regenerated for all crops.

## 2026-10-07 - Barley variety mix (Bundessortenamt Beschreibende Sortenliste 2007-2026)
- 20 PDFs (61 MB) in data/raw/bundessortenamt. phase0_barley_varieties.py parses winter barley tables
  (registration year, seed-multiplication area per variety for the last 3-4 years). Area tables only
  in the 2013 and 2015-2026 editions -> seed years 2010-2025 (harvests 2011-2026).
- Index (barley_variety_index.csv): area-weighted registration year rises ~1 year per year (2006.0 in
  2010 -> 2020.3 in 2025); variety age stable 3.7-5.5 years; share of varieties <= 5 years old 0.48-0.79;
  multi-row share 0.73-0.77 until 2019, then 0.56-0.69; hybrid share 2-14 % of seed area.
- Finding: the variety turnover is smooth - no acceleration around 2012-2017 that would explain the
  barley yield jump; breeding progress through variety turnover is already a steady trend (what the
  technology term models). Hybrids too small. Not usable as a model input either (starts 2011, so a
  forward check fitted <= 2008 has no data). Barley's recent rise stays unexplained by variety data;
  remaining candidates: management (earlier sowing, fungicides), weather patterns, or changes in the
  official yield statistics method. Decision: no barley input change.

## 2026-10-07 - MODIS district greenness (district improvement, v9 candidate)
- Strategy (owner discussion): district model = MODIS backbone 2000-today (one consistent sensor for
  training and prediction); Sentinel-2 later as crop-specific district features (Thuenen crop maps
  2017+) harmonised to MODIS in the overlap; field model = Sentinel-2 (+ S1) with the district model as
  context. ESA WorldCover used as cropland mask only (no crop types, single year).
- phase0_modis_districts.py: MOD13Q1 v061 NDVI (Terra, 250 m, 16-day, Feb-Oct) via Planetary
  Computer (open; signed URLs - the collection token was refused); tiles h18v03 + h18v04; cropland
  = WorldCover 2021 class 40 share >= 60 % per 250 m pixel (80 m overview); 1.85 M cropland pixels.
  2018 check: Uckermark peak 0.72 late April, 0.45 by late June (drought ripening); Dueren peak 0.74
  mid-June; Munich district long season. ~2,200 clean pixels per district-date. All years running
  (4 parallel blocks, ~5 s per band read).
- MODIS collection complete 2000-2026 (27 years, 100 % districts; national median NDVI 2003 0.496
  lowest, 2022 0.553, wet years ~0.60-0.62). Gap: Terra composites on Planetary Computer end
  2025-06-26 (11 composites in 2025), 2026 patchy (7) - test year 2025 lacks July+ (maize / potato
  windows nearly empty). Options later: Aqua MYD13Q1 or the Sentinel-2 bridge.
- Forward test with MODIS features started (run_modis_forward.sh, VISTA_MODIS=1).

## 2026-10-07 - Forward test with MODIS features (fit <= 2008, predict 2009-2017; RMSE without -> with)
- Wheat: boosting 0.77 -> 0.75, hybrid 0.79 -> 0.76, blend 0.6-0.9 0.75-0.78 -> 0.73-0.75 (~ -3 %).
- Barley: boosting 0.93 -> 0.93, hybrid 0.87 -> 0.86, blend 0.8 ~0.875 -> ~0.865 (~ -1 %).
- Grain maize: boosting 1.13 -> 1.12, hybrid 1.30 -> 1.24, blend 0.2 1.12 -> 1.11 (~ -1 %).
- Silage: boosting 5.99 -> 5.94, blend 0.6 5.90 -> 5.89 (~0).
- Potato (v7 physics): boosting 6.83 -> 6.75, hybrid 7.29 -> 7.03, blend 0.4 6.72 -> 6.60 (~ -2 %).
- No crop gets worse; gains small to moderate (largest wheat and potato). Decision: MODIS features in
  v9 for all crops. Caveat: Terra data end June 2025 -> 2025 summer features missing (trees use
  their missing-value branch; never seen in training) - to be noted with the v9 test look.

## 2026-10-07 - DISTRICT v9 FROZEN (before its one test look) - md5 in data/processed/frozen_v9_md5.txt
- v9 = v8 + MODIS greenness features (2000+) in the learned correction and boosting, all crops;
  in-season forecasts use MODIS only up to the forecast date; ranges from forward errors of the
  model WITH MODIS (potato: canopy physics + MODIS forward run). Physics as v8. Files *_v9;
  run with VISTA_WEATHER=v8 VISTA_MODIS=1. Known gap: Terra MODIS ends June 2025.

## 2026-10-07 - Field-level groundwork: DLR CropTypes + Brandenburg field blocks
- Owner's aim clarified: a field map product (every field with crop + history, yield in-season and
  after harvest, within-field yield map, LAI curve, stage, harvest date, water demand/use/stress,
  soil, warnings, comparison with district) - like DLR CropTypes / PROMET but per field.
- DLR CropTypes v2 (download.geoservice.dlr.de/CROPTYPES/files): 10 m main crop per pixel, Germany,
  2017-2024 (one COG ~457 MB per year, EPSG:32632, CC BY 4.0; Gessner et al. 2025, Asam et al. 2022).
  Use: crop-specific Sentinel-2 district references (F1), crop history / previous crop per field,
  crops outside NRW. Owner: stream only the needed windows (no full download).
- Brandenburg DFBK (dfbk.zip, 96.9 MB, dl-de/by-2-0, updated 2026-10-05) downloaded to
  data/raw/brandenburg: 90,816 field blocks (permanent boundaries, 1:2,400) with category AL/GL,
  area, district. Plan: blocks x CropTypes 2017-2024 -> crop per block per year -> multi-year
  Brandenburg (held-out state) field pilot. Blocks may hold several fields -> use single-crop blocks.
- v9 bug fix (no effect on results where data exists): MODIS features for a forecast date before the crop window (potato 12 weeks) -> empty -> features missing; crashed potato forecast. md5 appended; potato forecast chain restarted.
- Brandenburg blocks x CropTypes 2017-2024 done (phase4_bb_blocks_croptypes.py ->
  data/processed/brandenburg/blocks_croptypes.csv, ~2 min/year streamed). Winter wheat main crop in
  ~3,100-3,600 blocks/year, only ~650-980 with purity >= 0.9: most blocks hold several fields.
  Plan (proposed to owner): field boundaries = Brandenburg Antrag parcels 2026 (antrag.zip, 119 MB,
  pending permission); crop per year 2017-2024 from CropTypes inside each 2026 field; a field-year
  counts if >= 90 % of its pixels show one crop; pure DFBK blocks as fallback. Use: S2 per field
  (sample first, credits to be approved), field model, district sums vs official Brandenburg yields
  2018-2024 (held-out state, pre-registered).
- Brandenburg Antrag 2026 downloaded (antrag.zip 119 MB, dl-de/by-2-0; antrag_land_gem.shp, cp1252,
  EPSG:25833): 288,938 parcels; winter wheat (codes 112/115) 8,966 fields, 161,658 ha (official ~160k),
  median 10.7 ha; variety and farm IDs anonymised.
- Field-yield sources for Brandenburg (search): (1) FernEE - official statistics pilot (Hesse lead,
  Statistik Berlin-Brandenburg since 2023): trains ML yield models on the official 'Besondere Ernte-
  und Qualitaetsermittlung' (BEE) sample harvests on real fields + InVeKoS field geometries + satellite;
  publishes district and field results for 2024 (district deviations -26 % .. +21 %); contact
  agrar@statistik-bbb.de. BEE = measured field yields - access only by cooperation / research data
  centre. (2) ZALF AgroScapeLab Quillow (Uckermark, 160 km2): monitoring incl. yield data; ZALF Open
  Research Data / request. (3) ZALF patchCROP. (4) large Brandenburg farms (yield-mapping combines),
  via Landesbauernverband / LELF.

## 2026-10-07 - PLAN ADDED: 2026 state-level forecast check (owner approved)
- Truth: BMEL-Statistik 'Getreideernte - 2. vorlaeufiges Ergebnis' 2026 (state yields, preliminary).
- Pre-registration before any look: frozen district model (v8, and v9 where MODIS exists - Terra
  ends June 2025, so 2026 runs without MODIS summer features), wheat and barley (others if
  published), area-weighted state means of district predictions vs preliminary state yields;
  metrics RMSE / bias over states, vs the boosting benchmark; one look; numbers may be revised.
- Prerequisites: district weather 2026 (HYRAS 2026 + SARAH 2026 + station wind / sunshine), 2026
  stage observations (DWD recent phenology), 2026 master rows (N extrapolated as for 2020-2025).
- No 2026 yields have been looked at.
- 2026-10-07: BMEL 0112060-0000.xlsx (16 KB) downloaded to data/raw/bmel - NOT opened until the 2026 predictions are frozen.
- 2026-10-07 evening: phase0_weather_2026.py written (rules above); owner hibernating - jobs resume / rerun tomorrow.
- 2026-10-08: weather_daily_v8_2026.npz built (HYRAS 2026 to early October, 279 days; SARAH; station
  wind; remaining days = 2016-2025 calendar-day means). 2026 Jan-Sep: mean 12.1 C, rain 456 mm,
  ET0 707 mm (dry, warm). v9 jobs resumed after hibernation (still running).
- 2026-10-08: all background jobs had stopped after hibernation / app restart; resumed with run_finish_district.sh (v9 resume per lead file + 2026 predictions v8 and v9, sequential).
- 2026 check rule change BEFORE any look: state weights = district crop area of the latest census year with an area (2020); areas exist only in census years. aggregate step separated (phase2_predict2026.py aggregate).
- Bug found: phase2_intervals wrote v9 ranges under the _v8 file name (tag hard-coded). Grain and
  silage maize v9 range files (written 15:36 / 15:40) renamed to _v9; their v8 range files were
  overwritten - the v8 coverage numbers are preserved in the log and district_summary_forecast_v8.csv.
  Fixed (tag = VTAG); wheat, barley, potato ranges not yet written -> correct names.
- Owner decision: platform = next high priority after the district work; v1 detailed and technical, English; field explorer in v2.
- Wheat and barley v9 range files (15:47) were also written under _v8 (process started before the fix) -> renamed to _v9; v8 numbers preserved in log / district_summary_forecast_v8.csv. Only potato _v8 range files remain genuine v8.

## 2026-10-08 - Comparison with AgroForecast-DE (owner question)
- AgroForecast-DE (geonextgis.github.io/AgroForecast-DE, model 'CropFusionNet', GCFS2.2 seasonal
  forecasts, 50 members): winter wheat + silage maize only; issue dates Apr/May/Jul 2026. Their own
  winter wheat performance file: train 2001-2019 RMSE 0.52; validation 2020-2022 RMSE 0.84 (R2 0.47);
  TEST 2023-2024 RMSE 0.98, R2 0.20, MAPE 12.3 % (n 546). State 2026 ranges q10-q90 only ~ +-0.03 t/ha.
- Ours on the same years 2023-2024 (all districts, n 554; already-used test years, re-slice only):
  v9 in-season forecast 4 weeks before harvest RMSE 0.97, R2 0.20, MAPE 12.9 %; 8 weeks 0.98 / 0.20 /
  13.0 %; final model at harvest 1.14 (2023 overestimate). -> essentially a tie on their 2 test years;
  ours is validated on 8 test years + held-out states, 5 crops, calibrated ranges (~75 % for wheat)
  vs their near-zero-width ranges.
- Their July 2026 state forecasts saved (data/processed/check2026/agroforecast_de_winter_wheat_2026-07_state.geojson;
  national district mean 6.9 t/ha, -5.1 %) BEFORE opening the BMEL truth -> added to the pre-registered
  2026 state check as a second comparison (same truth, both unseen).

## 2026-10-08 - 2026 PREDICTIONS FROZEN (before opening the BMEL truth) - md5 in check2026/frozen_2026_md5.txt
- National area-weighted 2026 predictions (blend / boosting), v8 | v9: wheat 7.94 / 7.83 | 7.56 / 7.28;
  barley 7.30 / 7.48 | 6.99 / 7.24; grain maize 10.04 / 10.18 | 9.53 / 9.55; silage 46.6 / 47.7 |
  44.5 / 44.8; potato 43.5 / 44.6 | 42.1 / 42.2 t/ha. v9 uses the partial 2026 MODIS data (to July).
- AgroForecast-DE July 2026 wheat state forecasts saved. Now opening data/raw/bmel/0112060-0000.xlsx.

## 2026-10-08 - RESULT: 2026 check (one look; national only - the BMEL table has no states)
- Truth: BMEL/Destatis 'Getreideernte 2026, 2. vorlaeufiges Ergebnis' (23 Sep 2026), national yields:
  winter wheat 7.26, winter barley 7.78, grain maize + CCM 7.76 t/ha (maize very preliminary - harvest
  under way). Silage maize and potato not in the table. Pre-registered state comparison impossible ->
  national, n = 1 per crop (weak evidence).
- Errors (prediction - truth, t/ha): wheat ours v8 +0.68, v9 +0.30, boosting v8 +0.57 / v9 +0.02,
  AgroForecast-DE July (area-weighted states) -0.40; naive 2025 +0.64, naive 2020-25 mean +0.32.
  Barley ours v8 -0.48, v9 -0.79, boosting -0.30 / -0.54 (all too low again: the recent-rise issue;
  MODIS made it worse). Grain maize ours v8 +2.28, v9 +1.77, boosting +2.42 / +1.79: 2026 (warm, dry,
  456 mm Jan-Sep) cut maize ~23 % below 2025 and no model caught it.
- Reading: wheat v9 reasonable (4 %) but not better than simple references; barley level bias persists;
  maize drought response clearly too weak - a priority for the maize physics. Final judgement: official
  district yields 2026 (spring 2027).
- Report rev 38: section 8e (v9, AgroForecast-DE comparison, 2026 check). DISTRICT LEVEL FINISHED FOR THIS PHASE. Next (owner): platform v1, then field level.

## 2026-10-08 - PLAN: district quality release v10 BEFORE platform v1 (owner)
Goal: defensible, precise model for the agricultural community; time is allowed, results matter.
Workstreams (every change decided by forward validation on training years, frozen, one test look;
final judge official district yields 2026 in spring 2027):
 A lead-time-dependent model mixing (forward forecasts on 2009-2017)
 B season-rest scenarios: warming-adjusted past years, analog-year weighting, DWD GCFS2.2 hindcasts
 C ranges with an explicit year-wide error component (all crops)
 D maize drought / heat physics (silking window, rooting, irrigation share) - literature first
 E barley level: residual analysis by state and year, sowing / management, statistics method change
 F satellite after MODIS: Sentinel-2 district bridge (with field F1) so v10 runs live from 2026
 G potato and silage physics review
 H operations: automatic data updates, versioned releases, data checks, seasonal pre-registered
   evaluation published on the platform

## 2026-10-08 - Blueprint for a defensible model (report section 8f, rev 39)
- Wide search: Jakeman et al. 2006 good modelling practice; JRC MARS MCYFS (25 years, own accuracy
  assessment: soft wheat MAPE 3.1 % FR .. 21 % PT, 1993-2015); AgMIP / AgML CY-Bench public
  benchmark; leakage literature (random splits leak); probabilistic verification (CRPS, PIT,
  reliability); farmer-trust studies (explainability, transparency, evidence); EU code of conduct on
  agricultural data sharing (farmer owns data, consent, GDPR pseudonymisation); Destatis harvest
  reporters' in-season estimates (June, July/Aug, Aug/Sep) as the benchmark users already have.
- 10 pillars with status; v10 priorities updated: (1) official in-season benchmarks (Destatis
  reporters, JRC MARS) + last-year persistence, (2) year-wide error term + CRPS/PIT/below-normal
  probability, (3) maize drought/heat physics + stress-test matrix, (4) lead-time mixing + scenarios,
  (5) automatic data checks, one-command reproduction, model card, (6) external review, CY-Bench,
  publication, yearly pre-registered evaluation.

## 2026-10-08 - v10 priority 1: official in-season benchmarks - data
- GENESIS 41241 has final annual values only (no in-season estimates).
- Destatis Fachserie 3 R. 3.2.1 'Wachstum und Ernte - Feldfruechte' (harvest reporters' in-season
  estimates, Germany + states): series DESerie_mods_00000335 in the Statistische Bibliothek (API search);
  88 issues 2005-2022 indexed (data/raw/fachserie_index.json); spreadsheet (xlsx) versions downloaded for
  86 issues (149 MB, data/raw/fachserie); 2 issues without spreadsheet (2005 annual, Aug/Sep 2011).
- Owner shared Halder et al. 2025 (Smart Agricultural Technology; ZALF / Uni Bonn): crop-type mapping
  with Sentinel-1 + Sentinel-2 10-day composites, BiLSTM ~93 % overall accuracy, trained on EuroCrops
  for Lower Saxony, NRW, Brandenburg (2021, 2023); data on request. Relevance: recipe for our own
  current-season crop maps (DLR CropTypes ends 2024) in F1 / in-season district features; not yields.

## 2026-10-08 - v10 priority 1: benchmark data extracted + comparison rules (pre-registered before MARS look)
- Destatis in-season (phase0_destatis_inseason.py -> data/processed/benchmarks/destatis_inseason.csv,
  3530 values, Germany + 13-15 states): wheat June 'Erste Schaetzung' 2014-2022 (reporters' pre-harvest
  estimate), wheat/barley July-Aug + Aug-Sep 'vorlaeufiges Ergebnis' (after harvest, mainly sample
  harvests), potato + silage maize Aug-Sep, grain maize Sept 2005-2009 only, year-end finals. No
  in-season barley estimate in June. Parser fixes: cereal totals 'ohne / einschl. Koernermais' excluded,
  state names with dot leaders, labels above the 'Ertrag' header, stage from the issue number.
- Publication schedule (Fachserie 'Uebersicht' sheets): June issue end Jul/early Aug, Jul-Aug early Sep,
  Aug-Sep mid Oct. Users see the numbers only then.
- JRC MARS Bulletins: index of 221 issues via the JRC Publications Repository search API
  (data/raw/mars/bulletins_index.csv); owner approved 2009-2026 (165 issues, ~1.65 GB transfer; each PDF
  read and deleted at once, yield-table page text kept in data/raw/mars/pages). Germany rows parsed
  from the country tables (phase0_mars_bulletins.py -> mars_germany.csv).
- Comparison rules (phase5_benchmarks.py; re-slice of the frozen v9/v8 test forecasts 2018-2025,
  nothing refitted):
  truth = Destatis 41241 final (Germany, states), grain maize from Fachserie year-end issues, else
  area-weighted district mean (flagged); ours = district forecast medians weighted by latest census
  area; for each official figure our LATEST forecast date on or before its date (publication date
  'pub', and stricter end of survey month 'ref'); baselines last-year persistence and 5-year mean;
  RMSE and MAE national (n = years) and states.
- Honesty note: the first Destatis-vs-ours numbers (Aug-Sep issues, potato / silage / cereals) were
  printed while building the script, before these rules were written; no rule was changed after that
  look except adding the 'pub' publication dates found in the schedule sheets. MARS not yet looked at.

## 2026-10-08 - v10 priority 2: honest ranges - PRE-REGISTRATION (training data only)
Problem: 80 % ranges catch the truth < 80 % in test years for barley, maize, potato; errors in one year
are shared by all districts (2018), but ranges draw each district's error independently -> state /
national ranges and a 'below normal' probability would be far too confident.
Method choice (eval_ranges_v10.py), v8 forward files (no MODIS cut 2002 exists): build each method
from forward errors fitted <= 2002 (years 2003-2008), score on forward errors fitted <= 2008
(2009-2017, true weather). Candidates:
  A  current: pooled district errors, drawn independently per district
  B  year + local: error = year-wide part (mean error of all districts in that year) + local rest;
     year part drawn from a Student-t with the pool's year sd, n_years - 1 degrees of freedom and the
     (1 + 1/n) prediction factor (few years -> wider); local part from the empirical local pool
  C  as B with a state-year part (year+state mean) between year and local
Scores (2009-2017): district 80 % coverage + worst year, CRPS, PIT spread; state and national
area-weighted 80 % coverage and CRPS from joint draws (one year draw shared by all districts); Brier
score of P(yield < district's previous 5-year official mean) vs the base rate.
Choice rule: lowest national + state CRPS among methods whose district coverage is within 0.70-0.90;
ties (CRPS within 2 %) -> the simpler. Then apply to v9 (pool = v9 forward errors 2009-2017),
freeze (md5), one look at the test years 2018-2025 (district, state, national).

## 2026-10-08 - v10 priority 2: range method check (training years only) - RESULT + AMENDMENT
eval_ranges_v10.py -> range_method_v10.csv (build 2003-2008 forward errors, score 2009-2017):
  80 % coverage (district / state / national), A = current, B = year-wide + local, C = + state-year
  wheat   A 0.84 / 0.44 / 0.33   B 0.90 / 0.93 / 1.00   C 0.91 / 0.97 / 1.00
  barley  A 0.78 / 0.33 / 0.22   B 0.83 / 0.76 / 0.89   C 0.83 / 0.91 / 0.89
  maize   A 0.76 / 0.40 / 0.33   B 0.81 / 0.70 / 0.89   C 0.81 / 0.77 / 0.89
  silage  A 0.76 / 0.40 / 0.22   B 0.78 / 0.53 / 1.00   C 0.79 / 0.76 / 1.00
  potato  A 0.78 / 0.65 / 0.56   B 0.82 / 0.89 / 1.00   C 0.82 / 0.92 / 1.00
  Brier (below previous-5-yr mean) B vs base rate: wheat 0.226/0.244, barley 0.217/0.242,
  maize 0.178/0.233, silage 0.217/0.249, potato 0.228/0.247 (all better than base rate).
- The pre-registered choice rule (lowest state+national CRPS among district coverage 0.70-0.90) would
  give wheat A (B/C district coverage 0.90/0.91 just over), barley B (C within 2 %), maize B, silage B
  (tie), potato A (national CRPS on only 9 years). The rule omitted state/national coverage; A
  catches the national yield in only 22-56 % of years. AMENDMENT (before any test look, training data
  only): method B for all crops. Both the rule's outcome and the amendment are recorded here.
- National coverage 1.00 for B with 9 years suggests B is somewhat wide nationally (Student-t with 5
  degrees of freedom from 6 build years); kept, as the conservative side, to be revisited with more years.
- v10 ranges FROZEN before the test look (md5 in data/processed/ranges_v10/frozen_md5.txt):
  phase2_ranges_v10.py (joint draws: shared weather scenario + shared year error from v9 forward
  errors 2009-2017 + local error, both-new local x CV ratio; district/state/national ranges and
  P(below previous-5-yr mean)), score_ranges_v10.py (coverage, interval score, PIT tails, Brier vs
  base rate; state/national vs official truth). Wheat ranges generated for a bug check (set-up
  numbers only printed, no scores).

## 2026-10-08 - v10 priority 3: maize drought / heat - literature + stress-test matrix (training years)
Literature (short): kernel number of maize is set +-2 weeks around silking; heat (Tmax > ~33-35 C at ear
level) and water deficit in that window cut kernel set / raise abortion (Rattalino Edreira et al. 2011;
Eyshi Rezaei et al. 2015 (crop model sensitivity to 2-week flowering extremes, Austria); yields fall
steeply above ~29-30 C daily exposure (Schlenker & Roberts 2009); in Europe drought, not heat, drives
losses in low-yield years and drought stress intensifies for maize (Webber et al. 2018, Nat. Commun.);
German silage maize: soil-moisture anomalies in Aug/Sep cut yield > 10 % (Peichl et al. 2018 NHESS).
Our model: water factor cuts daily growth; heat only in the first 20 % of grain fill (after
flowering); NO water-stress effect on kernel set at silking; no pre-silking heat; no drought-heat link.
Stress matrix (stress_matrix.py -> stress_matrix.csv; forward errors 2003-2017, bias = pred - official):
  grain maize  normal/not hot -0.62, dry/hot +0.15 (n=597): ~0.8 t/ha (8 %) too optimistic when dry+hot
  silage maize normal summer -0.67, dry summer +0.91: ~1.6 t/ha too optimistic in dry summers
  potato       dry/not hot -2.46, dry/hot -0.02: ~2.4 t/ha too optimistic when hot
  wheat        +0.3 t/ha relative in dry/hot; barley: no clear stress pattern (level bias -0.27)
Candidate changes (to be chosen by forward validation 2009-2017 + stress matrix, then frozen):
  M1 kernel-set water stress around silking (grain maize): grain x exp(-c_ws x mean(1 - fw) in
     flowering -10..+20 d)
  M2 heat window from 10 d before flowering + canopy warming under drought (Tmax + d x (1 - fw))
  M3 silage / potato: drought-accelerated senescence + late-summer water (review r_dry, zmax limits)
  M4 stress-window indices (window water balance, hot days) as features of the learned correction

## 2026-10-08 - v10 priority 2: TEST LOOK (one look, frozen code verified by md5) - RESULT
score_ranges_v10.py -> data/processed/ranges_v10/score_v10.csv. 80 % coverage at lead 4 weeks,
district v10 (v9) / state v10 / national v10 (n years):
  wheat 0.74 (0.75) / 0.63 / 1.00 (8); barley 0.72 (0.68) / 0.53 / 0.63 (8); grain maize 0.58 (0.56) /
  0.45 / 1.00 (4); silage 0.78 (0.76) / 0.53 / 0.75 (8); potato 0.67 (0.68) / 0.63 / 0.63 (8).
  Interval scores v10 ~ v9 at district level (+-2 %). Brier P(below previous-5-yr mean), district /
  national vs base rate: wheat 0.256/0.260, 0.148/0.257; barley 0.268/0.271, 0.135/0.239; maize
  0.246/0.265, 0.052/0.267; silage 0.186/0.252, 0.009/0.242; potato 0.276/0.256, 0.256/0.239 (worse).
- Verdict: structural fix works for national ranges and national below-normal probabilities (wheat,
  barley, maize, silage); district ranges unchanged and still < 80 %; state ranges too narrow (45-63 %);
  potato probabilities no better than base rate. Cause: year-wide error learned from 9 calm forward
  years (2009-2017) is smaller than in 2018-2025 (2018/2019/2022 droughts); maize systematic drought bias.
- Next (training data only, new pre-registration): year-wide error from rolling forward cuts (~20
  years incl. 2003 and 1990s droughts); maize physics (step 3). No change is made from this look.

## 2026-10-08 - v10 priority 1: benchmark RESULT (full MARS 2018-2025 incl. OCR) + report rev 40
- OCR pass (phase0_mars_ocr.py, owner-approved re-download of 38 issues, ~814 MB): 2021, 2023-2026 read;
  PDFs deleted; data/raw/mars/ocr keeps OCR words + grey page images (42 MB). Crop identity check added
  in phase5_benchmarks.py (row 5-yr avg within 12 % of official 5-yr mean; drops rows the text order
  attached to the wrong crop, e.g. 2022 'potato 2.1').
- National RMSE % of mean yield, MARS vs ours (test years; grain maize 2018-2021 only, 4 years):
  wheat May 7.2/5.4, Jun 7.0/3.9, Jul 4.9/4.5, Aug 3.4/4.2; barley May 9.2/11.3, Jun 7.5/8.3, Jul
  4.9/7.7, Aug 4.8/7.7 (MARS better throughout); grain maize Jul 7.7/9.4, Aug 7.3/4.6, Sep 14.3/3.8, Oct
  14.3/2.4; silage Jul 7.9/9.0, Aug 4.7/4.6, Sep 4.3/3.7, Oct 3.8/4.1; potato May 10.5/9.1, Jun
  11.3/10.4, Jul 8.8/8.4, Aug-Oct 5.8-4.4/6.3. Persistence worse than both everywhere.
- Report rev 40: sections 8g (benchmarks, chart widget 90d0b24f-50ae), 8h (step 2 ranges result),
  8i (step 3 status).

## 2026-10-08 - v10 step 3: first variant results (forward 2009-2017, training only)
- Grain maize physics variants (eval_step3.py -> step3_variants_stress.csv): blend RMSE 1.121 current,
  1.115 kernel_water, 1.117 drought_heat, 1.119 stressfeat, 1.120 all; stress gap (bias dry+hot minus
  bias normal) 0.69 -> 0.68 at best. Fitted c_ws ~0.012 and d_dry ~0.13 (near zero): data do not support
  the mechanisms as formulated. Grain maize forecast = 0.2 hybrid + 0.8 boosting, so physics has
  little leverage; the stress optimism sits mostly in the boosting part (monthly weather only).
- Added variant 'stressboost': flowering-window water balance and hot days as master columns (inputs of
  boosting and correction); queued for grain and silage maize with and without 'stressfeat'.
- Reading note: part of any stress gap is generic shrinkage (imperfect models over-predict low and
  under-predict high yields); target = clearly smaller gap, not zero.

## 2026-10-08 - v10 deep diagnosis (owner: "beat all cases"; training years only)
Target (pre-harvest): at every forecast date, every crop, national/state/district: more accurate than
MARS, Destatis reporters and simple rules, with ranges holding 80 %. After-harvest measurements excluded.
- Error decomposition (diagnose_errors.py -> error_decomposition.csv; forward errors 2003-2017),
  share of squared error year / state-year / district-constant / rest:
  wheat .26/.22/.20/.32, barley .29/.23/.17/.31, grain maize .21/.19/.17/.43, silage .09/.29/.21/.41,
  potato .19/.18/.23/.39.
- District offsets forward test (test_district_offsets.py): offsets from 2003-2008 forward errors (year
  part removed, shrinkage k = 3 fixed a priori) applied to 2009-2017: RMSE wheat 0.768->0.710 (-7.5 %),
  barley 0.873->0.836 (-4.3 %), grain maize 1.119->1.081 (-3.4 %), silage 5.90->5.36 (-9.1 %), potato
  6.72->6.20 (-7.7 %). State offsets help only silage (-8.4 %) and potato (-2.9 %). Candidate v10 change
  (only where a district has history; new regions get none).
- Attack plan by error part: year-wide -> fusion with MARS / Destatis estimates once published,
  soil moisture (ERA5-Land, ESA CCI, DWD), MODIS land surface temperature; state-year -> regional
  satellite and soil moisture; district -> offsets (above); rest -> noise floor estimate; barley level
  -> recent-trend anchoring; ranges -> ~20 forward years.

## 2026-10-08 - v10 new inputs: soil moisture + canopy temperature (owner approved both downloads)
- A. DWD AMBAV 2.0 crop soil moisture (grids_germany/daily/soil_moisture/<wheat|maize>, 1 km, % nFK,
  0-60 cm, 1991-2026, ~134 MB per crop-year, ~9.6 GB transfer): phase0_dwd_soilmoisture.py downloads one
  file, averages per district (all valid cells; VG250 on the GK3 grid, cache district_grid_dwd_sm.npz),
  deletes it -> data/processed/soil_moisture/sm_<crop>_<year>.npz. Check 2018 maize national mean:
  Apr 107, Jun 84, Jul 33, Aug 13 % nFK (drought visible).
- B. MODIS LST MOD11A2 (Terra) + MYD11A2 (Aqua) 8-day 1 km 2000-2026 via Planetary Computer, streamed:
  phase0_modis_lst.py -> data/processed/modis_lst/lst_<MOD|MYD>_<year>.npz (daytime LST over cropland
  >= 50 % WorldCover, QC good/other). Index later: LST - HYRAS Tmax anomaly. Continuity note: MODIS ends
  within a few years -> VIIRS / Sentinel-3 LST for live use.
- Test design (as step 3): district features (monthly + flowering window), forward validation
  (<= 2008 -> 2009-2017), judged on RMSE, stress gap, year / state-year error shares, normal years;
  freeze, one test look.

## 2026-10-08 - v10 step 3 final variant results + combined forward test started
- Step 3 (eval_step3.py): grain maize stressboost stress gap 0.69 -> 0.48 but RMSE 1.121 -> 1.124 (fails
  "no worse overall" alone; kept as candidate); silage drought_heat gap 1.05 -> 0.77, RMSE 5.893 -> 5.876
  (best); other variants no gain.
- New variants 'soil' (newdata_features.soil_features) and 'lst' (lst_features) in phase2_growth.
- MODIS LST reading parallelised (8 threads); first year MYD 2018 plausible (July cropland LST 36 C).
- run_combined.sh: round 1 soil (wheat, barley, potato+canopy, grain maize +/- stressboost, silage
  drought_heat+soil, with recomputed baselines), round 2 + lst. eval_combined.py adds district offsets
  (approximation: offsets from current model's 2003-2008 errors; finalists to be redone with cut 2002).
- PRE-REGISTERED selection rule for the combined test (owner: the chosen combinations may not be the
  best): every crop gets all combinations of {soil, stressboost, drought_heat, soil+lst...} (round 3,
  run_combined2.sh), each fitted <= 2008 (scored 2009-2017) AND <= 2002 (scored 2003-2008). A combination
  is chosen only if it beats the current model in BOTH periods on RMSE (district offsets applied
  consistently) and does not increase the stress gap; among those, lowest mean RMSE over both periods;
  ties (< 1 %) -> fewer ingredients. Round 1/2 are screening only.
- SCREENING RESULT + AMENDMENT (21:57, training years only): raw soil-moisture inputs made wheat and
  barley worse (2009-2017 RMSE wheat 0.761 -> 0.789, barley 0.869 -> 0.921, bias more negative).
  Diagnosis: inputs missing before 1991 coincide with the old-technology era (missingness = time
  signal); absolute levels mix soil type with yield potential. Amendment (before the full run): soil and
  LST inputs as district anomalies vs a fixed reference period preceding both forward periods (soil
  1991-2002; LST 2000-2008, caveat cut 2002), missing -> 0. Old soil outputs deleted; full combined run
  (run_combined2.sh) restarted with all combinations; selection rule unchanged.

## 2026-10-09 - overnight: part 1 done; LST crash fixed
- Combined test part 1 (all combinations without LST, cut 2008 and cut 2002, 52 runs) finished 04:14,
  no errors.
- MODIS LST run stopped at MOD 2004 (one QC file unreadable on the server -> RuntimeError). Fix:
  unreadable composites are skipped and logged; remaining years Aqua only (MYD 2002-2026), which the
  features use (Terra only a fallback for 2000-2001, before both test periods).
- PART 1 RESULT (eval_select.py -> combined_select.csv; pre-registered rule): no combination beats the
  current model in BOTH periods for any crop. Soil anomalies: period A (2003-2008) wheat -4.0 %, potato
  -2.0 %, maize -0.6 %, barley +0.5 %, silage +2.0 %; period B (2009-2017) +1.6 to +2.6 % for all ->
  not robust. Step 3 variants within +-1 %. District offsets (learned from each combination's own
  period-A errors) on period B: wheat -6.4 %, barley -4.6 %, maize -3.5 %, silage -9.0 %, potato -7.7 %.
- Design gap fixed: part 2 now also tests LST alone, LST+stressboost, LST+drought_heat (all earlier LST
  combinations included soil).

## 2026-10-09 - barley diagnosis + PRE-REGISTRATION: annual level update (training years only)
- Barley forward errors by year (fit <= 2008): -0.37 (2009) ... -1.06 (2017); observed national trend
  2009-2017 +0.149 t/ha/yr vs predicted +0.051 -> the frozen model lags fast barley progress. Wheat: no lag.
- Operational realism: a product is updated every year with the newest official yields; the forward test
  froze the model for 6-9 years. Proposed: annual level update using only information available at
  forecast time: national error of years t-1..t-3 (national yields of t-1 are final by early t) and
  district offsets from district errors up to t-2 (district statistics appear with ~1 year delay).
- Variants (fixed a priori): N3 = national mean error of last 3 available years; N5 = last 5; D = district
  offsets (year part removed, k = 3) from all errors up to t-2; N3+D; N5+D. Within each period the
  update uses only errors of earlier years of the same forecast stream (period A starts without history;
  period B may use period-A errors as history, as an operator would).
- Rule: a variant is adopted for a crop if it lowers RMSE in BOTH periods; among those lowest mean RMSE;
  ties < 1 % -> simpler (D < N3 < N5 < N+D).
- RESULT (test_level_update.py -> level_update_test.csv): change in RMSE period A / period B:
  wheat D -6.3 / -7.2 %; barley D -5.7 / -6.5 % (N3+D -15.2 % in B but +4.6 % in A -> rejected);
  grain maize N3+D -9.7 / -6.8 % (chosen; D -5.3 / -6.7); silage D -8.2 / -12.8 %; potato D -5.7 /
  -10.7 %. National-only updates noisy (year errors are mostly weather). ADOPTED for v10 (pre-registered
  rule): annual district offsets (all crops) + national 3-year update for grain maize.
- Next (pre-registration below): refit schedule test, because frozen models lag fast breeding (barley).
- PRE-REGISTRATION refit schedule (run_refit.sh, training only): forward cuts 2011 and 2014 in addition to
  2008; 'refit every 3 years' = predictions 2009-2011 from cut 2008, 2012-2014 from cut 2011, 2015-2017
  from cut 2014, compared with the frozen cut-2008 model, both + annual district offsets. Adopted if it
  lowers 2009-2017 RMSE for the crop by > 1 % (and is the operating procedure anyway); if adopted, the v10
  test look will use the same expanding schedule (each test year predicted only from data before it).
- REFIT RESULT (test_refit.py -> refit_test.csv; 2009-2017 RMSE, both with annual district offsets):
  wheat 0.706 frozen vs 0.734 refit (+3.9 %), barley 0.813 vs 0.742 (-8.6 %, ADOPTED), grain maize 1.046 vs
  1.042 (-0.4 %, below 1 %), silage 5.140 vs 5.221 (+1.6 %), potato 5.921 vs 6.040 (+2.0 %). Barley bias
  2017 -1.06 -> -0.72. v10: barley refit every 3 years (test look with the same expanding schedule:
  fit <= 2017 -> 2018-2020, <= 2020 -> 2021-2023, <= 2023 -> 2024-2025); other crops frozen.
- Rolling forward cuts 1990 and 1996 done (all crops): forward errors now 1991-2017 (27 years) for the
  range recalibration.

## 2026-10-09 - v10 FREEZE DECISIONS + PRE-REGISTERED TEST LOOK (before any v10 test result)
v10 = v9 (incl. MODIS greenness) + the following, all chosen on training years:
 1 annual district offsets D: offset_d(t) = sum of district errors (year part removed) of years <= t-2 /
   (n + 3); error history = v9 forward errors 2009-2017 (fit <= 2008, MODIS) + v9 out-of-sample errors of
   test years <= t-2 (end-of-season prediction = lead-0 median). Applied to every lead's median forecast.
 2 grain maize: + national update N3 = mean national error of years t-1..t-3 (same error history; national
   yields of t-1 are final by early t).
 3 barley: refit every 3 years: test years 2018-2020 from the v9 model (fit <= 2017), 2021-2023 from a
   model refitted on all rows <= 2020 except the held-out states (env VISTA_REFIT=2020: role test_years ->
   train for years <= 2020; physics, correction, boosting refitted), 2024-2025 from VISTA_REFIT=2023.
   Held-out states (test_both) stay out of every refit. Offsets D on top as in 1.
 4 ranges: method B; year-wide error from the rolling forward errors 1991-2017 (v8 cuts 1990, 1996, 2002,
   2008; after offsets D computed the same way within that history), local part from v9 forward errors
   2009-2017 after D; both-new local x CV ratio as before.
 Not in v10: soil moisture, stressboost, drought-heat (failed two-period rule); MODIS LST undecided ->
 possible v10.1 if it passes later.
Scores (one look): district RMSE per lead vs v9 (test_years, test_both); national + state benchmark vs
MARS / Destatis / persistence (phase5_benchmarks, new 'ours_v10'); range coverage, interval score, Brier.
- DISCLOSURE (08:56): phase2_growth.py main() prints its standard comparison table incl. test rows; the
  barley refit-2020 run thereby printed end-of-season test RMSE (new years, blend 0.89 t/ha; mixes the
  refit model's 2021-2025 rows). Not requested, not used: all v10 rules were frozen before; nothing changed.
- Sentinel-2 test: HR-VPP PPI collection fails server-side (load_stac error); CLMS HRL crop type codes
  identified from a 5x5 km sample (1110 wheat 31 %, 1120 barley 13 %, 1130 maize, 1310 potatoes,
  1430 rapeseed 14 %; no winter/spring or grain/silage split). Test 2 = S2 L2A NDVI, SCL mask, dekad
  median, wheat, Thuringia 2022 (phase0_s2_district_test2.py).
- Sentinel-2 test 2 (full coverage, wheat, Thuringia 2022) failed after 22 min (cluster shuffle errors);
  cost 39 credits, 37 gigapixels for one state / crop / year -> full coverage of Germany unaffordable.
  New design (phase0_s2_district_sample.py): 100 m mode crop map, interior cells (3x3 same crop), up to 50
  random cells per district x crop, 40 m squares, per-scene NDVI mean server-side, dekads locally.
  Thuringia 2022: 74 district x crop samples (wheat 1050, barley 1050, maize 929, potato 180 cells).
- Sentinel-2 SAMPLED test (Thuringia 2022) SUCCESS: 30 min, 55 credits, 5.8 gigapixels, 4316 values,
  ~56 valid dates per district x crop. Dekad medians: barley peak 0.81 late May, senescence mid-June;
  wheat peak 0.83 late May/early June, drop mid-July; maize from mid-June, peak 0.70 Jul-Aug; potato peak
  0.78 mid-July. MODIS cropland mix: peak 0.77 late May, flat 0.37-0.39 in August (crops blurred).
  Cost estimate Germany ~950 credits / year at these settings -> reduce (25 cells, Apr-Sep, cloud <= 60 %)
  to ~300-400 / year before the full run (owner decision on credits).
- Sentinel-2 CHEAP test (25 cells, Apr-Sep, cloud <= 60 %): 22 credits (vs 55), 2.05 gigapixels; dekad
  district-crop medians vs full sample: correlation barley 0.986, maize 0.980, potato 0.995, wheat 0.986,
  mean abs difference 0.01-0.03 NDVI -> adopted settings. Estimate Germany ~380 credits / year,
  2017-2025 ~3,400, + 2026 ~3,800 (owner decision; possibly split over two months).
- CLMS crop-type maps available 2017-2023 only (2024: server error, 2025-26 not published / rate limited).
  Owner approved option A (2022-2026, ~1,900 credits); started 2022 + 2023 for all states (~760 credits,
  run_s2_germany.sh); proposed remaining budget for 2019-2021; 2024 via DLR CropTypes; 2025-26 need own
  crop classification from S2 time series (also needed for live use and field level).
- Owner approved remaining S2 budget for 2019-2021 (queued after 2022/2023 streams).
- S2 Germany run fixes: crop map via batch job (sync limit 20000 px exceeded for large states), retry on
  429 rate limit; streams restarted (2022, 2023; then 2021+2019 and 2020).
- Owner: Sentinel-1 gap-filling belongs to the field level; order = district goal first, then field level.

## 2026-10-09 - v10 TEST LOOK (one look; freeze verified, 35 fingerprints OK) - RESULT
Disclosures before the look: barley refit-2023 run printed its standard test table (not used); 2026 v10
national predictions printed next to the known 2026 national truth (wheat 7.62 vs 7.26, barley 7.13 vs
7.78, maize 9.70 vs 7.76 prelim.) - an already-used truth, not part of the test.
- District RMSE v10 vs v9 (score_v10_district.py -> v10_district_score.csv), all leads, known districts /
  held-out states: wheat -11..-13 % / -16..-18 %; barley -5..-12 % / -8..-10 %; grain maize -4..-6 % /
  -4..-7 %; silage -13..-20 % / -15..-22 %; potato -7..-9 % / -14..-17 %. v10 better in every cell.
- Ranges v10b (score_v10b.csv), lead 4, 80 % coverage district / state / national: wheat 0.76/0.55/0.50,
  barley 0.68/0.42/0.50, maize 0.58/0.40/0.50(n=4), silage 0.81/0.73/1.00, potato 0.63/0.55/0.75.
  Interval scores better than v9 (sharper forecasts) but state/national ranges narrower than the first
  v10 ranges and under-cover: the 27-year year error (wheat sd 0.20) is smaller than in 2018-2025.
  Range goal NOT met -> v11 (adaptive ranges).
- National vs MARS, RMSE % of mean (v10 / MARS): wheat May 5.3/7.2, Jun 3.9/7.0, Jul 4.5/4.9, Aug 4.4/3.4;
  barley May 11.5/9.2, Jun 7.7/7.5, Jul 6.5/4.9, Aug 6.6/4.8; grain maize Jul 10.4/7.7, Aug 4.1/7.3,
  Sep 2.6/14.3, Oct 1.8/14.3 (4 yrs); silage Jul 8.6/7.9, Aug 4.0/4.7, Sep 2.6/4.3, Oct 2.6/3.8;
  potato May 9.4/10.5, Jun 9.2/11.3, Jul 8.3/8.8, Aug 5.4/5.8, Sep 5.4/5.6, Oct 5.4/4.4.
  v10 better in 15 of 22 cells; behind: wheat Aug, barley May/Jul/Aug (Jun ~ tie), maize Jul, silage
  Jul, potato Oct.
- vs Destatis: wheat June estimate national 4.2/6.0 (win), states 8.1/7.8 (close loss; v9 8.8);
  silage Aug/Sep national 1.9/3.1 (win), states 10.0/5.7; potato national 5.4/4.0, states 13.6/9.8.
- Verdict: v10 is a clear, broad improvement over v9 at district level (incl. new regions) and beats
  the simple rules everywhere; vs official forecasts it wins most early-season national cells; open
  gaps for v11: barley national, late season, July maize, state level vs reporters, ranges.
v10 FROZEN as is (no change from this look).
- Report rev 41: section 8j (v10 result + chart 45313173-4310); model card results filled.

## 2026-10-09 - v11 started (plan docs/v11_plan.md, draft pre-registration)
- Step 1: in-season forecasts of training years 2009-2017 with the v9 forward models (fitted <= 2008),
  weather scenarios 1979-2008 only (VISTA_FORWARD=2008; phase2_phenology relabels train rows > 2008 as
  'fwd'; phase2_forecast picks the forward MODIS model, technology rule last year = cut);
  run_forward_inseason.sh, all crops. Needed for fusion weights (A) and adaptive ranges (C).
- v11 A fusion (fusion_train.py, training years 2009-2017): MARS 2009-2016 bulletins fall in varying months
  -> per-month only wheat July had >= 6 years (fused 0.192 / 0.294 vs ours 0.217 / 0.391, MARS 0.269 / 0.336
  in halves 2009-12 / 2013-17). AMENDMENT (after seeing this one cell): pool bulletins into season phases
  (Mar-Jun, Jul, Aug-Oct; latest bulletin per year and phase); adoption rule = fused better than ours in
  both halves with cross-fitted weights (as written in the plan).
- Fusion data check: training-year MARS rows fail the identity check for wheat 2012-2014 and barley 2014:
  2012-2014 bulletins print three crop tables side by side (total / soft / durum wheat); the text parser
  mixes columns there. Training years only (test years unaffected). Fix = OCR of those pages, needs a
  re-download of ~30 bulletins (~30-60 MB) -> owner's OK pending. Pooled-phase result so far: wheat mid
  (Jul) fusion adopted (both halves better), barley late not (one half worse).
- Owner approved MARS 2012-2014 re-download + OCR (23 bulletins, 53 MB): phase0_mars_ocr.py old -> mars_germany_ocr_old.csv (layout last|this|avg, forecast 2nd value, check_old).
- docs/data_requests.md: drafts for FDZ/Statistik BB (BEE plot yields), ZALF patchCROP, YieldSAT authors, TUM agrihub (open, download needs OK), farm networks (cooperation offer); owner sends.
- TUM agrihub yield maps 1990-2000: licence 'Other (Not Open)', share returns 401 -> not downloadable; access request draft added (docs/data_requests.md #4). Nothing downloaded except the public documentation PDF (read, not kept).
- MARS 2012-2014 OCR done (30/39/54 tables, all but one pass the % check). Fusion (training years, phases,
  cross-fitted halves 2009-12 / 2013-17), adopted = fused better than ours in both halves:
  wheat early 0.437->0.343 / 0.697->0.608 (MARS 0.149 / 0.595), wheat Jul 0.217->0.192 / 0.508->0.498,
  wheat late 0.278->0.198 / 0.386->0.353, maize Jul 0.945->0.877 / 0.847->0.756, maize late 0.959->0.857 /
  0.513->0.446: ADOPTED; barley late 0.386->0.461 / 0.771->0.700: not adopted (A worse). Silage, potato pending.
- LST download complete (2025/2026 Aqua thin: 16 / 14 composites); LST combination tests (pass 2) running.
- PRE-REGISTERED fusion sensitivity (owner OK): shrinkage of the weight towards 0.5 reduced from 50 % to
  25 % (w = 0.75 w* + 0.125). Adopted for a crop-phase only if fused(25 %) < fused(50 %) in BOTH halves
  (cross-fitted) and fused(25 %) < ours in both halves; otherwise 50 % stays. Training years only.
- Fusion sensitivity RESULT: 25 % shrinkage worse than 50 % in at least one half for every crop-phase (wheat
  early 0.389/0.600 vs 0.343/0.608; Jul 0.201/0.499 vs 0.192/0.498; late 0.236/0.354 vs 0.198/0.353; maize Jul
  0.900/0.798 vs 0.877/0.756; late 0.879/0.459 vs 0.857/0.446; barley not adopted). Weights with less shrinkage
  swing to 0.8-0.9 and overfit the half they are learned on -> 50 % kept.
- Barley fusion with Destatis NOT run: Destatis has no pre-harvest barley estimate (only Jul/Aug and Aug/Sep results, published early Sep / mid Oct, after the mid-July barley harvest) -> would copy a measurement, not a forecast. Barley gap remains for S2 crop-specific inputs and more MARS years.
- PRE-REGISTERED v11 C adaptive ranges (ranges_adaptive.py; training forecasts 2009-2017 only):
  forecast distribution per district-year = median + normal error with variance = weather spread (scenario
  10-90 % -> sd) + year-wide sd + local sd. Static: year / local sd from forward errors of all years before t
  (v8 rolling cuts, from 1991). Adaptive: year sd(t) = max(static, RMS of national-mean errors of years t-5..t-1);
  local sd(t) = max(static, RMS of local errors of years t-6..t-2). National / state: area-weighted median,
  year sd shared, local part averaged (variance / effective n). Scores 2011-2017 (history >= 2 years):
  80 % coverage + interval score, district / state / national. Adopted if adaptive improves the interval
  score at all three levels or improves coverage towards 0.80 without worse interval score.

## 2026-10-09 - FIELD: within-field relative yield maps - PRE-REGISTRATION (YieldSAT pixel maps never used)
Licence note: YieldSAT is CC BY-NC-ND (research only) -> method development / testing only; a commercial
product must be trained on own cooperation data (patchCROP, farms).
- Fields: YieldSAT wheat, yield-map quality Good or Average, 2016-2022; pixels with a valid yield value,
  field edge removed (2-pixel erosion, headlands); field kept if >= 200 pixels.
- Target: relative yield = pixel yield / field median, clipped 0.3-1.7.
- Features per pixel (Sentinel-2 clips, clear = SCL 4/5): monthly medians (Apr, May, Jun, Jul) of NDVI and
  NDRE (B8A/B05), each divided by the field median of that month; season maximum NDVI (relative); sum of
  relative NDVI June-July (green leaf duration); terrain from the YieldSAT DEM layers relative to the field
  mean. Missing month -> 1.0 (= field average).
- Models: B0 flat map (1.0 everywhere); B1 relative June NDVI with one linear slope fitted on the
  training farms; M1 ridge on all features; M2 gradient boosting on all features (300 pixels per field
  sampled for training).
- Evaluation: leave-one-FARM-out (6 farms; main) and leave-one-year-out (secondary). Per field: pixel
  correlation r, RMSE of relative yield, zone agreement (field terciles: share of pixels whose predicted
  tercile equals the true one; chance = 0.33). Report median over fields, overall and per farm.
- Success: M1 or M2 beats B1 on median r and zone agreement in leave-one-farm-out.
- RESULT within-field relative maps (166 wheat fields, 182,831 pixels, 6 farms; relative_maps_scores.csv):
  leave-one-farm-out median per-field r / RMSE / zone agreement: B0 flat 0 / 0.164 / 0.333; B1 June NDVI
  0.381 / 0.152 / 0.465; M1 ridge 0.410 / 0.149 / 0.472; M2 boosting 0.406 / 0.149 / 0.470. LOYO: M1 0.444,
  M2 0.444 / zone 0.480. Per farm (M1): 0.40-0.57. Criterion met (M1 > B1 on r and zone), modest gain;
  combine yield-map noise at 10 m limits the attainable r. Field level paused until district v11 is done.
- 14:27 LST combination runs (v10.1 candidate) paused to free CPU for v11 (resumable via run_combined2.sh).
- Bug fixed: fusion_train.py overwrote fusion_train.csv when run for one crop (silage: no phase with >= 6 MARS training years -> empty); now merges; wheat/barley/maize rerun (deterministic).
- RESULT adaptive ranges (training 2011-2017; coverage static->adaptive district/state/national; iscore):
  barley 0.66->0.70 / 0.44->0.50 / 0.33->0.41, iscore better at all levels -> ADOPTED; grain maize 0.75->0.77 /
  0.64->0.68 / 0.55->0.55, iscore better at all levels -> ADOPTED; wheat 0.82->0.83 / 0.72->0.73 / 0.69->0.74,
  district iscore 3.004->3.012 worse -> static; silage district / national iscore worse -> static. Potato pending.
  Range goal not reachable by width alone (barley national 0.41).

## 2026-10-09 - v11 DISTRICT: FINAL RULES (before any v11 test-year result)
v11 = v10 (annual district offsets, maize national update, barley refit) + :
 A fusion with the latest JRC MARS bulletin (build_v11.py): adopted crop-phases from fusion_train.csv (wheat
   Mar-Jun / Jul / Aug-Oct; grain maize Jul / Aug-Oct; potato as its training result decides by the same rule;
   silage and barley: none), w_ours from training years, district forecasts scaled by fused / ours.
 C ranges (ranges_v11.py): normal-error method; adaptive for barley and grain maize (potato by the same rule),
   static otherwise.
 Sentinel-2 crop-specific inputs: NOT in v11 (later v11.1, data 2019-2023 still being processed).
Evaluation: the test years 2018-2025 were already looked at for v10 -> any v11 result on them is INDICATIVE
(reported as such); the clean test is the frozen 2026 prediction vs official 2026 district yields (spring 2027).
- v11 INDICATIVE RESULT (test years already seen): district RMSE v11 vs v10: wheat -0.7..+0.7 %, barley /
  silage unchanged, grain maize +0.8..+5.6 % (worse). National vs MARS (v10 / v11 / MARS %): wheat May 5.3/5.4/7.2,
  Jun 3.9/4.3/7.0, Jul 4.5/4.6/4.9, Aug 4.4/4.1/3.4; maize Aug 4.1/4.6/7.3, Sep 2.6/4.1/14.3, Oct 1.8/7.0/14.3.
  Fusion did NOT transfer: MARS late maize skill shifted between periods (good 2009-17, 14 % error 2018-21), so the
  fixed training weight imported its misses. DECISION (conservative, disclosed as based on already-seen years):
  fusion not adopted for release; v10 stays the released district model. Both v10 and v11 2026 predictions to be
  frozen for the clean spring-2027 comparison. Next levers: crop-specific Sentinel-2 (v11.1), state level, ranges.
- v10 FINAL RANGES (ranges_v11.py _v10 -> data/processed/ranges_v10_final; normal-error method chosen on
  training years; adaptive barley + grain maize, static wheat + silage; potato pending). INDICATIVE coverage
  on already-seen test years, lead 4, district / state / national: wheat 0.85/0.73/1.00; barley 0.83/0.71/0.75;
  grain maize 0.67/0.61/1.00 (4 yrs); silage 0.88/0.82/1.00 (vs first v10 ranges 0.58-0.81 / 0.40-0.73 /
  0.50-1.00). These ranges are used for the platform.
- v11 2026 predictions frozen (make_pred2026_v11.py; md5 check2026/frozen_2026_v11_md5.txt) for the spring-2027 v10 vs v11 comparison.
- Potato: adaptive ranges district 0.746->0.783 coverage, iscore better district/state but national 14.313->14.318
  (worse) -> static by rule; potato fusion (Jul, 6 yrs) not adopted.
- 14:44 Platform v1 (phase 1, district, v10 + final ranges) published as private artifact https://claude.ai/artifact/J74PjkEiwkrw6TQakiNkr1 (platform/index.html + platform/data via export_platform.py).
- 14:51 Code + platform pushed to https://github.com/Sanjusajimon220/german-crop-outlook (public, owner-created repo); Pages to be enabled by owner (main /docs).
- 15:10 Groundtruth site: complete deploy folder D:\groundtruth-earth-site (owner's pages + home page, fire-spread maps, favicon, og-image taken from the live site; new story page crop-outlook-germany.html from the site template; investigations card + Farming filter; platform in crop-outlook/). Link check: 17 pages, no broken local links. Deploy (drag and drop on Netlify) is the owner's action.
- 16:20 Platform: richer map (layers official yield, water use mm, water demand mm, MODIS greenness for all
  districts; detailed tooltip; 2026 water ratio exported; grey explained = no official district statistic).
  Site header no longer sticky (owner request). Copied to the site folder and repo docs, artifact v3.

## v11.1 PRE-REGISTRATION: crop-specific Sentinel-2 district features (written 2026-10-09 16:25, before any feature-vs-yield result)
- Data: S2 L2A NDVI, 25 interior cells per district x crop (CLMS crop types), Apr-Sep, cloud <= 60 %, 2019-2023
  (2022/2023 complete; 2019-2021 running). Crops: wheat -> winter_wheat, barley -> winter_barley,
  maize -> grain_maize and silage_maize, potatoes -> potato. Composites: dekad median of valid dates.
- Features at each forecast date (only dekads ending before the date): peak NDVI so far, mean of the last 3
  valid dekads, integral of max(NDVI - 0.2, 0) over dekads (missing dekads linearly interpolated, none after
  the last valid one). Each feature also as anomaly vs the state mean of the same crop and year
  (within-year spatial signal). District x crop rows need >= 5 cells and >= 4 valid dekads.
- Target: v10 district residual (official - v10 median) per crop and lead (ranges_v10_final).
- Model: ridge on standardised features (alpha = 10, fixed), correction applied with 50 % shrinkage.
- Evaluation: leave-one-year-out over the S2 years (2019-2023); the three held-out states (01, 08, 12)
  are also excluded from fitting in every fold and scored separately.
- Adoption per crop x lead: district RMSE lower on average by >= 3 % AND lower in >= 4 of 5 left-out years
  (>= 3 of 4 if a year is missing), held-out states not worse, and state / national area-weighted error
  not worse on average.
- CAVEAT (disclosed): 2019-2023 are v10 test years already looked at; any adoption is provisional. The
  clean test is 2026+ and needs crop maps for 2024+ (DLR 2024 / own classification), not yet available.
- 16:55 Overnight plan (owner: "use full night, finish all tasks"). (1) S2 2019-2021 downloads -> pre-registered
  v11.1 test (s2_eval.py, automatic). (2) LST combination test resumed with PRIORITY ORDER (no rule change):
  'physvar=lst' alone for all crops, cut 2008 and cut 2002, two parallel streams (run_lst_core.sh); same
  two-period adoption rule as the combined test (eval_combined.py / eval_select.py). Remaining LST
  combinations only if LST alone passes. (3) S2 -> MODIS bridge (satellite vs satellite, no yields).
- 17:10 S2 -> MODIS bridge first look (s2_bridge.py; satellite vs satellite, no yields; LOYO; 2022/2023 complete,
  2020/2021 only 2 states): r 0.66-0.82, RMSE 0.06-0.08 NDVI, within-date r 0.39-0.60. Usable but weaker than
  hoped (MODIS cropland also contains rapeseed / grassland not sampled). Rerun with all years after downloads.
- 17:25 LST ALONE result (two-period rule; A = fit <=2002 scored 2003-08, B = fit <=2008 scored 2009-17):
  wheat A +1.5 % / B -0.8 %; barley +0.9 / +2.2; grain maize +1.4 / +1.5; silage +2.0 / +1.8; potato -4.7 / -0.1
  but the stress gap worsens in B (1.75 -> 1.85). NO crop passes -> LST NOT adopted (v10.1 candidate closed).
  By the plan, the remaining LST combinations are not run. Canopy temperature stays out of the model.

## 2026-10-10 - v11.1 Sentinel-2 test result (run once, 02:20, exactly as pre-registered)
- Data complete: S2 2019-2023, all 16 states (2019 done 02:13, no retries); features per crop-year ~700-2100 rows.
- ADOPTED by the rule (provisional, years already seen in the v10 test): winter wheat lead 6 (-6.0 % district
  RMSE, 4/5 years, held-out 0.980 -> 0.921) and lead 12 (-7.6 %, 4/5, held-out 1.242 -> 1.137, state 0.78 -> 0.58);
  winter barley lead 8 (-5.9 %, 4/5); silage maize lead 4 (-3.1 %, 4/5) and lead 6 (-6.1 %, 5/5).
- NOT adopted: potato (all leads worse, -0.9 to -2.7 %); wheat leads 0/2 (no gain), 4 (+4.7 % but 3/5 years),
  8 (+2.4 % < 3 %); barley 0-6 (state level worse); silage 0/2/8/12.
- Grain maize NOT TESTABLE: official district yields in the v10 file only 2018-2021 -> 3 S2 years (< 4).
- Pattern: S2 crop greenness helps EARLY in the season (6-12 weeks before harvest), not close to harvest where
  the weather + MODIS model already holds the information. Potato samples are too small/noisy.
- National errors over 5 years are noisy (5 values); not used beyond the "not worse" check.
- Bridge (S2 -> MODIS, all years, LOYO): r 0.60-0.83, RMSE 0.068-0.075, within-date r 0.47-0.62. S2 can
  carry the MODIS signal only partly; a live MODIS replacement needs a better land-cover match (add rapeseed /
  grassland samples) - open.
- Next (needs owner decision): crop maps 2024-2026 (DLR 2024 / own classification, new S2 credits) to apply the
  adopted corrections live and in the clean 2026 check; implement v11.1 = v10 + S2 correction at the adopted
  crop x lead cells.
- 10:xx Owner approved: (1) DLR CropTypes 2024 + 2023 download (2 x 455 MB, CC BY 4.0; 2023 = overlap check
  CLMS vs DLR sampling before using DLR for 2024+); (2) v11.1 build.
- v11.1 BUILT AND FROZEN (build_v11_1.py; data/processed/v11_1/model.json md5 c99e65fcddf4bf41f65f04aec3c1adb6):
  v10 + 0.5 x ridge S2 correction at wheat 12/6, barley 8, silage 6/4 weeks; fitted 2019-2023 without held-out
  states; leave-one-year-out check reproduces the test exactly (wheat12 0.891->0.823, wheat6 0.765->0.719,
  barley8 1.036->0.974, silage6 5.08->4.78, silage4 4.84->4.69). Ranges = v10 ranges shifted by the correction.
  Live use / 2026 check needs S2 features for 2024+ on DLR crop maps (next).

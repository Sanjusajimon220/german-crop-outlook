import { test } from "node:test";
import assert from "node:assert/strict";
import { weekIndex, sourceLabel, withinNoise, nearestRgbDate, fmtUnc , leavesField, clearRgbDate, fieldUrl, noDataReason, zoomAllowed, areaToEnter, groupMedian, exportCheck, fieldRgbDate, areaRgbDate, legendSpec, seriesChart, SOURCE_COLOURS, cropStyle, vsMedianText, cropSummary, cropMedians, monthlyFrames, zoneSummary, zoneAreas, zoneTotal, PRODUCTS, CLASS_STYLES, classLegend, classByCrop, stageMarks, calendarChart, playIcon, nextFrame, TRAITS, traitsSection, STAGE_LABELS, calendarNote, retrievalNote } from "../../docs/js/field_core.js";

test("play / pause icon and the next frame of a time slider (stops at the end unless it loops)", () => {
  assert.match(playIcon(false), /<path d="M4 2.5v11l9-5.5z"/);
  assert.equal((playIcon(true).match(/h3v11h-3z/g) || []).length, 2);
  assert.equal(nextFrame(3, 5), 4);
  assert.equal(nextFrame(5, 5), null);
  assert.equal(nextFrame(5, 5, true), 0);
});

const calWeeks = ["2024-10-07", "2024-10-14", "2024-10-21", "2024-10-28", "2024-11-04"];

test("stage marks sit at the stage date as a fractional week, unreached stages left out", () => {
  const m = stageMarks({ emergence: "2024-10-17", heading: "2024-10-28", yellow_ripe: null }, ["emergence", "heading", "yellow_ripe"], calWeeks);
  assert.deepEqual(m.map(x => x.stage), ["emergence", "heading"]);
  assert.equal(m[0].i, 10 / 7);                       // 10 days after the first Monday
  assert.equal(m[1].i, 3);
  assert.equal(m[1].label, "heading");
});

test("calendar chart: curve, one dashed threshold per stage with its name, a dot where the curve reaches it", () => {
  const curve = { dev: [null, 2, 5, 8, 12], thr: [3, 10], dates: { emergence: "2024-10-17", heading: "2024-11-01" } };
  const svg = calendarChart({ curve, stages: ["emergence", "heading"], weeks: calWeeks, week: 4 });
  assert.equal((svg.match(/stroke-dasharray="4 3"/g) || []).length, 2);
  assert.ok(svg.includes(">emergence<") && svg.includes(">heading<"));
  assert.equal((svg.match(/<circle/g) || []).length, 2);
  assert.ok(!svg.includes("null") && !svg.includes("NaN"));
  assert.match(calendarChart({ curve: null, stages: [], weeks: calWeeks, week: 0 }), /no crop calendar/i);
});

test("weekly chart draws optional stage marks as dotted vertical lines with a tooltip", () => {
  const marks = stageMarks({ emergence: "2024-10-17", heading: "2024-10-28" }, ["emergence", "heading"], calWeeks);
  const svg = seriesChart({ values: [1, 1, 1, 1, 1], unc: [0, 0, 0, 0, 0], src: [1, 1, 1, 1, 1], season: [1, 1, 1, 1, 1], week: 0, weeks: calWeeks, max: 4, label: "LAI", marks });
  assert.equal((svg.match(/stroke-dasharray="1 2"/g) || []).length, 2);
  assert.ok(svg.includes("<title>expected heading: 2024-10-28</title>"));
});

const weeks = ["2024-05-20", "2024-05-27", "2024-06-03"];

test("weekIndex picks the Monday on or before the date", () => {
  assert.equal(weekIndex(weeks, "2024-05-27"), 1);
  assert.equal(weekIndex(weeks, "2024-05-30"), 1);
  assert.equal(weekIndex(weeks, "2024-06-10"), 2);
  assert.equal(weekIndex(weeks, "2024-05-01"), -1);
});

test("sourceLabel names every source code", () => {
  assert.equal(sourceLabel(0), "no data");
  assert.equal(sourceLabel(1), "measured");
  assert.equal(sourceLabel(2), "straight line");
  assert.equal(sourceLabel(3), "radar-filled");
  assert.equal(sourceLabel(4), "after last clear image");
});

test("withinNoise compares the absolute difference with the uncertainty", () => {
  assert.equal(withinNoise(0.2, 0.4), true);
  assert.equal(withinNoise(-0.5, 0.4), false);
  assert.equal(withinNoise(0.4, 0.4), false);
});

test("nearestRgbDate returns the closest clear image within 10 days", () => {
  const d = ["2024-05-01", "2024-06-25"];
  assert.equal(nearestRgbDate(d, "2024-05-27"), null);
  assert.equal(nearestRgbDate(d, "2024-05-06"), "2024-05-01");
  assert.equal(nearestRgbDate(d, "2024-06-17"), "2024-06-25");
  assert.equal(nearestRgbDate([], "2024-06-17"), null);
});

test("fmtUnc formats value and uncertainty", () => {
  assert.equal(fmtUnc(3.04, 0.41, 1), "3.0 ± 0.4");
  assert.equal(fmtUnc(0.761, 0.02, 2), "0.76 ± 0.02");
  assert.equal(fmtUnc(null, 0.4, 1), "no data");
});

test("only a user zoom-out leaves the field view, not the zoom-in animation", () => {
  assert.equal(leavesField({ active: true, k: 3, user: false, enterK: 40 }), false);   // animation from k = 1 towards the area
  assert.equal(leavesField({ active: true, k: 3, user: true, enterK: 40 }), true);     // the user zoomed out
  assert.equal(leavesField({ active: true, k: 30, user: true, enterK: 40 }), false);   // still close enough
  assert.equal(leavesField({ active: false, k: 3, user: true, enterK: 40 }), false);   // not in the field view
});

test("Sentinel-2 image comes from the nearest week the field was measured clear", () => {
  const weeks = ["2022-06-13", "2022-06-20", "2022-06-27", "2022-07-04", "2022-07-11"];
  const src = [1, 2, 3, 3, 1];                        // measured, line, radar, radar, measured
  const rgb = ["2022-06-14", "2022-07-03", "2022-07-12"];
  assert.equal(clearRgbDate(weeks, src, rgb, 3), "2022-07-12");   // 4 Jul itself was cloudy for this field
  assert.equal(clearRgbDate(weeks, src, rgb, 1), "2022-06-14");   // ties go to the earlier week
  assert.equal(clearRgbDate(weeks, [2, 2, 3, 3, 2], rgb, 3), null); // never measured clear: no image
});

test("field files carry the data version so a rebuild never mixes cached files", () => {
  assert.equal(fieldUrl("field/", "pilot_2024/px.bin", "20261020113500"), "field/pilot_2024/px.bin?v=20261020113500");
  assert.equal(fieldUrl("field/", "index.json", null), "field/index.json");
});

test("a week without a value says why", () => {
  assert.equal(noDataReason([0, 0, 1, 2, 0], 0), "before the first clear Sentinel-2 image");
  assert.equal(noDataReason([0, 0, 0], 1), "no clear Sentinel-2 image of this field this season");
  assert.equal(noDataReason([1, 2, 0], 2), "no value this week");
  assert.equal(noDataReason([1, 2, 3], 1), null);
});

test("map zoom never takes over page scrolling on the district map", () => {
  assert.equal(zoomAllowed({ type: "wheel", ctrlKey: false, k: 1, field: false }), false);   // plain scroll = page scroll
  assert.equal(zoomAllowed({ type: "wheel", ctrlKey: true, k: 1, field: false }), true);     // ctrl + scroll zooms
  assert.equal(zoomAllowed({ type: "wheel", ctrlKey: false, k: 80, field: true }), true);    // inside the field view
  assert.equal(zoomAllowed({ type: "touchstart", touches: 1, k: 1, field: false }), false);  // one-finger swipe scrolls
  assert.equal(zoomAllowed({ type: "touchstart", touches: 2, k: 1, field: false }), true);   // pinch zooms
  assert.equal(zoomAllowed({ type: "mousedown", button: 0, k: 1, field: false }), false);    // no drag at full view
  assert.equal(zoomAllowed({ type: "mousedown", button: 0, k: 3, field: false }), true);     // drag once zoomed in
});

test("zooming into a study area opens it", () => {
  const boxes = { pilot: [[10, 10], [20, 10], [20, 20], [10, 20]] };
  assert.equal(areaToEnter({ k: 12, enterK: 40, centre: [15, 15], boxes }), "pilot");
  assert.equal(areaToEnter({ k: 5, enterK: 40, centre: [15, 15], boxes }), null);    // not zoomed in enough
  assert.equal(areaToEnter({ k: 12, enterK: 40, centre: [30, 15], boxes }), null);   // looking elsewhere
});

test("median of the same crop for the list column", () => {
  const fields = { a: { group: "wheat", lai: [1, 2] }, b: { group: "wheat", lai: [3, null] }, c: { group: "maize", lai: [9, 9] } };
  assert.equal(groupMedian(fields, "wheat", 0), 2);
  assert.equal(groupMedian(fields, "wheat", 1), 2);
  assert.equal(groupMedian(fields, "barley", 0), null);
});

test("exports are blocked for fields only partly inside the study area", () => {
  assert.deepEqual(exportCheck({ rates: [60, 50, 40], partial: false }), { ok: true, msg: "" });
  assert.equal(exportCheck({ rates: [60, 50, 40], partial: true }).ok, false);
  assert.equal(exportCheck({ rates: [60, "", 40], partial: false }).ok, false);
});

test("field panel image: nearest date this field itself was clear", () => {
  const rgb = ["2022-06-11", "2022-06-21", "2022-07-01", "2022-07-06"];
  assert.equal(fieldRgbDate(rgb, [0, 1], "2022-06-29"), "2022-06-21");   // 1 Jul is nearer but not clear over this field
  assert.equal(fieldRgbDate(rgb, [], "2022-07-04"), null);
});

test("map true-colour layer: nearest date that was clear over most of the area", () => {
  const rgb = ["2022-06-11", "2022-06-21", "2022-07-01"], share = [0.9, 0.2, 0.7];
  assert.equal(areaRgbDate(rgb, share, "2022-06-20"), "2022-06-11");    // 21 June mostly cloudy
  assert.equal(areaRgbDate(rgb, share, "2022-06-29"), "2022-07-01");
  assert.equal(areaRgbDate(rgb, [0.1, 0.1, 0.1], "2022-06-20"), null);  // no clear image near this week
});

test("vs median only for groups with enough fields", () => {
  const f = {}; for (let i = 0; i < 12; i++) f[i] = { group: "wheat", lai: [i] };
  f.x = { group: "legume", lai: [3] }; f.y = { group: "legume", lai: [1] };
  assert.equal(groupMedian(f, "wheat", 0, 10), 5.5);
  assert.equal(groupMedian(f, "legume", 0, 10), null);
});

test("legend follows the field layer", () => {
  assert.deepEqual(legendSpec("lai"), { kind: "ramp", scheme: "YlGn", min: 0, max: 5, unit: "green LAI (m²/m²)" });
  assert.deepEqual(legendSpec("ndvi"), { kind: "ramp", scheme: "YlGn", min: 0, max: 1, unit: "NDVI" });
  assert.equal(legendSpec("source").kind, "swatches");
  assert.equal(legendSpec("rgb").kind, "text");
});

test("field image only within 10 days of the week", () => {
  const rgb = ["2022-05-01", "2022-06-21"];
  assert.equal(fieldRgbDate(rgb, [0, 1], "2022-06-27"), "2022-06-21");
  assert.equal(fieldRgbDate(rgb, [0], "2022-06-27"), null);                 // 1 May is 57 days away: no image
});

test("weekly chart: dots coloured by source, gaps without broken paths, message when empty", () => {
  const v = [null, 1.0, 1.5, null, 2.5], u = [null, 0.2, 0.4, null, 0.6], src = [0, 1, 2, 0, 3], season = [0, 1, 1, 1, 1];
  const svg = seriesChart({ values: v, unc: u, src, season, week: 2, weeks: ["2022-04-04", "2022-04-11", "2022-04-18", "2022-04-25", "2022-05-02"], max: 4, label: "LAI" });
  assert.equal((svg.match(/<circle/g) || []).length, 3);
  assert.ok(svg.includes(SOURCE_COLOURS[1]) && svg.includes(SOURCE_COLOURS[2]) && svg.includes(SOURCE_COLOURS[3]));
  assert.ok(!svg.includes("null") && !svg.includes("NaN"));
  assert.match(seriesChart({ values: [null, null], unc: [null, null], src: [0, 0], season: [0, 0], week: 0, weeks: ["2022-04-04", "2022-04-11"], max: 1, label: "NDVI" }), /no value/i);
});

test("crop names in English with one colour per crop", () => {
  assert.deepEqual(cropStyle("wheat", "Winterweichweizen"), { label: "Winter wheat", colour: "#d9a441" });
  assert.equal(cropStyle("barley", "Wintergerste").label, "Winter barley");
  assert.equal(cropStyle("barley", "Sommergerste").label, "Spring barley");
  assert.equal(cropStyle("maize", "Silomais").label, "Silage maize");
  assert.equal(cropStyle("maize", "Mais (ohne Silomais)").label, "Grain maize");
  assert.equal(cropStyle("sugar beet", "Zuckerrüben").label, "Sugar beet");
  assert.notEqual(cropStyle("wheat", "x").colour, cropStyle("barley", "x").colour);
  assert.equal(cropStyle("something new", "Xyz").label, "Xyz");                 // unknown: keep the register name
});

test("vs median in percent only when the median is large enough", () => {
  assert.equal(vsMedianText(3.8, 3.45, 0.6), "+10 % (within noise)");
  assert.equal(vsMedianText(0.1, 0.33, 0.2), "−0.2 LAI (beyond noise)".replace("beyond", "beyond"));
  assert.equal(vsMedianText(null, 2, 0.5), "–");
});

test("crop summary of a study area: one entry per crop label, largest area first", () => {
  const props = [
    { id: 1, group: "wheat", crop: "Winterweichweizen", ha: 5 }, { id: 2, group: "wheat", crop: "Winterweichweizen", ha: 3 },
    { id: 3, group: "maize", crop: "Silomais", ha: 4 }, { id: 4, group: "maize", crop: "Mais (ohne Silomais)", ha: 1 },
  ];
  const s = cropSummary(props);
  assert.deepEqual(s.map(x => [x.label, x.n, x.ha]), [["Winter wheat", 2, 8], ["Silage maize", 1, 4], ["Grain maize", 1, 1]]);
  assert.ok(s[0].colour.startsWith("#"));
});

test("median curve per crop for the area chart (crops with too few fields left out)", () => {
  const props = [1, 2, 3].map(id => ({ id, group: "wheat", crop: "Winterweichweizen" })).concat([{ id: 9, group: "potato", crop: "Kartoffeln" }]);
  const fields = { 1: { lai: [1, 2] }, 2: { lai: [3, null] }, 3: { lai: [2, 4] }, 9: { lai: [5, 5] } };
  const m = cropMedians(props, fields, 3);
  assert.deepEqual(m["Winter wheat"], [2, 3]);
  assert.equal(m["Potato"], undefined);
});

test("common special crops get English names too", () => {
  assert.equal(cropStyle("other", "Gartenkürbis").label, "Pumpkin");
  assert.equal(cropStyle("other", "Möhre (auch Futtermöhre)").label, "Carrot");
  assert.equal(cropStyle("other cereal", "Winterhafer").label, "Winter oats");
});

test("animation frames: the first week of every month", () => {
  const weeks = ["2024-09-30", "2024-10-07", "2024-10-28", "2024-11-04", "2024-11-25", "2024-12-02"];
  assert.deepEqual(monthlyFrames(weeks), [0, 1, 3, 5]);     // Sep, Oct, Nov, Dec
});

test("median curves per crop also for NDVI", () => {
  const props = [1, 2, 3].map(id => ({ id, group: "wheat", crop: "Winterweichweizen" }));
  const fields = { 1: { lai: [1], ndvi: [0.5] }, 2: { lai: [2], ndvi: [0.7] }, 3: { lai: [3], ndvi: [0.6] } };
  assert.deepEqual(cropMedians(props, fields, 3, "ndvi")["Winter wheat"], [0.6]);
});

test("field margin strips get an English name", () => {
  assert.equal(cropStyle("other", "Randstreifen (Acker/DK)").label, "Field margin strip");
});

test("cropStyle: register names of small crops and landscape elements are shown in English", () => {
  const cases = [["Blühstreifen (nur AUM)", "Flower strip"], ["Blühfläche (AUM-Altmaßnahme)", "Flower area"],
    ["ÖR 1b Blühfläche auf AL", "Flower area"], ["ÖVF-Streifen AL", "Ecological focus strip"],
    ["Streuobst mit DGL-Nutzung", "Meadow orchard"], ["Salatgurke (auch Einlegegurke)", "Cucumber"],
    ["Salat (Garten, Lollo Rosso.)", "Lettuce"], ["Futterrübe/Runkelrübe", "Fodder beet"],
    ["Gemenge Getr./Leg.(mehr Getr.)", "Mixed crop"], ["Riesenkürbis (auch Hokkaido)", "Pumpkin"],
    ["Unbefestigte Mieten AL", "Field clamp"], ["Kernobst z.B. Äpfel, Birnen", "Pome fruit"]];
  for (const [crop, label] of cases) assert.equal(cropStyle("other", crop).label, label, crop);
  assert.equal(cropStyle("other cereal", "Wintermenggetreide").label, "Mixed cereals");
});

test("zoneSummary: reliability, size of the difference and likely cause in plain words", () => {
  const s = zoneSummary({ held: 4, tests: 5, spread: 0.08, seasons: 5, slope: 0.8, twi: -0.6, nfk: -12 });
  assert.equal(s.reliability, "reliable");
  assert.match(s.lines[0], /4 of 5 seasons/);
  assert.match(s.lines[1], /8 %/);
  assert.ok(s.lines.some(l => /drier/.test(l)) && s.lines.some(l => /steeper/.test(l)) && s.lines.some(l => /less plant-available water/.test(l)));
  const flat = zoneSummary({ held: 2, tests: 4, spread: 0.03, seasons: 4 });
  assert.equal(flat.uniform, true);
  assert.equal(flat.reliability, "weak");
  assert.match(flat.lines[1], /fairly uniform/);
  assert.match(flat.lines[2], /No clear soil or terrain pattern/);
  assert.equal(zoneSummary(null), null);
});

test("zoneAreas: hectares per zone from 10 m pixels (0.01 ha each); one zone puts everything in the middle", () => {
  assert.deepEqual(zoneAreas([0, 0, 1, 2, 2, 2, -1], 3), [0.02, 0.01, 0.03]);
  assert.deepEqual(zoneAreas([0, 2, 1], 1), [0, 0.03, 0]);
});

test("zoneTotal: total amount for the field, only when every zone has a rate", () => {
  assert.equal(zoneTotal([2.9, 4.6, 2.9], [150, 160, 170]), 2.9 * 150 + 4.6 * 160 + 2.9 * 170);
  assert.equal(zoneTotal([2.9, 4.6, 2.9], [150, "", 170]), null);
  assert.equal(PRODUCTS.nitrogen.unit, "kg N/ha");
});

test("zoneAreas: edge pixels count with their share of the field", () => {
  assert.deepEqual(zoneAreas([0, 1, 2, 2], 3, 0.01, [1, 0.5, 1, 0.2]), [0.01, 0.005, 0.012]);
  assert.deepEqual(zoneAreas([2, 2, 2, 2], 3, 0.01, [1, 1, 1, 0.5]), [0, 0, 0.035]);
});

test("classLegend: field count and hectares per class, in the fixed order, not-assessed last", () => {
  const props = [{ id: 1, ha: 2 }, { id: 2, ha: 3 }, { id: 3, ha: 1.5 }, { id: 4, ha: 4 }];
  const cls = { fields: { 1: { prod_class: "high" }, 2: { prod_class: "low" }, 3: { prod_class: "high" }, 4: { prod_class: "not assessed" } } };
  const L = classLegend(cls, props, "prod");
  assert.deepEqual(L.map(r => [r.label, r.n, r.ha]), [["low", 1, 3], ["average", 0, 0], ["high", 2, 3.5], ["not assessed", 1, 4]]);
  assert.equal(L[2].colour, CLASS_STYLES.prod.high);
  assert.equal(classLegend(null, props, "prod"), null);
});

test("classByCrop: field counts per crop and class, largest crop first; classLegend can filter by crop", () => {
  const props = [{ id: 1, ha: 2, group: "wheat", crop: "Winterweichweizen" }, { id: 2, ha: 3, group: "wheat", crop: "Winterweichweizen" },
    { id: 3, ha: 1, group: "sugar beet", crop: "Zuckerrüben" }];
  const cls = { fields: { 1: { var_class: "uniform" }, 2: { var_class: "variable, reliable zones" }, 3: { var_class: "variable, reliable zones" } } };
  const t = classByCrop(cls, props, "var");
  assert.equal(t[0].crop, "Winter wheat");
  assert.equal(t[0].counts.uniform, 1); assert.equal(t[0].counts["variable, reliable zones"], 1);
  assert.equal(t[1].crop, "Sugar beet"); assert.equal(t[1].counts["variable, reliable zones"], 1);
  const L = classLegend(cls, props, "var", "Sugar beet");
  assert.deepEqual(L.map(r => r.n), [0, 1, 0, 0]);
});

test("trait section: three weekly charts with values, typical errors and the note on reliability", () => {
  const weeks = ["2022-03-07", "2022-03-14", "2022-03-21"];
  const tr = { ccc: [100, 150, null], ccc_unc: [20, 26, null], cab: [40, 45, null], cab_unc: [5, 6, null],
               cbrown: [0.1, 0.2, null], cbrown_unc: [0.1, 0.13, null], src: [1, 2, 0] };
  const html = traitsSection({ tr, season: [1, 1, 1], week: 1, weeks });
  assert.deepEqual(TRAITS.map(t => t.key), ["ccc", "cab", "cbrown"]);
  assert.equal((html.match(/<svg class="chart"/g) || []).length, 3);
  assert.match(html, /150 ± 26/);                                    // CCC this week, rounded as the trait says
  assert.match(html, /45\.0 ± 6\.0/);
  assert.match(html, /0\.20 ± 0\.13/);
  assert.match(html, /not measured plant nitrogen/i);
  assert.equal(traitsSection({ tr: null, season: [], week: 0, weeks }), "");     // no traits file: nothing shown
});

test("stage labels for maize and potato calendars", () => {
  for (const [k, v] of [["tassel", "tasselling"], ["flowering", "flowering"], ["dough_ripe", "dough ripe"], ["canopy_closed", "canopy closed"]])
    assert.equal(STAGE_LABELS[k], v);
});

test("calendar stage labels of close thresholds do not overlap", () => {
  const weeks = ["2022-04-04", "2022-04-11", "2022-04-18", "2022-04-25"];
  const curve = { dev: [0, 10, 50, 90], thr: [10, 70, 74, 80, 83], dates: {} };
  const svg = calendarChart({ curve, stages: ["emergence", "tassel", "flowering", "milk_ripe", "dough_ripe"], weeks, week: 1 });
  const ys = [...svg.matchAll(/<text x="[\d.]+" y="([\d.]+)" text-anchor="end">(?:emergence|tasselling|flowering|milk ripe|dough ripe)</g)].map(m => +m[1]).sort((a, b) => a - b);
  assert.equal(ys.length, 5);
  for (let k = 1; k < ys.length; k++) assert.ok(ys[k] - ys[k - 1] >= 10, `labels ${ys[k - 1]} and ${ys[k]} overlap`);
  assert.match(calendarChart({ curve: null, stages: [], weeks, week: 0 }), /winter barley and maize only/);
});

test("season timing label: crop named, regional and checked; other crops say why there is none", () => {
  const wheat = calendarNote({ crop: "wheat", sowing: "2021-10-12" }, [{ label: "heading", date: "2022-05-19" }]);
  assert.match(wheat, /winter wheat/); assert.match(wheat, /2021-10-12/); assert.match(wheat, /heading 05-19/);
  assert.match(wheat, /not measured on this field/i); assert.match(wheat, /\+4 days/);
  assert.match(calendarNote({ crop: "barley", sowing: "2021-09-25" }, []), /winter barley.*\+14 days/s);
  assert.match(calendarNote(null, [], "maize"), /only for winter wheat and winter barley/);
  assert.equal(calendarNote(null, [], "grass"), "");                          // non-arable: nothing
});

test("retrieval note only for crops the cereal-trained network was not made for", () => {
  for (const g of ["maize", "sugar beet", "potato", "rapeseed"]) assert.match(retrievalNote(g), /trained on cereal canopies/);
  for (const g of ["wheat", "barley", "other cereal", "grass / fodder", "other"]) assert.equal(retrievalNote(g), "");
});

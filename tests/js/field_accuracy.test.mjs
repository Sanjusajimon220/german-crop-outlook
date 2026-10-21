import { test } from "node:test";
import assert from "node:assert/strict";
import { item2Scatter, item1Table, item3Boxes, item4Chart, fieldAccuracyHtml } from "../../docs/js/field_accuracy.js";

const count = (s, re) => (s.match(re) || []).length;

test("item2Scatter: one dot per Landsat date, diagonal, and the win count in the label", () => {
  const svg = item2Scatter({ won: 2, of: 3, points: [{ season: 2021, n: 40, ours: 0.9, ppi: 0.8 }, { season: 2022, n: 300, ours: 0.7, ppi: 0.6 },
    { season: 2023, n: 50, ours: 0.5, ppi: 0.6 }] });
  assert.equal(count(svg, /<circle /g), 3);
  assert.match(svg, /<line[^>]*class="diag"/);
  assert.match(svg, /2 of 3/);
});

test("item1Table: one row per crop, one cell per season, values to two decimals", () => {
  const html = item1Table([{ crop: "wheat", season: 2021, rho: 0.931 }, { crop: "wheat", season: 2022, rho: 0.948 },
    { crop: "maize", season: 2021, rho: 0.879 }, { crop: "maize", season: 2022, rho: null }]);
  assert.equal(count(html, /<tr>/g), 3);                         // header + 2 crops
  assert.match(html, />0\.93</); assert.match(html, />0\.88</); assert.match(html, />–</);
});

test("item3Boxes: a box per crop and event, zero line drawn", () => {
  const svg = item3Boxes([{ crop: "wheat", event: "start", q1: 20, median: 30, q3: 40 }, { crop: "wheat", event: "peak", q1: -5, median: 3, q3: 10 },
    { crop: "wheat", event: "end", q1: -25, median: -20, q3: -15 }]);
  assert.equal(count(svg, /<rect class="box"/g), 3);
  assert.match(svg, /class="zero"/);
});

test("item4Chart: CGLS line, ours after fix 2 line, title with area", () => {
  const svg = item4Chart("pilot", [["2021-05-10", 2.0, 1.8, 1.9], ["2021-05-20", 2.2, 2.0, 2.1], ["2021-05-31", null, 1.0, 1.1]]);
  assert.equal(count(svg, /<path /g), 3);                        // CGLS, before, after
  assert.match(svg, /pilot/i);
});

test("fieldAccuracyHtml: all sections present with the key results in plain words", () => {
  const data = { area_test: { lai: { won: 17, of: 20, areas: [] }, ndvi: { won: 16, of: 20, areas: [] } },
    uncertainty: { lai: { line_le15: 0.43, line_16_30: 0.54, radar_gt30: 0.62, live_le30: 0.7, live_gt30: 0.86 } },
    fix1: { adopted: false, verdict: { lai: { winter_better_areas: 3, pass: true }, ndvi: { winter_better_areas: 2, pass: false } } },
    fix2: { lai: { flagged_share: 0.0166, ratio: 2.8, adopted: true, flagged_vs_landsat: 0.538, unflagged_vs_landsat: 0.192 },
      ndvi: { flagged_share: 0.0517, ratio: 2.78, adopted: false, flagged_vs_landsat: 0.118, unflagged_vs_landsat: 0.043 },
      rebuild: { changed_pct: 4.28, mean_where_changed: 0.48 } },
    item1: [], item2: { won: 169, of: 195, median_ours: 0.88, median_ppi: 0.808, points: [] }, item3: [],
    item4: { stats: { before: { "all | pooled": { bias: -0.133, rmse: 0.616, r: 0.791 } }, after: { "all | pooled": { bias: -0.113, rmse: 0.609, r: 0.792 } } }, series: {} } };
  const html = fieldAccuracyHtml(data);
  for (const s of ["17 of 20", "not adopted", "adopted for LAI", "169 of 195", "0.79", "Limits"]) assert.ok(html.includes(s), s);
});

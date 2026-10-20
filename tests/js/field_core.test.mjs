import { test } from "node:test";
import assert from "node:assert/strict";
import { weekIndex, sourceLabel, withinNoise, nearestRgbDate, fmtUnc , leavesField, clearRgbDate } from "../../docs/js/field_core.js";

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

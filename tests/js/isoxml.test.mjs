import { test } from "node:test";
import assert from "node:assert/strict";
import { buildTaskData } from "../../docs/js/isoxml.js";

const grid = { minLat: 50.9, minLon: 6.3, dLat: 0.0001, dLon: 0.00015, rows: 3, cols: 4 };
const classes = Int8Array.from([0, 1, 2, -1, 0, 1, 2, 2, 1, 1, 0, -1]);
const args = { field: { id: 7493815, crop: "Winterweichweizen" }, zones: { classes, n: 3 }, rates: [100, 80, 60], grid };

test("buildTaskData writes a task with one product and a type-2 grid", () => {
  const { xml } = buildTaskData(args);
  assert.match(xml, /<ISO11783_TaskData /);
  assert.match(xml, /<TSK /);
  assert.match(xml, /<PDT A="PDT1"/);
  assert.match(xml, /<GRD [^>]*I="2"/);
  assert.match(xml, /<GRD [^>]*E="4"[^>]*F="3"/);                  // columns, rows
  assert.match(xml, /<TZN A="1"[^>]*>\s*<PDV A="0006"/);            // grid zone with mass-per-area rate
});

test("type-2 grid cells hold the rate in mg/m2 (kg/ha x 100) as little-endian int32", () => {
  const { bin } = buildTaskData(args);
  assert.equal(bin.length, grid.rows * grid.cols * 4);
  const v = new DataView(bin.buffer);
  assert.deepEqual([0, 1, 2, 3].map(i => v.getInt32(4 * i, true)), [10000, 8000, 6000, 0]);
});

test("cells outside the field get rate 0", () => {
  const { bin } = buildTaskData(args);
  assert.equal(new DataView(bin.buffer).getInt32(4 * 11, true), 0);
});

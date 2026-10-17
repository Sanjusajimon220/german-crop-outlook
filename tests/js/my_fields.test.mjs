import { test } from "node:test";
import assert from "node:assert/strict";
import { loadMyFields, saveMyFields, zones } from "../../docs/js/field_core.mjs";

const mem = init => { const m = new Map(init); return { getItem: k => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, v) }; };
const KEY = "cropOutlook.myFields.v1";

test("loadMyFields handles missing and corrupt storage", () => {
  assert.deepEqual(loadMyFields(mem([])), { fields: [], rates: {} });
  assert.deepEqual(loadMyFields(mem([[KEY, "{not json"]])), { fields: [], rates: {} });
  assert.deepEqual(loadMyFields(mem([[KEY, '{"fields":"x"}']])), { fields: [], rates: {} });
  assert.deepEqual(loadMyFields(null), { fields: [], rates: {} });
});

test("saveMyFields round-trips through storage", () => {
  const s = mem([]);
  const st = { fields: [{ area: "pilot", year: 2024, id: 7493815 }], rates: { 7493815: [100, 80, 60] } };
  saveMyFields(s, st);
  assert.deepEqual(loadMyFields(s), st);
});

test("zones terciles with equal weights", () => {
  const z = zones([1, 2, 3, 4, 5, 6, 7, 8, 9], Array(9).fill(1));
  assert.equal(z.n, 3);
  assert.ok(Math.abs(z.cuts[0] - 3.67) < 0.01 && Math.abs(z.cuts[1] - 6.33) < 0.01);
  assert.deepEqual(Array.from(z.classes), [0, 0, 0, 1, 1, 1, 2, 2, 2]);
});

test("zones with uniform values", () => {
  const z = zones(Array(20).fill(2.5), Array(20).fill(1));
  assert.equal(z.n, 1);
  assert.deepEqual(Array.from(z.classes), Array(20).fill(0));
});

test("zones with fewer than 9 pixels gives one class", () => {
  const z = zones([1, 5, 9], [1, 1, 1]);
  assert.equal(z.n, 1);
});

test("zones ignore missing values", () => {
  const z = zones([1, 2, 3, 4, NaN, 5, 6, 7, 8, 9], Array(10).fill(1));
  assert.equal(z.classes[4], -1);
  assert.equal(z.n, 3);
});

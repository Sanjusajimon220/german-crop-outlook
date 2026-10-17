import { test } from "node:test";
import assert from "node:assert/strict";
import { affineFromCorners, applyAffine, invertAffine } from "../../docs/js/field_core.js";

// image 800 x 800 px; corners (UL, UR, LL) placed in screen coordinates
const UL = [100, 50], UR = [900, 80], LL = [70, 850];

test("affineFromCorners maps the image corners onto the given points", () => {
  const m = affineFromCorners(UL, UR, LL, 800, 800);
  assert.deepEqual(applyAffine(m, 0, 0).map(Math.round), UL);
  assert.deepEqual(applyAffine(m, 800, 0).map(Math.round), UR);
  assert.deepEqual(applyAffine(m, 0, 800).map(Math.round), LL);
  assert.deepEqual(applyAffine(m, 800, 800).map(Math.round), [870, 880]);
});

test("invertAffine takes points back to image pixels", () => {
  const m = affineFromCorners(UL, UR, LL, 800, 800), inv = invertAffine(m);
  const [x, y] = applyAffine(m, 123.5, 456.25);
  const [c, r] = applyAffine(inv, x, y);
  assert.ok(Math.abs(c - 123.5) < 1e-9 && Math.abs(r - 456.25) < 1e-9);
});

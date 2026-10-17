// Pure helpers for the field level (no DOM); tested with `node --test tests/js/*.test.mjs`.

const DAY = 864e5;
const t = s => Date.parse(s + "T00:00:00Z");

// index of the last week (Monday, "YYYY-MM-DD") on or before date; -1 before the first week
export function weekIndex(weeks, date) {
  const d = t(date);
  let k = -1;
  for (let i = 0; i < weeks.length; i++) if (t(weeks[i]) <= d) k = i;
  return k;
}

const SOURCES = ["no data", "measured", "straight line", "radar-filled", "after last clear image"];
export function sourceLabel(src) {
  return SOURCES[src] ?? "no data";
}

// a difference smaller than the uncertainty is "within noise"
export function withinNoise(diff, unc) {
  return Math.abs(diff) < unc;
}

// closest clear Sentinel-2 image to the week, at most 10 days away
export function nearestRgbDate(dates, week) {
  let best = null, bd = Infinity;
  for (const d of dates) {
    const g = Math.abs(t(d) - t(week)) / DAY;
    if (g < bd) { bd = g; best = d; }
  }
  return bd <= 10 ? best : null;
}

export function fmtUnc(v, u, d) {
  if (v == null || !isFinite(v)) return "no data";
  return `${(+v).toFixed(d)} ± ${(+u).toFixed(d)}`;
}

// ---- MY FIELDS: picked fields and zone rates, kept in the browser (no accounts) ----
export const MY_FIELDS_KEY = "cropOutlook.myFields.v1";
const EMPTY = () => ({ fields: [], rates: {} });

export function loadMyFields(storage) {
  try {
    const s = storage && storage.getItem(MY_FIELDS_KEY);
    if (!s) return EMPTY();
    const v = JSON.parse(s);
    if (!v || !Array.isArray(v.fields) || typeof v.rates !== "object" || v.rates === null) return EMPTY();
    return { fields: v.fields, rates: v.rates };
  } catch (e) {
    return EMPTY();
  }
}

export function saveMyFields(storage, state) {
  storage.setItem(MY_FIELDS_KEY, JSON.stringify({ fields: state.fields, rates: state.rates }));
}

// ---- zones: weighted terciles of a period-mean LAI (one value per pixel; NaN = outside / no data) ----
function wquantile(sorted, q) {
  // sorted: [[value, weight], ...]; equals numpy's linear quantile for equal weights
  const C = []; let c = 0;
  for (const [, w] of sorted) { c += w; C.push(c); }
  const den = C[C.length - 1] - sorted[sorted.length - 1][1];
  const pos = sorted.map(([, w], i) => (C[i] - w) / den);
  for (let i = 1; i < sorted.length; i++) {
    if (pos[i] >= q) {
      const f = (q - pos[i - 1]) / (pos[i] - pos[i - 1] || 1);
      return sorted[i - 1][0] + f * (sorted[i][0] - sorted[i - 1][0]);
    }
  }
  return sorted[sorted.length - 1][0];
}

export function zones(values, weights) {
  const classes = new Int8Array(values.length).fill(-1);
  const ok = [];
  values.forEach((v, i) => { if (v != null && isFinite(v)) ok.push([v, weights[i] > 0 ? weights[i] : 0, i]); });
  const vs = ok.map(o => o[0]);
  if (ok.length < 9 || Math.max(...vs) - Math.min(...vs) < 1e-9) {
    ok.forEach(o => { classes[o[2]] = 0; });
    return { classes, cuts: [NaN, NaN], n: 1 };
  }
  const sorted = ok.map(o => [o[0], o[1]]).sort((a, b) => a[0] - b[0]);
  const cuts = [wquantile(sorted, 1 / 3), wquantile(sorted, 2 / 3)];
  ok.forEach(([v, , i]) => { classes[i] = v <= cuts[0] ? 0 : v <= cuts[1] ? 1 : 2; });
  const n = new Set(ok.map(o => classes[o[2]])).size;
  return { classes, cuts, n };
}

// ---- geometry: image placement by three corners (the UTM grid is rotated against the web map north) ----
// matrix [a, b, c, d, e, f]: x = a*col + c*row + e, y = b*col + d*row + f (SVG matrix order)
export function affineFromCorners(ul, ur, ll, w, h) {
  return [(ur[0] - ul[0]) / w, (ur[1] - ul[1]) / w, (ll[0] - ul[0]) / h, (ll[1] - ul[1]) / h, ul[0], ul[1]];
}

export function applyAffine(m, col, row) {
  return [m[0] * col + m[2] * row + m[4], m[1] * col + m[3] * row + m[5]];
}

export function invertAffine(m) {
  const [a, b, c, d, e, f] = m, det = a * d - b * c;
  return [d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det];
}

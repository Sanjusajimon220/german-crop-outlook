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

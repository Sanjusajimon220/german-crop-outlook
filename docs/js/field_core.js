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

// the field view closes only when the USER zooms out past half the entry zoom (programmatic zoom-in passes low k too)
export function leavesField({ active, k, user, enterK }) {
  return active && user && k < enterK / 2;
}

// every field file URL carries the data version from index.json: field files are cached for a week, so a rebuild
// must never mix a new px.json with old px.bin bytes
export function fieldUrl(base, path, v) {
  return v ? `${base}${path}?v=${v}` : `${base}${path}`;
}

// why a week has no value (src 0), or null when it has one
export function noDataReason(src, w) {
  if (src[w] !== 0) return null;
  const first = src.indexOf(1);
  if (first < 0) return "no clear Sentinel-2 image of this field this season";
  if (w < first) return "before the first clear Sentinel-2 image";
  return "no value this week";
}

// d3-zoom filter: on the district map plain scrolling and one-finger swipes stay page scrolling (district map unchanged);
// ctrl + scroll or a pinch zooms; dragging pans only once zoomed in; inside the field view the map takes the gestures
export function zoomAllowed({ type, ctrlKey, touches, button, k, field }) {
  if (type === "wheel") return field || !!ctrlKey;
  if (type.startsWith("touch")) return field || (touches ?? 0) >= 2 || k > 1.01;
  if (type === "mousedown") return (button ?? 0) === 0 && (field || k > 1.01);
  return false;
}

// zooming in (k at least a quarter of the field zoom) with the view centre inside a study area opens that area
export function areaToEnter({ k, enterK, centre, boxes }) {
  if (k < enterK / 4) return null;
  const [x, y] = centre;
  for (const [a, poly] of Object.entries(boxes)) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside) return a;
  }
  return null;
}

// median green LAI of the fields of one crop group in week w (null if fewer than minN fields have a value)
export function groupMedian(fields, group, w, minN = 1) {
  const v = Object.values(fields).filter(f => f.group === group && f.lai[w] != null).map(f => f.lai[w]).sort((a, b) => a - b);
  if (!v.length || v.length < minN) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// zone exports need a rate per zone and a field fully inside the study area (the grid would cover only the inside part)
export function exportCheck({ rates, partial }) {
  if (partial) return { ok: false, msg: "This field is only partly inside the study area: exports are disabled (they would cover the inside part only)." };
  const ok = Array.isArray(rates) && rates.length === 3 && rates.every(v => v !== "" && v != null && isFinite(v));
  return ok ? { ok: true, msg: "" } : { ok: false, msg: "Set a rate for every zone to enable the exports." };
}

// field vs the median of its crop: percent when the median is at least 0.5 LAI, otherwise the difference in LAI units
export function vsMedianText(v, med, unc) {
  if (v == null || med == null) return "–";
  const d = v - med, sign = d > 0 ? "+" : d < 0 ? "−" : "", noise = Math.abs(d) < (unc ?? 0) ? "within noise" : "beyond noise";
  const amount = med >= 0.5 ? `${Math.round(Math.abs(100 * d / med))} %` : `${Math.abs(d).toFixed(1)} LAI`;
  return `${sign}${amount} (${noise})`;
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

// image for the field panel: nearest week this FIELD was measured clear (src 1), then the image within 3 days of it
export function clearRgbDate(weeks, src, dates, w) {
  const order = weeks.map((_, k) => k).sort((a, b) => Math.abs(a - w) - Math.abs(b - w) || a - b);
  for (const k of order) {
    if (src[k] !== 1) continue;
    let best = null, bd = Infinity;
    for (const d of dates) {
      const g = Math.abs(t(d) - t(weeks[k])) / DAY;
      if (g < bd) { bd = g; best = d; }
    }
    if (bd <= 3) return best;
  }
  return null;
}

function nearestDate(dates, week, maxDays = Infinity) {
  let best = null, bd = Infinity;
  for (const d of dates) { const g = Math.abs(t(d) - t(week)) / DAY; if (g < bd) { bd = g; best = d; } }
  return bd <= maxDays ? best : null;
}

// field image: nearest of the dates when >= 95 % of THIS field was clear (clr = indices into rgb dates), within 10 days
export function fieldRgbDate(rgb, clr, week, maxDays = 10) {
  return clr && clr.length ? nearestDate(clr.map(i => rgb[i]), week, maxDays) : null;
}

// crop label in English and one colour per crop (farm-software conventions); group from the NRW register mapping
const CROP_COLOUR = { wheat: "#d9a441", barley: "#e8c872", "other cereal": "#c9a96e", maize: "#9bc53d", "sugar beet": "#a05ea8",
                      potato: "#a47148", rapeseed: "#f2d40c", "grass / fodder": "#4f9a5b", legume: "#6fb3a8", fallow: "#8d8a7a", other: "#7b8496" };
const SPECIAL = [[/^Gartenk(ü|u)rbis/i, "Pumpkin"], [/^M(ö|o)hre/i, "Carrot"], [/^Zwiebel/i, "Onion / leek"], [/^Erdbeere/i, "Strawberry"],
  [/^Spargel/i, "Asparagus"], [/^Winterhafer/i, "Winter oats"], [/^Sommerhafer/i, "Spring oats"], [/^Winterroggen/i, "Winter rye"],
  [/^Sommerroggen/i, "Spring rye"], [/^Wintertriticale/i, "Winter triticale"], [/^Sommertriticale/i, "Spring triticale"],
  [/^Gurke/i, "Cucumber"], [/^Zichorie/i, "Chicory"], [/^Ackerbohne/i, "Field bean"], [/^Gem(ü|u)seerbse/i, "Vegetable pea"],
  [/^Luzerne/i, "Lucerne"], [/^Kleegras/i, "Clover-grass"], [/^Baumschule/i, "Tree nursery"], [/^Chinaschilf|Miscanthus/i, "Miscanthus"], [/^Randstreifen/i, "Field margin strip"],
  // small crops and landscape elements of the NRW register (all names found in the four study areas, 2021-2025)
  [/^Bl(ü|u)hstreifen/i, "Flower strip"], [/Bl(ü|u)hfl(ä|a)che/i, "Flower area"], [/^(Ö|O)VF-Streifen/i, "Ecological focus strip"],
  [/^Streuobst/i, "Meadow orchard"], [/^Wildacker/i, "Game cover crop"], [/^Salatgurke/i, "Cucumber"], [/^Salat/i, "Lettuce"],
  [/^Gem(ü|u)sekohl/i, "Cabbage"], [/^Vertragsnaturs|^Naturschutz/i, "Nature conservation"], [/^Unbefestigte Mieten/i, "Field clamp"],
  [/menggetreide/i, "Mixed cereals"], [/^Gemenge|^Mischkulturen/i, "Mixed crop"], [/^Futterr(ü|u)be|Runkelr(ü|u)be/i, "Fodder beet"],
  [/Heide/i, "Heath"], [/^Uferrandstr/i, "Riparian buffer strip"], [/^Phacelia/i, "Phacelia"], [/^Beerenobst/i, "Berries"],
  [/^Sonnenblume/i, "Sunflower"], [/^Wildsamen/i, "Wild seed production"], [/beetweiser Anbau/i, "Market garden"],
  [/^Steinobst/i, "Stone fruit"], [/^Kernobst/i, "Pome fruit"], [/^Weihnachtsb(ä|a)ume/i, "Christmas trees"], [/^Forstfl/i, "Forest"],
  [/^Hanf/i, "Hemp"], [/^Silphium/i, "Silphium"], [/k(ü|u)rbis/i, "Pumpkin"], [/^Rhabarber/i, "Rhubarb"], [/^Tomate/i, "Tomato"],
  [/rettich/i, "Radish"], [/aus der Erzeugung/i, "Set-aside"], [/Erosion/i, "Erosion protection strip"], [/^Rebland/i, "Vineyard"],
  [/Senf/i, "Mustard"], [/^Mangold|Rote Be(e)?te|Rote R(ü|u)be/i, "Chard / beetroot"], [/^Tulpe/i, "Tulips"], [/hirse/i, "Millet"],
  [/^Sonstige Dauerkulturen/i, "Other permanent crops"], [/^Versuchsfl/i, "Trial plots"]];

export function cropStyle(group, crop) {
  const sp = SPECIAL.find(([re]) => re.test(String(crop || "")));
  if (sp && !["wheat", "barley", "maize", "sugar beet", "potato", "rapeseed"].includes(group))
    return { label: sp[1], colour: CROP_COLOUR[group] || CROP_COLOUR.other };
  const c = String(crop || ""), season = /^Winter/i.test(c) ? "Winter " : /^Sommer/i.test(c) ? "Spring " : "";
  const label = group === "wheat" ? `${season || "Winter "}wheat`
    : group === "barley" ? `${season || "Winter "}barley`
    : group === "maize" ? (/ohne Silomais/i.test(c) ? "Grain maize" : /Silomais/i.test(c) ? "Silage maize" : /Mais/i.test(c) ? "Grain maize" : "Maize")
    : group === "sugar beet" ? "Sugar beet" : group === "potato" ? "Potato" : group === "rapeseed" ? `${season || "Winter "}rapeseed`
    : group === "grass / fodder" ? "Grassland / fodder" : group === "legume" ? "Legume" : group === "fallow" ? "Fallow" : c || group;
  return { label: label.charAt(0).toUpperCase() + label.slice(1), colour: CROP_COLOUR[group] || CROP_COLOUR.other };
}

// crops of a study area: [{label, colour, group, n, ha, ids}], largest area first (props = fields.geojson properties)
export function cropSummary(props) {
  const by = {};
  for (const p of props) {
    const k = cropStyle(p.group, p.crop), e = by[k.label] ||= { label: k.label, colour: k.colour, group: p.group, n: 0, ha: 0, ids: [] };
    e.n++; e.ha += +p.ha || 0; e.ids.push(p.id);
  }
  return Object.values(by).map(e => ({ ...e, ha: Math.round(e.ha * 10) / 10 })).sort((a, b) => b.ha - a.ha || a.label.localeCompare(b.label));
}

// weekly median green LAI per crop label (crops with fewer than minN fields left out) for the area chart
export function cropMedians(props, fields, minN = 10, key = "lai") {
  const out = {};
  for (const e of cropSummary(props)) {
    if (e.n < minN) continue;
    const series = e.ids.map(id => fields[id]?.[key]).filter(Boolean);
    if (series.length < minN) continue;
    out[e.label] = series[0].map((_, w) => {
      const v = series.map(s => s[w]).filter(x => x != null).sort((a, b) => a - b);
      if (!v.length) return null;
      const m = Math.floor(v.length / 2);
      return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
    });
  }
  return out;
}

// animation frames: index of the first week (Monday) of every month
export function monthlyFrames(weeks) {
  return weeks.map((w, i) => [w.slice(0, 7), i]).filter(([m], i, a) => i === 0 || a[i - 1][0] !== m).map(([, i]) => i);
}

// colours of the value sources (measured, straight line, radar, after the last clear image); index 0 = no value
export const SOURCE_COLOURS = ["#555", "#4caf7d", "#7cc4ec", "#f2a93b", "#c07ad6"];

// weekly chart as an SVG string (no DOM, no d3): +- band and line over the weeks with a value, one dot per week
// coloured by its source, off-season weeks shaded, the selected week dashed
export function seriesChart({ values, unc, src, season, week, weeks, max, label }) {
  const n = values.length;
  if (!values.some(v => v != null)) return `<p class="note">No value this season for ${label}.</p>`;
  const W = 460, H = 160, L = 34, R = 10, T = 10, B = 24, f = v => +v.toFixed(1);
  const x = i => f(L + (n > 1 ? (i * (W - L - R)) / (n - 1) : 0)), y = v => f(H - B - (Math.max(0, Math.min(max, v)) / max) * (H - B - T));
  const runs = []; let cur = [];
  values.forEach((v, i) => { if (v != null) cur.push(i); else if (cur.length) { runs.push(cur); cur = []; } });
  if (cur.length) runs.push(cur);
  const band = runs.map(r => `M${r.map(i => `${x(i)},${y(values[i] + (unc[i] ?? 0))}`).join("L")}L${r.slice().reverse().map(i => `${x(i)},${y(values[i] - (unc[i] ?? 0))}`).join("L")}Z`).join("");
  const line = runs.map(r => `M${r.map(i => `${x(i)},${y(values[i])}`).join("L")}`).join("");
  const bw = f((W - L - R) / Math.max(1, n - 1));
  const off = season.map((s, i) => (s ? "" : `<rect x="${f(x(i) - bw / 2)}" y="${T}" width="${bw}" height="${H - B - T}" fill="#8b96b4" fill-opacity=".12"/>`)).join("");
  const ticks = [0, max / 2, max].map(t => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#8b96b4" stroke-opacity=".25"/><text x="${L - 5}" y="${y(t) + 4}" text-anchor="end">${+t.toFixed(2)}</text>`).join("");
  const months = weeks.map((w, i) => [w, i]).filter(([w]) => +w.slice(8, 10) <= 7 && ["11", "01", "03", "05", "07"].includes(w.slice(5, 7)))
    .map(([w, i]) => `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${["", "Jan", "", "Mar", "", "May", "", "Jul", "", "", "", "Nov"][+w.slice(5, 7)]}</text>`).join("");
  const dots = values.map((v, i) => (v == null ? "" : `<circle cx="${x(i)}" cy="${y(v)}" r="2.8" fill="${SOURCE_COLOURS[src[i]] || SOURCE_COLOURS[0]}"><title>${weeks[i]}: ${v}</title></circle>`)).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly ${label}">${off}${ticks}${months}
    <path d="${band}" fill="#8b96b4" fill-opacity=".22"/><path d="${line}" fill="none" stroke="#c9d2e6" stroke-width="1.4"/>${dots}
    <line x1="${x(week)}" x2="${x(week)}" y1="${T}" y2="${H - B}" stroke="#8b96b4" stroke-dasharray="3 3"/></svg>`;
}

// map true-colour layer: nearest date with at least minShare of the area's field pixels clear, within maxDays
export function areaRgbDate(rgb, share, week, minShare = 0.6, maxDays = 21) {
  return nearestDate(rgb.filter((_, i) => (share?.[i] ?? 0) >= minShare), week, maxDays);
}

// legend of the field layers (colours drawn by the page; ramps match the exported PNGs)
export function legendSpec(layer) {
  if (layer === "lai") return { kind: "ramp", scheme: "YlGn", min: 0, max: 5, unit: "green LAI (m²/m²)" };
  if (layer === "ndvi") return { kind: "ramp", scheme: "YlGn", min: 0, max: 1, unit: "NDVI" };
  if (layer === "unc") return { kind: "ramp", scheme: "Purples", min: 0, max: 0.8, unit: "± LAI (typical error)" };
  if (layer === "change") return { kind: "ramp", scheme: "RdBu", min: -1, max: 1, unit: "LAI change since last week (grey: within noise)" };
  if (layer === "source") return { kind: "swatches", items: ["no data", "measured", "straight line", "radar-filled", "after last clear image"] };
  if (layer === "alert") return { kind: "swatches", items: ["alert: below the median field or falling while it rises (2 weeks)"] };
  return { kind: "text" };
}

export function fmtUnc(v, u, d) {
  if (v == null || !isFinite(v)) return "no data";
  return `${(+v).toFixed(d)} ± ${(+u).toFixed(d)}`;
}

// ---- MY FIELDS: picked fields and zone rates, kept in the browser (no accounts) ----
export const MY_FIELDS_KEY = "cropOutlook.myFields.v1";
const EMPTY = () => ({ fields: [], rates: {} });

export function loadMyFields(storage, defaults = []) {
  try {
    const s = storage && storage.getItem(MY_FIELDS_KEY);
    if (!s) return { fields: defaults.slice(), rates: {} };          // first visit: the example fields
    const v = JSON.parse(s);
    if (!v || !Array.isArray(v.fields) || typeof v.rates !== "object" || v.rates === null) return EMPTY();
    const good = f => f && typeof f.area === "string" && Number.isInteger(f.year) && Number.isFinite(f.id);
    return { fields: v.fields.filter(good), rates: v.rates };
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

// Module A: plain-language summary of a field's 5-season zones (meta from field/<area>_2025/zones.json)
export function zoneSummary(m) {
  if (!m || !m.tests) return null;
  const share = m.held / m.tests, reliability = share >= 0.8 ? "reliable" : share >= 0.6 ? "fairly reliable" : "weak";
  const uniform = m.spread < 0.05;
  const lines = [`The zone order held in ${m.held} of ${m.tests} seasons it was not built from (${reliability}).`,
    uniform ? `This field is fairly uniform: the zones differ by less than 5 % in growth, so variable rates bring little here.`
            : `The strong zone grew about ${Math.round(m.spread * 100)} % more than the weak zone (season total of green leaf).`];
  const cause = [];
  if (m.twi != null && m.twi <= -0.3) cause.push("the weak zone lies on the drier, upper or convex part of the field");
  if (m.twi != null && m.twi >= 0.3) cause.push("the weak zone lies in a wetter, lower part where water may stand");
  if (m.slope != null && m.slope >= 0.5) cause.push("the weak zone is steeper");
  if (m.nfk != null && m.nfk <= -10) cause.push("the soil map shows less plant-available water in the weak zone");
  if (m.nfk != null && m.nfk >= 10) cause.push("the soil map shows more plant-available water in the weak zone, so other causes are likely");
  lines.push(cause.length ? `Likely reason: ${cause.join("; ")}.`
    : "No clear soil or terrain pattern; other causes (management, compaction, old field boundaries) are possible.");
  return { reliability, uniform, lines };
}

// variable-rate products the user can choose (only the label and unit change; the rates are always the user's)
export const PRODUCTS = { nitrogen: { label: "Nitrogen", unit: "kg N/ha" }, seed: { label: "Seed", unit: "kg/ha" }, other: { label: "Other product", unit: "kg/ha" } };

// hectares per zone (weak, middle, strong) from the field's 10 m pixels; with one zone everything counts as middle
export function zoneAreas(classes, n, pixelHa = 0.01, weights = null) {
  const a = [0, 0, 0];
  classes.forEach((c, i) => { if (c >= 0) a[n === 1 ? 1 : c] += pixelHa * (weights ? weights[i] : 1); });
  return a.map(v => Math.round(v * 1000) / 1000);
}

// total amount for the field (kg); null until every zone has a rate
export function zoneTotal(areas, rates) {
  if (!rates || rates.some(v => v === "" || v == null || !isFinite(v))) return null;
  return areas.reduce((s, a, i) => s + a * Number(rates[i]), 0);
}

// per-field classes on the study-area map (field/<area>_2025/classes.json)
export const CLASS_STYLES = {
  prod: { low: "#d9534f", average: "#f0ad4e", high: "#2e9e5b", "not assessed": "#3a4157" },
  var: { uniform: "#7cc4ec", "variable, reliable zones": "#c07ad6", "variable, less stable zones": "#f2a93b", "not assessed": "#3a4157" },
};
export function classLegend(classes, props, kind, crop = null) {
  if (!classes) return null;
  const key = kind === "prod" ? "prod_class" : "var_class", rows = Object.keys(CLASS_STYLES[kind]).map(label => ({ label, colour: CLASS_STYLES[kind][label], n: 0, ha: 0 }));
  for (const p of props) {
    if (crop && cropStyle(p.group, p.crop).label !== crop) continue;
    const c = classes.fields[String(p.id)]?.[key] || "not assessed", r = rows.find(x => x.label === c);
    if (r) { r.n++; r.ha += +p.ha || 0; }
  }
  return rows.map(r => ({ ...r, ha: Math.round(r.ha * 10) / 10 }));
}

// field counts per crop (label) and class, largest crop area first
export function classByCrop(classes, props, kind) {
  if (!classes) return [];
  const key = kind === "prod" ? "prod_class" : "var_class", by = {};
  for (const p of props) {
    const crop = cropStyle(p.group, p.crop).label, c = classes.fields[String(p.id)]?.[key] || "not assessed";
    const e = by[crop] ||= { crop, ha: 0, counts: Object.fromEntries(Object.keys(CLASS_STYLES[kind]).map(k => [k, 0])) };
    e.ha += +p.ha || 0; e.counts[c] = (e.counts[c] || 0) + 1;
  }
  return Object.values(by).sort((a, b) => b.ha - a.ha);
}

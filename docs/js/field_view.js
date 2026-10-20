// Field level in the EXPLORER: zoom from districts to the NRW study areas, fields and 10 m pixels.
// Uses the globals of the district app (S, META, GEO, d3, css, fmt, rowFor, cropData, drawMap, drawPanel, doyDate, LEADS).
import { weekIndex, sourceLabel, withinNoise, nearestRgbDate, clearRgbDate, fmtUnc, affineFromCorners, leavesField,
         fieldUrl, noDataReason, zoomAllowed, areaToEnter, fieldRgbDate, areaRgbDate, legendSpec, groupMedian, seriesChart, SOURCE_COLOURS, cropStyle, vsMedianText } from "./field_core.js";

const BASE = "field/";
const ENTER_K = 40;                                   // zoom factor at which an area switches to fields
const GROUP_CROP = { wheat: "winter_wheat", barley: "winter_barley", maize: "silage_maize", potato: "potato" };
const ARABLE = new Set(["wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"]);
const CROP_GROUP = { winter_wheat: "wheat", winter_barley: "barley", silage_maize: "maize", grain_maize: "maize", potato: "potato" };
export const FV = { v: null, idx: null, districts: {}, active: false, area: null, year: null, geo: null, series: null, rgb: [],
                    week: 0, layer: "lai", sel: null, cache: {} };

export const url = p => fieldUrl(BASE, p, FV.v);                // ?v=<data version> on every field file
const getJ = (p, opt) => fetch(url(p), opt).then(r => { if (!r.ok) throw new Error(p); return r.json(); });
const proj = () => d3.geoMercator().fitSize([560, 720], GEO);

export async function loadSeason(area, year) {
  const k = `${area}_${year}`;
  if (!FV.cache[k]) {
    const [geo, series, rgb] = await Promise.all([getJ(`${k}/fields.geojson`), getJ(`${k}/series.json`), getJ(`${k}/rgb.json`)]);
    FV.cache[k] = { geo, series, rgb };
  }
  return FV.cache[k];
}

function seasonFor(area) {
  const s = FV.idx.areas[area].seasons, y = +S.year;
  return s.includes(y) ? y : s.reduce((a, b) => (Math.abs(b - y) < Math.abs(a - y) ? b : a), s[0]);
}

function defaultWeek(weeks) {
  const c = META.crops[S.crop], doy = c.leads[LEADS[S.lead]];
  const d = new Date(Date.UTC(FV.year, 0, 1)); d.setUTCDate(doy);
  const k = weekIndex(weeks, d.toISOString().slice(0, 10));
  return k < 0 ? 0 : k;
}

export async function initField() {
  try { FV.idx = await fetch(BASE + "index.json", { cache: "no-store" }).then(r => r.json()); } catch (e) { return; }
  FV.v = FV.idx.version || null;
  FV.districts = await getJ("districts.json").catch(() => ({}));
  const svg = d3.select("#map");
  const zoom = d3.zoom().scaleExtent([1, 600]).translateExtent([[0, 0], [560, 720]])
    .filter(e => zoomAllowed({ type: e.type, ctrlKey: e.ctrlKey, touches: e.touches?.length, button: e.button,
                                k: d3.zoomTransform(svg.node()).k, field: FV.active }))
    .on("zoom", e => {
      d3.select("#mapg").attr("transform", e.transform);
      if (leavesField({ active: FV.active, k: e.transform.k, user: !!e.sourceEvent, enterK: ENTER_K })) leaveField();
      else if (!FV.active && e.sourceEvent && !FV.entering) {               // zooming into a study area opens it
        const a = areaToEnter({ k: e.transform.k, enterK: ENTER_K, centre: e.transform.invert([280, 360]), boxes: areaBoxes() });
        if (a) enterField(a);
      }
    });
  svg.call(zoom).on("dblclick.zoom", null);
  FV.zoom = zoom;
  drawAreas();
  const sel = document.getElementById("fieldlayer");
  sel.onchange = e => { FV.layer = e.target.value; drawFields(); };
  window.FIELDMODE = { get active() { return FV.active; }, onSlide: i => { FV.week = i; label(); drawFields(); if (FV.sel) fieldPanel(FV.sel); } };
}

function areaBoxes() {
  const p = proj(), out = {};
  for (const a of Object.keys(FV.idx.areas)) {
    const [ul, ur, ll] = FV.idx.areas[a].corners_wgs84, lr = [ur[0] + ll[0] - ul[0], ur[1] + ll[1] - ul[1]];
    out[a] = [ul, ur, lr, ll].map(c => p(c));
  }
  return out;
}

function areaPath(a) {
  const p = proj(), [ul, ur, ll] = FV.idx.areas[a].corners_wgs84;
  const lr = [ur[0] + ll[0] - ul[0], ur[1] + ll[1] - ul[1]];
  return "M" + [ul, ur, lr, ll].map(c => p(c).join(",")).join("L") + "Z";
}

export function drawAreas() {
  if (!FV.idx) return;
  const g = d3.select("#mapg").selectAll("path.area").data(Object.keys(FV.idx.areas), d => d);
  g.join("path").attr("class", "area").attr("d", areaPath)
    .on("click", (e, a) => { e.stopPropagation(); enterField(a); })
    .append("title").text(a => `Field level: study area ${a} (NRW) — click to open`);
}

export async function enterField(area) {
  const year = seasonFor(area);
  let d;
  FV.entering = true;
  try { d = await loadSeason(area, year); } catch (e) { FV.entering = false; return; }   // data missing: stay on districts
  FV.entering = false;
  FV.area = area; FV.year = year; FV.active = true;
  d3.select("#mapg").classed("fieldmode", true);              // districts in neutral grey: no forecast colour behind field gaps
  FV.geo = d.geo; FV.series = d.series; FV.rgb = d.rgb;
  FV.week = defaultWeek(FV.series.weeks);
  const p = proj(), [ul, ur, ll] = FV.idx.areas[area].corners_wgs84;
  const a = p(ul), b = p(ur), c = p(ll), cx = (b[0] + c[0]) / 2, cy = (b[1] + c[1]) / 2, k = 0.9 * 560 / Math.abs(b[0] - a[0]);
  d3.select("#map").transition().duration(750).call(FV.zoom.transform, d3.zoomIdentity.translate(280 - k * cx, 360 - k * cy).scale(k));
  document.getElementById("fieldlayer-ctl").hidden = false;
  const inp = document.getElementById("lead");
  inp.max = FV.series.weeks.length - 1; inp.value = FV.week;
  label(); drawFields();
  document.getElementById("panel").innerHTML = `<h2>Study area ${area} · season ${FV.year}</h2>
    <p class="note">${Object.keys(FV.series.fields).length} registered fields. Click a field for its weekly green LAI, NDVI and Sentinel-2 image.
    ${FV.year !== +S.year ? `Field data exist for 2021-2025; showing ${FV.year}.` : ""}</p>
    <p><button id="fv-back">Back to districts</button></p>${attribution()}`;
  document.getElementById("fv-back").onclick = leaveField;
}

export function leaveField() {
  if (!FV.active) return;
  FV.active = false; FV.sel = null;
  d3.select("#mapg").classed("fieldmode", false);
  d3.select("#fieldg").remove();
  document.getElementById("fieldlayer-ctl").hidden = true;
  const inp = document.getElementById("lead");
  inp.max = 5; inp.value = S.lead;
  d3.select("#map").transition().duration(500).call(FV.zoom.transform, d3.zoomIdentity);
  setLeadLabel(); drawMap(); drawPanel();
}

function label() {
  const w = FV.series.weeks[FV.week];
  const d = new Date(w + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  document.getElementById("leadval").textContent = `Week of ${d} · re-analysis (past season ${FV.year})`;
}

const SRC_COL = SOURCE_COLOURS;
function fieldFill(f) {
  const s = FV.series.fields[f.properties.id], w = FV.week;
  if (!s) return "none";
  if (FV.layer === "source") return SRC_COL[s.src[w]];
  if (FV.layer === "unc") return s.lai_unc[w] == null ? "none" : d3.interpolatePurples(Math.min(1, s.lai_unc[w] / 0.8));
  if (FV.layer === "change") {
    if (w === 0 || s.lai[w] == null || s.lai[w - 1] == null) return "none";
    const dv = s.lai[w] - s.lai[w - 1];
    return withinNoise(dv, s.lai_unc[w] ?? 0) ? "#8b96b4" : d3.interpolateRdBu(0.5 + Math.max(-0.5, Math.min(0.5, dv / 2)));
  }
  if (FV.layer === "alert") return s.alert[w] ? "#e0562a" : "none";
  return "none";
}

export function drawFields() {
  if (!FV.active) return;
  const p = proj(), k = `${FV.area}_${FV.year}`, wk = FV.series.weeks[FV.week];
  let g = d3.select("#fieldg");
  if (g.empty()) g = d3.select("#mapg").append("g").attr("id", "fieldg");
  const [ul, ur, ll] = FV.idx.areas[FV.area].corners_wgs84.map(c => p(c));
  const m = affineFromCorners(ul, ur, ll, 800, 800);
  const rgbD = FV.layer === "rgb" ? areaRgbDate(FV.rgb, FV.series.rgb_share, wk) : null;
  const pix = FV.layer === "lai" || FV.layer === "ndvi" || FV.layer === "rgb";
  const src = FV.layer === "rgb" ? (rgbD ? url(`${k}/rgb/${rgbD}.jpg`) : null) : url(`${k}/${FV.layer}/${wk}.png`);
  g.selectAll("image.ov").data(pix && src ? [src] : [])
    .join("image").attr("class", "ov").attr("href", d => d).attr("width", 800).attr("height", 800)
    .attr("transform", `matrix(${m.join(",")})`).style("image-rendering", "pixelated");
  const path = d3.geoPath(p);
  g.selectAll("path.f").data(FV.geo.features, f => f.properties.id)
    .join("path").attr("class", f => "f" + (f.properties.id === FV.sel ? " sel" : "")).attr("d", path)
    .attr("fill", pix ? "transparent" : fieldFill)
    .on("click", (e, f) => { e.stopPropagation(); FV.sel = f.properties.id; drawFields(); fieldPanel(f.properties.id); })
    .selectAll("title").data(f => [f]).join("title")
    .text(f => { const s = FV.series.fields[f.properties.id]; return `Field ${f.properties.id} · ${f.properties.crop}` + (s ? ` · LAI ${fmtUnc(s.lai[FV.week], s.lai_unc[FV.week], 1)}` : ""); });
  fieldLegend(rgbD);
}

// colour bar and note under the map follow the field layer (the district t/ha scale is not shown in the field view)
function fieldLegend(rgbD) {
  const L = legendSpec(FV.layer), wk = FV.series.weeks[FV.week];
  let h = "";
  if (L.kind === "ramp") {
    const ip = { YlGn: d3.interpolateYlGn, Purples: d3.interpolatePurples, RdBu: d3.interpolateRdBu }[L.scheme];
    const stops = d3.range(0, 1.01, 0.1).map(t => ip(L.scheme === "RdBu" ? t : 0.05 + 0.95 * t));
    h = `<span>${L.min}</span><span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span><span>${L.max} ${L.unit}</span>`;
  } else if (L.kind === "swatches") {
    const col = FV.layer === "alert" ? ["#e0562a"] : SRC_COL;
    h = L.items.map((t, i) => `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${col[i]}"></span> ${t}</span>`).join(" ");
  } else {
    h = rgbD ? `<span>Sentinel-2 true colour, ${rgbD} (nearest date clear over most of the area)</span>` : "<span>No clear Sentinel-2 image of this area within 3 weeks of this week.</span>";
  }
  document.getElementById("legend").innerHTML = h;
  document.getElementById("mapnote").textContent = `Field level, study area ${FV.area}, week of ${wk}: weekly values per 10 m pixel (re-analysis of a past season). `
    + "Dark grey: not a registered field or no value yet. Click a field for its curve, uncertainty and Sentinel-2 image.";
}

export function attribution() {
  const y = FV.year ?? "";
  return `<p class="note">Contains modified Copernicus Sentinel data ${y}. Field boundaries: Land NRW (dl-de/by-2-0).</p>`;
}

async function districtContext(f) {
  const crop = GROUP_CROP[f.properties.group], code = f.properties.district;
  if (!crop || !code) return "";
  await cropData(crop);
  const r = (S.data[crop].fc[String(FV.year)] || {})[LEADS[S.lead]]?.[code];
  const name = META.names[code] || code;
  return r ? `<h3>District context</h3><p>${name} · ${META.crops[crop].label} ${FV.year}, forecast ${LEADS[S.lead]} weeks before harvest:
    <b>${fmt(r[0])} t/ha</b>${r[1] != null ? ` (80 %: ${fmt(r[1])}–${fmt(r[2])})` : ""}${r[3] != null ? ` · official ${fmt(r[3])}` : ""}</p>` : "";
}

function rgbCanvas(s) {
  const d = fieldRgbDate(FV.rgb, s.clr, FV.series.weeks[FV.week]);
  if (!d) return "";                                         // no image unless >= 95 % of the field was clear within 10 days
  return `<canvas id="fv-rgb" width="320" height="240" style="width:100%;max-width:320px;border-radius:6px"></canvas>
    <p class="note">Sentinel-2 true colour, ${d} (≥ 95 % of the field clear)</p>`;
}

function paintRgb(s) {
  const c = document.getElementById("fv-rgb");
  if (!c) return;
  const d = fieldRgbDate(FV.rgb, s.clr, FV.series.weeks[FV.week]), [r0, r1, c0, c1] = s.bb, pad = 6;
  const img = new Image();
  img.onload = () => {
    const sx = Math.max(0, c0 - pad), sy = Math.max(0, r0 - pad), sw = Math.min(800, c1 + pad) - sx, sh = Math.min(800, r1 + pad) - sy;
    const sc = Math.min(c.width / sw, c.height / sh), dw = sw * sc, dh = sh * sc;
    const ctx = c.getContext("2d");
    ctx.fillStyle = css("--panel"); ctx.fillRect(0, 0, c.width, c.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, sx, sy, sw, sh, (c.width - dw) / 2, (c.height - dh) / 2, dw, dh);
    ctx.strokeStyle = "#f2a93b"; ctx.lineWidth = 2;
    ctx.strokeRect((c.width - dw) / 2 + (c0 - sx) * sc, (c.height - dh) / 2 + (r0 - sy) * sc, (c1 - c0) * sc, (r1 - r0) * sc);
  };
  img.src = url(`${FV.area}_${FV.year}/rgb/${d}.jpg`);
}

export async function fieldPanel(id) {
  const f = FV.geo.features.find(x => x.properties.id === id), s = FV.series.fields[id], w = FV.week;
  if (!f || !s) return;
  const wk = FV.series.weeks[w], P = f.properties;
  const med = ARABLE.has(s.group) ? groupMedian(FV.series.fields, s.group, w, 10) : null;   // mixed groups have no meaningful median
  const diff = s.lai[w] != null && med != null ? s.lai[w] - med : null;
  const vsMed = vsMedianText(s.lai[w], med, s.lai_unc[w]).replace(/ \(.*\)$/, "");
  const cs = cropStyle(P.group, P.crop);
  let h = `<h2><span class="crop" style="--c:${cs.colour}"><i></i>${cs.label}</span> Field ${id}</h2>
    <div class="note">${P.crop} (NRW register) · ${fmt(P.ha, 2)} ha · season ${FV.year} · week of ${wk} · re-analysis (past season)</div>
    ${P.partial ? '<p><span class="pill bad">only partly inside the study area: values cover the inside part</span></p>' : ""}
    ${s.season[w] ? "" : '<p><span class="pill">before sowing / after harvest: previous crop, cover crop or bare soil</span></p>'}
    ${noDataReason(s.src, w) ? `<p class="note">No value this week: ${noDataReason(s.src, w)}.</p>` : ""}
    <div class="kpis">
      <div class="kpi"><div class="k">Green LAI</div><div class="v">${fmtUnc(s.lai[w], s.lai_unc[w], 1)}</div><div class="s">m²/m² · typical error of a 10 m value</div></div>
      <div class="kpi"><div class="k">NDVI</div><div class="v">${fmtUnc(s.ndvi[w], s.ndvi_unc[w], 2)}</div><div class="s">${s.ndvi_src[w] === 1 ? "± assumed sensor noise" : "± typical error"}</div></div>
      <div class="kpi"><div class="k">vs median ${s.group}</div><div class="v">${vsMed}</div><div class="s">${diff == null ? (ARABLE.has(s.group) ? "too few fields" : "not compared (mixed group)") : withinNoise(diff, s.lai_unc[w] ?? 0) ? "within noise" : "beyond the uncertainty"}</div></div>
    </div>
    <p><span class="pill">${sourceLabel(s.src[w])}</span> ${s.alert[w] === 1 ? '<span class="pill bad">below the median field for 2 weeks</span>' : s.alert[w] === 2 ? '<span class="pill bad">falling while the median rises</span>' : ""}</p>
    <h3>Weekly green LAI</h3>${seriesChart({ values: s.lai, unc: s.lai_unc, src: s.src, season: s.season, week: w, weeks: FV.series.weeks, max: 6, label: "green LAI" })}
    <h3>Weekly NDVI</h3>${seriesChart({ values: s.ndvi, unc: s.ndvi_unc, src: s.ndvi_src, season: s.season, week: w, weeks: FV.series.weeks, max: 1, label: "NDVI" })}
    <p class="note">${SOURCE_COLOURS.map((c, i) => i ? `<span style="margin-right:10px;white-space:nowrap"><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c}"></span> ${sourceLabel(i)}</span>` : "").join("")}<br>Band: ± typical error. Shaded weeks: before sowing / after harvest. Dashed: selected week.</p>
    ${rgbCanvas(s) ? `<h3>Sentinel-2 image</h3>${rgbCanvas(s)}` : ""}`;
  h += await districtContext(f);
  h += `<p class="note"><b>Field yield: not shown yet</b> (needs field yield data).</p>
    <p><button id="fv-add">Add to my fields</button> <button id="fv-back">Back to districts</button></p>${attribution()}`;
  document.getElementById("panel").innerHTML = h;
  paintRgb(s);
  document.getElementById("fv-back").onclick = leaveField;
  document.getElementById("fv-add").onclick = () => window.MYFIELDS?.add({ area: FV.area, year: FV.year, id });
}

// one line for the district panel: field greenness of the study-area fields in this district this week
export function districtFieldLine(code) {
  const grp = CROP_GROUP[S.crop], e = FV.districts?.[S.year]?.[code], g = e?.groups?.[grp];
  if (!g) return "";
  const c = META.crops[S.crop], dd = new Date(Date.UTC(+S.year, 0, 1)); dd.setUTCDate(c.leads[LEADS[S.lead]]);
  const k = weekIndex(e.weeks, dd.toISOString().slice(0, 10)), v = k >= 0 ? g.median_lai[k] : null;
  return v == null ? "" : `<h3>Field greenness (NRW study areas)</h3><p>${g.n_fields} ${grp} fields monitored · median green LAI <b>${fmt(v, 1)}</b> in the week of ${e.weeks[k]}.</p>`;
}

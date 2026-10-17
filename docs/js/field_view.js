// Field level in the EXPLORER: zoom from districts to the NRW study areas, fields and 10 m pixels.
// Uses the globals of the district app (S, META, GEO, d3, css, fmt, rowFor, cropData, drawMap, drawPanel, doyDate, LEADS).
import { weekIndex, sourceLabel, withinNoise, nearestRgbDate, fmtUnc, affineFromCorners } from "./field_core.js";

const BASE = "field/";
const ENTER_K = 40;                                   // zoom factor at which an area switches to fields
const GROUP_CROP = { wheat: "winter_wheat", barley: "winter_barley", maize: "silage_maize", potato: "potato" };
const CROP_GROUP = { winter_wheat: "wheat", winter_barley: "barley", silage_maize: "maize", grain_maize: "maize", potato: "potato" };
export const FV = { idx: null, districts: {}, active: false, area: null, year: null, geo: null, series: null, rgb: [],
                    week: 0, layer: "lai", sel: null, cache: {} };

const getJ = p => fetch(BASE + p).then(r => { if (!r.ok) throw new Error(p); return r.json(); });
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
  try { FV.idx = await getJ("index.json"); } catch (e) { return; }
  FV.districts = await getJ("districts.json").catch(() => ({}));
  const svg = d3.select("#map");
  const zoom = d3.zoom().scaleExtent([1, 600]).on("zoom", e => {
    d3.select("#mapg").attr("transform", e.transform);
    if (FV.active && e.transform.k < ENTER_K / 2) leaveField();
  });
  svg.call(zoom).on("dblclick.zoom", null);
  FV.zoom = zoom;
  drawAreas();
  const sel = document.getElementById("fieldlayer");
  sel.onchange = e => { FV.layer = e.target.value; drawFields(); };
  window.FIELDMODE = { get active() { return FV.active; }, onSlide: i => { FV.week = i; label(); drawFields(); if (FV.sel) fieldPanel(FV.sel); } };
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
  FV.area = area; FV.year = seasonFor(area); FV.active = true;
  const d = await loadSeason(area, FV.year);
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

const SRC_COL = ["#555", "#4caf7d", "#7cc4ec", "#f2a93b", "#c07ad6"];
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
  const pix = FV.layer === "lai" || FV.layer === "ndvi";
  g.selectAll("image.ov").data(pix ? [`${BASE}${k}/${FV.layer}/${wk}.png`] : [])
    .join("image").attr("class", "ov").attr("href", d => d).attr("width", 800).attr("height", 800)
    .attr("transform", `matrix(${m.join(",")})`).style("image-rendering", "pixelated");
  const path = d3.geoPath(p);
  g.selectAll("path.f").data(FV.geo.features, f => f.properties.id)
    .join("path").attr("class", f => "f" + (f.properties.id === FV.sel ? " sel" : "")).attr("d", path)
    .attr("fill", pix ? "transparent" : fieldFill)
    .on("click", (e, f) => { e.stopPropagation(); FV.sel = f.properties.id; drawFields(); fieldPanel(f.properties.id); })
    .selectAll("title").data(f => [f]).join("title")
    .text(f => { const s = FV.series.fields[f.properties.id]; return `Field ${f.properties.id} · ${f.properties.crop}` + (s ? ` · LAI ${fmtUnc(s.lai[FV.week], s.lai_unc[FV.week], 1)}` : ""); });
}

function curve(s) {
  const W = 460, H = 170, L = 34, R = 10, T = 10, B = 24, n = FV.series.weeks.length;
  const x = d3.scaleLinear().domain([0, n - 1]).range([L, W - R]);
  const y = d3.scaleLinear().domain([0, Math.max(4, d3.max(s.lai.map((v, i) => (v ?? 0) + (s.lai_unc[i] ?? 0))))]).nice().range([H - B, T]);
  const pts = s.lai.map((v, i) => [i, v, s.lai_unc[i]]);
  const area = d3.area().defined(p => p[1] != null).x(p => x(p[0])).y0(p => y(Math.max(0, p[1] - (p[2] ?? 0)))).y1(p => y(p[1] + (p[2] ?? 0)));
  const line = d3.line().defined(p => p[1] != null).x(p => x(p[0])).y(p => y(p[1]));
  const off = s.season.map((v, i) => (v ? "" : `<rect x="${x(i) - (W - L - R) / n / 2}" y="${T}" width="${(W - L - R) / n}" height="${H - B - T}" fill="${css("--line")}" fill-opacity=".35"/>`)).join("");
  const months = FV.series.weeks.map((w, i) => [w, i]).filter(([w]) => w.slice(8, 10) <= "07" && ["11", "01", "03", "05", "07"].includes(w.slice(5, 7)))
    .map(([w, i]) => `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${new Date(w + "T00:00:00Z").toLocaleDateString("en-GB", { month: "short", timeZone: "UTC" })}</text>`).join("");
  const yt = y.ticks(4).map(t => `<line class="ax" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 5}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
  const meas = pts.filter((p, i) => s.src[i] === 1 && p[1] != null).map(p => `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2.6" fill="${css("--ink")}"/>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Weekly green LAI of this field">${off}${yt}${months}
    <path d="${area(pts)}" fill="${css("--accent")}" fill-opacity=".18"/><path d="${line(pts)}" fill="none" stroke="${css("--accent")}" stroke-width="2"/>
    ${meas}<line x1="${x(FV.week)}" x2="${x(FV.week)}" y1="${T}" y2="${H - B}" stroke="${css("--muted")}" stroke-dasharray="3 3"/></svg>`;
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
  const wk = FV.series.weeks[FV.week], d = nearestRgbDate(FV.rgb, wk);
  if (!d) return `<p class="note">No clear Sentinel-2 image within 10 days of this week.</p>`;
  return `<canvas id="fv-rgb" width="320" height="240" style="width:100%;max-width:320px;border-radius:6px"></canvas>
    <p class="note">Sentinel-2 true colour, ${d}</p>`;
}

function paintRgb(s) {
  const c = document.getElementById("fv-rgb");
  if (!c) return;
  const d = nearestRgbDate(FV.rgb, FV.series.weeks[FV.week]), [r0, r1, c0, c1] = s.bb, pad = 6;
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
  img.src = `${BASE}${FV.area}_${FV.year}/rgb/${d}.jpg`;
}

export async function fieldPanel(id) {
  const f = FV.geo.features.find(x => x.properties.id === id), s = FV.series.fields[id], w = FV.week;
  if (!f || !s) return;
  const wk = FV.series.weeks[w], P = f.properties;
  const same = Object.values(FV.series.fields).filter(x => x.group === s.group && x.lai[w] != null).map(x => x.lai[w]).sort(d3.ascending);
  const med = same.length ? d3.quantile(same, 0.5) : null, diff = s.lai[w] != null && med != null ? s.lai[w] - med : null;
  const vsMed = diff == null ? "–" : `${diff > 0 ? "+" : ""}${fmt(100 * diff / med, 0)} %`;
  let h = `<h2>Field ${id} · ${P.crop}</h2>
    <div class="note">${P.group} · ${fmt(P.ha, 2)} ha · season ${FV.year} · week of ${wk} · re-analysis (past season)</div>
    ${P.partial ? '<p><span class="pill bad">only partly inside the study area: values cover the inside part</span></p>' : ""}
    ${s.season[w] ? "" : '<p><span class="pill">before sowing / after harvest: previous crop, cover crop or bare soil</span></p>'}
    <div class="kpis">
      <div class="kpi"><div class="k">Green LAI</div><div class="v">${fmtUnc(s.lai[w], s.lai_unc[w], 1)}</div><div class="s">m²/m² · typical error of a 10 m value</div></div>
      <div class="kpi"><div class="k">NDVI</div><div class="v">${fmtUnc(s.ndvi[w], s.ndvi_unc[w], 2)}</div><div class="s">${s.ndvi_src[w] === 1 ? "± assumed sensor noise" : "± typical error"}</div></div>
      <div class="kpi"><div class="k">vs median ${s.group}</div><div class="v">${vsMed}</div><div class="s">${diff == null ? "" : withinNoise(diff, s.lai_unc[w] ?? 0) ? "within noise" : "beyond the uncertainty"}</div></div>
    </div>
    <p><span class="pill">${sourceLabel(s.src[w])}</span> ${s.alert[w] === 1 ? '<span class="pill bad">below the median field for 2 weeks</span>' : s.alert[w] === 2 ? '<span class="pill bad">falling while the median rises</span>' : ""}</p>
    <h3>Weekly green LAI</h3>${curve(s)}
    <p class="note">Band: ± typical error. Dots: clear Sentinel-2 images. Grey weeks: before sowing / after harvest. Dashed: selected week.</p>
    <h3>Sentinel-2 image</h3>${rgbCanvas(s)}`;
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

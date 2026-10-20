// MY FIELDS: fields picked on the map (kept in this browser, no accounts), list + map, zones, rates, exports.
import { loadMyFields, saveMyFields, zones, fmtUnc, sourceLabel, withinNoise, nearestRgbDate, clearRgbDate,
         affineFromCorners, invertAffine, applyAffine, groupMedian, exportCheck, noDataReason, fieldRgbDate,
         seriesChart, SOURCE_COLOURS, legendSpec, cropStyle, vsMedianText, cropSummary, cropMedians, monthlyFrames } from "./field_core.js";
const badge = (g, c) => { const k = cropStyle(g, c); return `<span class="crop" style="--c:${k.colour}"><i></i>${k.label}</span>`; };

import { buildTaskData } from "./isoxml.js";
import { FV, loadSeason, attribution, url } from "./field_view.js";

const ARABLE = new Set(["wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"]);
const ZCOL = ["#d9534f", "#f0ad4e", "#2e9e5b"];
// example fields shown on a first visit (one per study area, season 2025); users can remove them or add their own
export const EXAMPLES = [{ area: "pilot", year: 2025, id: 8702226 }, { area: "selhausen", year: 2025, id: 8312694 },
                         { area: "north", year: 2025, id: 8910847 }, { area: "south", year: 2025, id: 8965510 }];
let ST = loadMyFields(safeStorage(), EXAMPLES);
let CUR = null;
const AREA_NAMES = { pilot: "Pilot · Düren", selhausen: "Selhausen", north: "North", south: "South" };
const AV = { area: "pilot", year: 2025, crop: null, geo: null, series: null, view: "crops", frame: 0, step: "month", timer: null };   // the study area shown
const NOVAL = "#3a4157";                     // field pixel without a value yet (before its first clear image)
const PXIDX = {};

function safeStorage() { try { return window.localStorage; } catch (e) { return null; } }
function persist() { try { saveMyFields(window.localStorage, ST); } catch (e) { /* storage blocked: keep in memory */ } }
const key = f => `${f.area}_${f.year}_${f.id}`;

export function add(f) {
  if (!ST.fields.some(x => key(x) === key(f))) ST.fields.push(f);
  persist(); render();
  const b = document.getElementById("fv-add");
  if (b) { b.textContent = "Added to my fields ✓"; b.disabled = true; }
}

function remove(f) {
  ST.fields = ST.fields.filter(x => key(x) !== key(f));
  delete ST.rates[f.id];
  persist(); if (CUR && key(CUR) === key(f)) CUR = null; render();
}

async function pixelBlock(area, year, id) {
  const k = `${area}_${year}`;
  if (!PXIDX[k]) PXIDX[k] = await fetch(url(`${k}/px.json`)).then(r => r.json());
  const e = PXIDX[k].fields[String(id)];
  const r = await fetch(url(`${k}/px.bin`), { headers: { Range: `bytes=${e.o}-${e.o + e.l - 1}` } });
  let buf = await r.arrayBuffer();
  if (r.status === 200) buf = buf.slice(e.o, e.o + e.l);               // host ignored the range: cut it ourselves
  const ds = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  const raw = new Uint8Array(await new Response(ds).arrayBuffer());
  return { raw, e, weeks: PXIDX[k].weeks, scale: PXIDX[k].scale, nodata: PXIDX[k].nodata };
}

function periodMean(pb, weeks, from, to) {
  const n = pb.e.n, out = new Float64Array(n).fill(NaN), cnt = new Float64Array(n), sum = new Float64Array(n);
  weeks.forEach((w, t) => {
    if (w < from || w > to) return;
    for (let i = 0; i < n; i++) { const v = pb.raw[t * n + i]; if (v !== pb.nodata) { sum[i] += v / pb.scale; cnt[i]++; } }
  });
  for (let i = 0; i < n; i++) if (cnt[i]) out[i] = sum[i] / cnt[i];
  return out;
}

const rm = f => `<button class="x" data-rm="${key(f)}" aria-label="Remove field ${f.id}">✕</button>`;

// one saved field as a compact row: crop colour stripe, crop and field, this week's LAI, vs median, alert
function listRow(f, data) {
  const s = data && data.series.fields[f.id], w = data && data.week;
  if (!s) return `<div class="frow"><div><b>Field ${f.id}</b><div class="note">data not available for ${f.area} ${f.year}</div></div><div></div>${rm(f)}</div>`;
  const cs = cropStyle(s.group, data.crop), med = ARABLE.has(s.group) ? groupMedian(data.series.fields, s.group, w, 10) : null;
  const al = s.alert[w] === 1 ? "below the median" : s.alert[w] === 2 ? "falling while others rise" : "";
  const on = CUR && key(CUR) === key(f);
  return `<div class="frow${on ? " on" : ""}" data-k="${key(f)}" style="--c:${cs.colour}">
    <div><b>${cs.label}</b>${data.partial ? ' <span title="only partly inside the study area">⚠</span>' : ""}<div class="note">field ${f.id} · ${f.year} · ${AREA_NAMES[f.area] || f.area}</div></div>
    <div class="fval"><b>${s.lai[w] == null ? "–" : fmtUnc(s.lai[w], s.lai_unc[w], 1)}</b><div class="note">${s.lai[w] == null ? noDataReason(s.src, w) || "no data" : `LAI · ${vsMedianText(s.lai[w], med, s.lai_unc[w]).replace(/ \(.*\)$/, "")} vs median`}</div>
      ${al ? `<div class="pill bad" style="margin-top:2px">${al}</div>` : ""}</div>${rm(f)}</div>`;
}

export async function render() {
  const box = document.getElementById("mf-list");
  if (!box) return;
  if (!ST.fields.length) {
    box.innerHTML = `<p>No fields yet. Your list is kept in this browser only (no account needed), so it starts empty on every new device.</p>
      <ol class="note"><li>Open the <b>Explorer</b> and click one of the four orange squares in NRW (or zoom in with Ctrl + scroll).</li>
      <li>Click a field, then <b>Add to my fields</b>.</li><li>Come back here for its zones, rates and the tractor (ISO-XML) export.</li></ol>
      <p><button id="mf-example">Add the 4 example fields</button> <span class="note">wheat, sugar beet, maize and barley, one per study area, season 2025</span></p>`;
    document.getElementById("mf-example").onclick = () => { EXAMPLES.forEach(f => { if (!ST.fields.some(x => key(x) === key(f))) ST.fields.push(f); }); persist(); render(); };
    document.getElementById("mf-detail").innerHTML = "";
    return;
  }
  const rows = [];
  for (const f of ST.fields) {
    let d = null;
    try { d = await loadSeason(f.area, f.year); } catch (e) { rows.push({ f, d: null, sort: 2 }); continue; }   // removed after a rebuild
    const wk = d.series.weeks, week = FV.active && FV.year === f.year ? FV.week : Math.max(0, wk.findIndex(w => w >= `${f.year}-05-20`));
    const P = d.geo.features.find(x => x.properties.id === f.id)?.properties || {};
    rows.push({ f, d: { series: d.series, week, partial: !!P.partial, crop: P.crop }, sort: d.series.fields[f.id]?.alert[week] ? 0 : 1 });
  }
  const crop = x => (x.d && x.d.series.fields[x.f.id] ? cropStyle(x.d.series.fields[x.f.id].group, x.d.crop).label : "~");
  rows.sort((a, b) => a.sort - b.sort || crop(a).localeCompare(crop(b)) || a.f.id - b.f.id);   // alerts first, then by crop
  box.innerHTML = `<div class="flist">${rows.map(x => listRow(x.f, x.d)).join("")}</div>
    <p class="note">${rows.some(x => x.d) ? `Week of ${rows.find(x => x.d).d.series.weeks[rows.find(x => x.d).d.week]}` : ""}${rows.some(x => x.d?.partial) ? " · ⚠ only partly inside the study area" : ""} · click a field to open it</p>`;
  box.querySelectorAll(".frow[data-k]").forEach(tr => tr.onclick = e => {
    if (e.target.dataset.rm) return;
    const f = ST.fields.find(x => key(x) === tr.dataset.k); openField(f).catch(() => {});
  });
  box.querySelectorAll("button[data-rm]").forEach(b => b.onclick = () => remove(ST.fields.find(x => key(x) === b.dataset.rm)));
  if (!CUR) { const first = rows.find(x => x.d && x.d.series.fields[x.f.id]); if (first) openField(first.f).catch(() => {}); }   // never an empty detail
}

async function openField(f) {
  const d = await loadSeason(f.area, f.year), s = d.series.fields[f.id];
  const geo = d.geo.features.find(x => x.properties.id === f.id);
  if (!s || !geo) { document.getElementById("mf-detail").innerHTML = `<p class="note">Data for field ${f.id} (${f.area} ${f.year}) are not available.</p>`; return; }
  const pb = await pixelBlock(f.area, f.year, f.id);
  const w0 = Math.max(0, d.series.weeks.findIndex(x => x >= (FV.active && FV.year === f.year ? d.series.weeks[FV.week] : `${f.year}-05-20`)));
  CUR = { ...f, pb, s, geo, weeks: d.series.weeks, rgb: d.rgb, week: w0, layer: "lai", tab: CUR?.tab || "overview" };
  render();
  const arable = ARABLE.has(s.group), rates = ST.rates[f.id] || ["", "", ""];
  const det = document.getElementById("mf-detail"), cs = cropStyle(s.group, geo.properties.crop), saved = ST.fields.some(x => key(x) === key(f));
  const src = SOURCE_COLOURS.map((c, i) => i ? `<span style="margin-right:10px;white-space:nowrap"><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c}"></span> ${sourceLabel(i)}</span>` : "").join("");
  det.innerHTML = `
    <div class="mf-head" style="align-items:flex-start">
      <div><div class="crop" style="--c:${cs.colour}"><i></i>${cs.label}</div><h2 class="mf-h" style="margin-top:6px">Field ${f.id}</h2>
        <div class="note">${geo.properties.ha} ha · season ${f.year} · ${AREA_NAMES[f.area] || f.area} · ${geo.properties.crop} (NRW register)</div></div>
      ${saved ? '<span class="note">Saved ✓</span>' : '<button id="mf-save">Save field</button>'}
    </div>
    ${geo.properties.partial ? '<p><span class="pill bad">only partly inside the study area: maps and zones cover the inside part only; exports disabled</span></p>' : ""}
    <div class="tabs" role="tablist">${[["overview", "Overview"], ["charts", "Weekly charts"], ["zones", "Zones & export"]].map(([k, t]) => `<button role="tab" data-tab="${k}" class="${k === CUR.tab ? "on" : ""}">${t}</button>`).join("")}</div>
    <div data-pane="overview">
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
        <span class="seg">${[["lai", "Green LAI"], ["ndvi", "NDVI"], ["zones", "Zones"]].map(([k, t]) => `<button data-layer="${k}" class="${k === "lai" ? "on" : ""}">${t}</button>`).join("")}</span>
        <input type="range" id="mf-week" min="0" max="${d.series.weeks.length - 1}" value="${w0}" style="flex:1 1 160px" aria-label="Week"><b id="mf-week-lbl" class="note"></b>
      </div>
      <canvas id="mf-map" width="360" height="300" style="width:100%;background:var(--bg);border-radius:8px;border:2px solid ${cs.colour}"></canvas>
      <div class="legend" id="mf-legend"></div><div class="note" id="mf-map-cap"></div>
      <div id="mf-kpi" class="kpis" style="margin-top:10px"></div>
    </div>
    <div data-pane="charts">
      <h3>Weekly green LAI</h3><div id="mf-lai-chart"></div>
      <h3>Weekly NDVI</h3><div id="mf-ndvi-chart"></div>
      <p class="note">${src}<br>Band: ± typical error. Shaded weeks: before sowing / after harvest. Dashed: selected week (Overview slider).</p>
    </div>
    <div data-pane="zones">
    ${arable ? `<p><label>From <input type="date" id="mf-from" value="${f.year}-04-01"></label> <label>to <input type="date" id="mf-to" value="${f.year}-06-30"></label></p>
    <p class="note">Zones = thirds of the field by mean green LAI over the period (shown on the Overview map with "Zones"). Rates are yours (kg/ha); the platform supplies the zones.</p>
    <p>${["weak", "mid", "strong"].map((n, i) => `<label style="margin-right:10px;white-space:nowrap"><span style="display:inline-block;width:10px;height:10px;background:${ZCOL[i]};border-radius:2px"></span> ${n} <input type="number" min="0" max="400" step="1" style="width:70px" data-z="${i}" value="${rates[i]}"> kg/ha</label>`).join("")}</p>
    <p><button id="mf-iso" class="pill">Export ISO-XML</button> <button id="mf-gis">Export GIS (GeoJSON)</button> <button id="mf-print">Print report</button></p>
    <p class="note" id="mf-msg"></p>` : `<p class="note">Zones and exports are offered for arable crops only (this field: ${cs.label}).</p>`}
    </div>
    ${attribution()}`;
  const showTab = t => { CUR.tab = t; det.querySelectorAll("[data-pane]").forEach(p => { p.hidden = p.dataset.pane !== t; }); det.querySelectorAll("button[data-tab]").forEach(b => b.classList.toggle("on", b.dataset.tab === t)); };
  det.querySelectorAll("button[data-tab]").forEach(b => b.onclick = () => showTab(b.dataset.tab));
  showTab(CUR.tab);
  const sv = document.getElementById("mf-save"); if (sv) sv.onclick = () => add({ area: f.area, year: f.year, id: f.id });
  det.querySelectorAll("button[data-layer]").forEach(b => b.onclick = () => {
    CUR.layer = b.dataset.layer; det.querySelectorAll("button[data-layer]").forEach(x => x.classList.toggle("on", x === b)); paint();
  });
  document.getElementById("mf-week").oninput = e => { CUR.week = +e.target.value; paint(); };
  paint();
  if (!arable) return;
  ["mf-from", "mf-to"].forEach(i => document.getElementById(i).onchange = paint);
  det.querySelectorAll("input[data-z]").forEach(inp => inp.oninput = () => {
    const r = [...det.querySelectorAll("input[data-z]")].map(x => x.value === "" ? "" : +x.value);
    ST.rates[f.id] = r; persist(); exportsEnabled();
  });
  document.getElementById("mf-iso").onclick = exportIso;
  document.getElementById("mf-gis").onclick = exportGis;
  document.getElementById("mf-print").onclick = () => window.print();
  exportsEnabled();
}

function exportsEnabled() {
  const { ok, msg } = exportCheck({ rates: ST.rates[CUR.id] || [], partial: !!CUR.geo.properties.partial });
  ["mf-iso", "mf-gis"].forEach(i => { const b = document.getElementById(i); if (b) b.disabled = !ok; });
  const m = document.getElementById("mf-msg"); if (m) m.textContent = msg;
}

function currentZones() {
  const from = document.getElementById("mf-from")?.value || `${CUR.year}-04-01`, to = document.getElementById("mf-to")?.value || `${CUR.year}-06-30`;
  const mean = periodMean(CUR.pb, CUR.weeks, from, to);
  return { z: zones(Array.from(mean), CUR.pb.e.w), from, to };
}

// the field's own 10 m pixels: drawn 1:1 on a small canvas, then scaled up as one image (crisp, no seams)
function fieldCanvas(canvas, pb, colourOf) {
  const { r, c } = pb.e, r0 = Math.min(...r), c0 = Math.min(...c), h = Math.max(...r) - r0 + 1, w = Math.max(...c) - c0 + 1;
  const off = document.createElement("canvas"); off.width = w; off.height = h;
  const o = off.getContext("2d");
  for (let i = 0; i < pb.e.n; i++) {
    const col = colourOf(i);
    if (col) { o.fillStyle = col; o.fillRect(c[i] - c0, r[i] - r0, 1, 1); }
  }
  const sc = Math.min(canvas.width / w, canvas.height / h), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, (canvas.width - w * sc) / 2, (canvas.height - h * sc) / 2, w * sc, h * sc);
}

const PNGS = {};
function areaPng(path) {                                  // the area's weekly overlay as pixel data (NDVI per pixel)
  if (!PNGS[path]) PNGS[path] = new Promise((ok, bad) => {
    const im = new Image();
    im.onload = () => {
      const cv = document.createElement("canvas"); cv.width = im.width; cv.height = im.height;
      const x = cv.getContext("2d"); x.drawImage(im, 0, 0); ok(x.getImageData(0, 0, im.width, im.height));
    };
    im.onerror = bad; im.src = url(path);
  });
  return PNGS[path];
}

async function paint() {
  const pb = CUR.pb, w = CUR.week, n = pb.e.n, s = CUR.s, wk = CUR.weeks[w];
  document.getElementById("mf-week-lbl").textContent = `week of ${wk}`;
  const map = document.getElementById("mf-map"), leg = document.getElementById("mf-legend"), cap = document.getElementById("mf-map-cap");
  if (CUR.layer === "zones") {
    const { z } = currentZones();
    fieldCanvas(map, pb, i => (z.classes[i] < 0 ? NOVAL : ZCOL[z.n === 1 ? 1 : z.classes[i]]));
    leg.innerHTML = ["weak", "mid", "strong"].map((t, i) => `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${ZCOL[i]}"></span> ${t}</span>`).join(" ");
    cap.textContent = z.n === 1 ? "One zone (field too small or too uniform)" : `${z.n} zones from the mean green LAI of the chosen period`;
  } else {
    const L = legendSpec(CUR.layer);
    if (CUR.layer === "lai") fieldCanvas(map, pb, i => { const v = pb.raw[w * n + i]; return v === pb.nodata ? NOVAL : d3.interpolateYlGn(0.05 + 0.95 * Math.min(1, v / pb.scale / L.max)); });
    else {
      const img = await areaPng(`${CUR.area}_${CUR.year}/ndvi/${wk}.png`).catch(() => null);
      fieldCanvas(map, pb, i => {
        if (!img) return NOVAL;
        const k = 4 * (pb.e.r[i] * img.width + pb.e.c[i]), a = img.data;
        return a[k + 3] ? `rgb(${a[k]},${a[k + 1]},${a[k + 2]})` : NOVAL;
      });
    }
    const stops = d3.range(0, 1.01, 0.1).map(t => d3.interpolateYlGn(0.05 + 0.95 * t));
    leg.innerHTML = `<span>${L.min}</span><span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span><span>${L.max} ${L.unit}</span>`
      + `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${NOVAL}"></span> no value yet</span>`;
    const sv = CUR.layer === "lai" ? s.src[w] : s.ndvi_src[w];
    cap.textContent = `${CUR.layer === "lai" ? "Green LAI" : "NDVI"} per 10 m pixel, week of ${wk} · ${sourceLabel(sv)}`;
  }
  document.getElementById("mf-kpi").innerHTML = `
    <div class="kpi"><div class="k">Green LAI</div><div class="v">${s.lai[w] == null ? "–" : fmtUnc(s.lai[w], s.lai_unc[w], 1)}</div><div class="s">${s.lai[w] == null ? noDataReason(s.src, w) || "no value" : sourceLabel(s.src[w])}</div></div>
    <div class="kpi"><div class="k">NDVI</div><div class="v">${s.ndvi[w] == null ? "–" : fmtUnc(s.ndvi[w], s.ndvi_unc[w], 2)}</div><div class="s">${s.ndvi[w] == null ? noDataReason(s.ndvi_src, w) || "no value" : sourceLabel(s.ndvi_src[w])}</div></div>
    <div class="kpi"><div class="k">Season</div><div class="v">${s.season[w] ? "in season" : "off-season"}</div><div class="s">${s.season[w] ? "the registered crop" : "previous crop / cover crop / bare soil"}</div></div>`;
  document.getElementById("mf-lai-chart").innerHTML = seriesChart({ values: s.lai, unc: s.lai_unc, src: s.src, season: s.season, week: w, weeks: CUR.weeks, max: 6, label: "green LAI" });
  document.getElementById("mf-ndvi-chart").innerHTML = seriesChart({ values: s.ndvi, unc: s.ndvi_unc, src: s.ndvi_src, season: s.season, week: w, weeks: CUR.weeks, max: 1, label: "NDVI" });
  if (document.getElementById("mf-areamap")) drawAreaMap(cropSummary(AV.geo ? AV.geo.features.map(f => f.properties) : []));   // highlight the open field
}

function lonLatGrid() {
  // ISO grid in lon/lat over the field, ~10 m cells; each cell takes the zone of the 10 m pixel at its centre
  const corners = FV.idx.areas[CUR.area].corners_wgs84;
  const toLL = affineFromCorners(corners[0], corners[1], corners[2], 800, 800), toPx = invertAffine(toLL);
  const { r, c } = CUR.pb.e, { z } = currentZones(), at = new Map();
  r.forEach((rr, i) => at.set(rr * 800 + c[i], z.classes[i]));
  const ll = r.map((rr, i) => applyAffine(toLL, c[i] + 0.5, rr + 0.5));
  const minLon = Math.min(...ll.map(p => p[0])), maxLon = Math.max(...ll.map(p => p[0]));
  const minLat = Math.min(...ll.map(p => p[1])), maxLat = Math.max(...ll.map(p => p[1]));
  const dLat = 10 / 111320, dLon = 10 / (111320 * Math.cos((minLat + maxLat) / 2 * Math.PI / 180));
  const rows = Math.ceil((maxLat - minLat) / dLat) + 1, cols = Math.ceil((maxLon - minLon) / dLon) + 1;
  const classes = new Int8Array(rows * cols).fill(-1);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const [px, py] = applyAffine(toPx, minLon + (j + 0.5) * dLon, minLat + (i + 0.5) * dLat);
    const k = Math.floor(py) * 800 + Math.floor(px);
    if (at.has(k)) classes[i * cols + j] = at.get(k);
  }
  return { grid: { minLat, minLon, dLat, dLon, rows, cols }, classes, n: z.n, toLL };
}

async function exportIso() {
  const g = lonLatGrid(), rates = ST.rates[CUR.id].map(Number);
  const zr = g.n === 1 ? [rates[1], rates[1], rates[1]] : rates;
  const { xml, bin } = buildTaskData({ field: { id: CUR.id, crop: CUR.geo.properties.crop }, zones: { classes: g.classes, n: g.n }, rates: zr, grid: g.grid });
  const zip = new JSZip();
  zip.file("TASKDATA/TASKDATA.XML", xml);
  zip.file("TASKDATA/GRD00001.BIN", bin);
  download(await zip.generateAsync({ type: "blob" }), `field_${CUR.id}_isoxml.zip`);
}

function exportGis() {
  const { r, c, n } = CUR.pb.e, { z, from, to } = currentZones(), rates = ST.rates[CUR.id].map(Number);
  const corners = FV.idx.areas[CUR.area].corners_wgs84, toLL = affineFromCorners(corners[0], corners[1], corners[2], 800, 800);
  const feats = [];
  for (let i = 0; i < n; i++) {
    if (z.classes[i] < 0) continue;
    const ring = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]].map(([dx, dy]) => applyAffine(toLL, c[i] + dx, r[i] + dy).map(v => +v.toFixed(7)));
    const zc = z.n === 1 ? 1 : z.classes[i];
    feats.push({ type: "Feature", properties: { field: CUR.id, zone: ["weak", "mid", "strong"][zc], rate_kg_ha: rates[zc], period: `${from}/${to}` },
                 geometry: { type: "Polygon", coordinates: [ring] } });
  }
  const gj = { type: "FeatureCollection", name: `field_${CUR.id}_zones`, attribution: `Contains modified Copernicus Sentinel data ${CUR.year}; field boundaries Land NRW (dl-de/by-2-0)`, features: feats };
  download(new Blob([JSON.stringify(gj)], { type: "application/geo+json" }), `field_${CUR.id}_zones.geojson`);
}

function download(blob, name) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---- study areas: every registered field of an area, coloured by crop, with crop filter and per-crop curves ----

export async function renderAreas() {
  const box = document.getElementById("mf-areas");
  if (!box || !FV.idx) return;
  const years = FV.idx.areas[AV.area].seasons;
  if (!years.includes(AV.year)) AV.year = years[years.length - 1];
  let d;
  try { d = await loadSeason(AV.area, AV.year); } catch (e) { box.innerHTML = `<p class="note">Field data for ${AV.area} ${AV.year} are not available.</p>`; return; }
  AV.geo = d.geo; AV.series = d.series;
  const props = d.geo.features.map(f => f.properties).filter(p => d.series.fields[p.id]);
  const all = cropSummary(props), total = all.reduce((a, c) => a + c.ha, 0);
  // main crops (>= 1 % of the area, at most 12) get their own legend row; the long tail is grouped as "Other crops"
  const main = all.filter((c, i) => i < 12 && c.ha >= 0.01 * total), rest = all.filter(c => !main.includes(c));
  AV.other = new Set(rest.map(c => c.label));
  const rows = main.concat(rest.length ? [{ label: "Other crops", colour: "#7b8496", n: rest.reduce((a, c) => a + c.n, 0), ha: rest.reduce((a, c) => a + c.ha, 0) }] : []);
  const w = Math.max(0, d.series.weeks.findIndex(x => x >= `${AV.year}-05-20`));
  const arable = props.filter(p => ARABLE.has(p.group)).map(p => d.series.fields[p.id].lai[w]).filter(v => v != null).sort((a, b) => a - b);
  const medLai = arable.length ? arable[Math.floor(arable.length / 2)] : null;
  box.innerHTML = `
    <div class="mf-head">
      <div><h2 class="mf-h">Study area ${AREA_NAMES[AV.area] || AV.area}</h2><div class="note">North Rhine-Westphalia · 8 × 8 km · weekly 10 m green LAI and NDVI</div></div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span class="seg">${Object.keys(FV.idx.areas).map(a => `<button data-area="${a}" class="${a === AV.area ? "on" : ""}">${AREA_NAMES[a] || a}</button>`).join("")}</span>
        <select id="mf-year" aria-label="Season">${years.map(y => `<option ${y === AV.year ? "selected" : ""}>${y}</option>`).join("")}</select>
      </div>
    </div>
    <div class="mf-stats">
      <div class="stat"><div class="k">Fields</div><div class="v">${props.length.toLocaleString("en")}</div></div>
      <div class="stat"><div class="k">Area</div><div class="v">${Math.round(total).toLocaleString("en")} ha</div></div>
      <div class="stat"><div class="k">Crops</div><div class="v">${all.length}</div><div class="note">${main.length} main</div></div>
      <div class="stat"><div class="k">Median LAI, arable</div><div class="v">${medLai == null ? "–" : medLai.toFixed(1)}</div><div class="note">week of ${d.series.weeks[w]}</div></div>
    </div>
    <div class="seg" style="margin-bottom:10px">${[["crops", "Crops"], ["lai", "Green LAI"], ["ndvi", "NDVI"]].map(([k, t]) => `<button data-view="${k}" class="${k === AV.view ? "on" : ""}">${t}</button>`).join("")}</div>
    <div class="mf-mapgrid">
      <div>
        <svg id="mf-areamap" viewBox="0 0 520 520" role="img" aria-label="Fields of the study area coloured by crop" ${AV.view === "crops" ? "" : "hidden"}></svg>
        <div id="mf-animbox" ${AV.view === "crops" ? "hidden" : ""}>
          <canvas id="mf-anim" width="800" height="800" style="width:100%;display:block;background:var(--bg);border-radius:8px;border:1px solid var(--line)" role="img" aria-label="Weekly map of the whole study area"></canvas>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">
            <button id="mf-play" aria-label="Play">▶ Play</button>
            <select id="mf-step" aria-label="Animation step"><option value="month" ${AV.step === "month" ? "selected" : ""}>Monthly</option><option value="week" ${AV.step === "week" ? "selected" : ""}>Weekly</option></select>
            <input type="range" id="mf-frame" min="0" value="0" style="flex:1 1 160px" aria-label="Date">
            <b id="mf-frame-lbl"></b>
          </div>
          <div class="legend" id="mf-anim-legend" style="margin-top:6px"></div>
        </div>
      </div>
      <aside class="legendlist" aria-label="Crops">
        <div class="k" style="margin-bottom:4px">Crops · click to filter</div>
        ${[{ label: "All crops", colour: "#8b96b4", n: props.length, ha: total }].concat(rows).map(c => `
          <button data-crop="${c.label}" class="${(AV.crop || "All crops") === c.label ? "on" : ""}" style="--c:${c.colour}">
            <i></i><span>${c.label}</span><span class="note">${Math.round(c.ha)} ha</span>
            <span class="bar"><span style="width:${(100 * c.ha / total).toFixed(1)}%"></span></span>
          </button>`).join("")}
      </aside>
    </div>
    <h3>Median green LAI per crop</h3><div id="mf-areachart"></div>`;
  box.querySelectorAll("button[data-area]").forEach(b => b.onclick = () => { stopAnim(); AV.area = b.dataset.area; AV.crop = null; renderAreas(); });
  box.querySelectorAll("button[data-view]").forEach(b => b.onclick = () => { stopAnim(); AV.view = b.dataset.view; AV.frame = null; renderAreas(); });
  document.getElementById("mf-year").onchange = e => { stopAnim(); AV.year = +e.target.value; renderAreas(); };
  box.querySelectorAll("button[data-crop]").forEach(b => b.onclick = () => { AV.crop = b.dataset.crop === "All crops" ? null : b.dataset.crop; renderAreas(); });
  drawAreaMap(all);
  drawAreaChart(props, all);
  if (AV.view !== "crops") initAnim();
}

// ---- whole-area animation of the weekly 10 m maps (monthly or weekly steps) ----
function animFrames() {
  const wk = AV.series.weeks;
  return AV.step === "month" ? monthlyFrames(wk) : wk.map((_, i) => i);
}

function stopAnim() {
  if (AV.timer) { clearInterval(AV.timer); AV.timer = null; }
  const b = document.getElementById("mf-play"); if (b) b.textContent = "▶ Play";
}

const FRAMES = {};
function frameImage(path) {
  if (!FRAMES[path]) FRAMES[path] = new Promise(ok => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => ok(null); im.src = url(path); });
  return FRAMES[path];
}

async function drawFrame() {
  const fr = animFrames(), k = Math.min(AV.frame ?? 0, fr.length - 1), w = fr[k], wk = AV.series.weeks[w];
  document.getElementById("mf-frame").value = k;
  const d = new Date(wk + "T00:00:00Z");
  document.getElementById("mf-frame-lbl").textContent = AV.step === "month"
    ? d.toLocaleDateString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" }) : `week of ${d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })}`;
  const im = await frameImage(`${AV.area}_${AV.year}/${AV.view}/${wk}.png`), c = document.getElementById("mf-anim");
  if (!c) return;
  const ctx = c.getContext("2d");
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--bg") || "#0a0e1a"; ctx.fillRect(0, 0, c.width, c.height);
  if (im) { ctx.imageSmoothingEnabled = false; ctx.drawImage(im, 0, 0, c.width, c.height); }
  fr.slice(k + 1, k + 3).forEach(i => frameImage(`${AV.area}_${AV.year}/${AV.view}/${AV.series.weeks[i]}.png`));   // preload the next frames
}

function initAnim() {
  const fr = animFrames(), sl = document.getElementById("mf-frame");
  if (AV.frame == null) AV.frame = Math.max(0, fr.findIndex(i => AV.series.weeks[i] >= `${AV.year}-05-01`));
  sl.max = fr.length - 1;
  const L = legendSpec(AV.view), stops = d3.range(0, 1.01, 0.1).map(t => d3.interpolateYlGn(0.05 + 0.95 * t));
  document.getElementById("mf-anim-legend").innerHTML = `<span>${L.min}</span><span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span><span>${L.max} ${L.unit}</span>`
    + `<span class="note">dark: not a field or no value yet · weekly re-analysis, ${AV.year} season</span>`;
  sl.oninput = e => { stopAnim(); AV.frame = +e.target.value; drawFrame(); };
  document.getElementById("mf-step").onchange = e => {
    const cur = animFrames()[AV.frame]; stopAnim(); AV.step = e.target.value;
    const f2 = animFrames(); AV.frame = Math.max(0, f2.findIndex(i => i >= cur)); sl.max = f2.length - 1; drawFrame();
  };
  document.getElementById("mf-play").onclick = () => {
    if (AV.timer) { stopAnim(); return; }
    document.getElementById("mf-play").textContent = "⏸ Pause";
    AV.timer = setInterval(() => {
      if (!document.getElementById("mf-anim")) { stopAnim(); return; }
      AV.frame = (AV.frame + 1) % animFrames().length; drawFrame();
    }, AV.step === "month" ? 900 : 350);
  };
  drawFrame();
}

function chipMatch(label) {
  return !AV.crop || AV.crop === label || (AV.crop === "Other crops" && AV.other && AV.other.has(label));
}

function drawAreaMap(crops) {
  const svg = d3.select("#mf-areamap"), proj = d3.geoMercator().fitSize([520, 520], AV.geo), path = d3.geoPath(proj);
  const col = Object.fromEntries(crops.map(c => [c.label, c.colour]));
  const tip = document.getElementById("mf-tip");
  svg.selectAll("path").data(AV.geo.features.filter(f => AV.series.fields[f.properties.id]), f => f.properties.id).join("path")
    .attr("d", path)
    .attr("fill", f => col[cropStyle(f.properties.group, f.properties.crop).label] || "#555")
    .attr("fill-opacity", f => (chipMatch(cropStyle(f.properties.group, f.properties.crop).label) ? 0.95 : 0.12))
    .attr("stroke", f => (CUR && CUR.area === AV.area && CUR.year === AV.year && CUR.id === f.properties.id ? "#fff" : "#0b1020"))
    .attr("stroke-width", f => (CUR && CUR.id === f.properties.id ? 2 : 0.4))
    .style("cursor", "pointer")
    .on("mousemove", (e, f) => {
      const P = f.properties, s = AV.series.fields[P.id], w = Math.max(0, AV.series.weeks.findIndex(x => x >= `${AV.year}-05-20`));
      tip.style.display = "block";
      const b = document.getElementById("mf-areas").getBoundingClientRect();
      tip.style.left = (e.clientX - b.left + 14) + "px"; tip.style.top = (e.clientY - b.top + 14) + "px";
      tip.innerHTML = `<b>${cropStyle(P.group, P.crop).label}</b> · field ${P.id}<br>${P.ha} ha${P.partial ? " · partly outside the area" : ""}<br>green LAI ${s.lai[w] == null ? "–" : fmtUnc(s.lai[w], s.lai_unc[w], 1)} (week of ${AV.series.weeks[w]})`;
    })
    .on("mouseleave", () => { tip.style.display = "none"; })
    .on("click", (e, f) => { tip.style.display = "none"; openField({ area: AV.area, year: AV.year, id: f.properties.id }).catch(() => {}); });
}

function drawAreaChart(props, crops) {
  const med = cropMedians(props, AV.series.fields, 10), weeks = AV.series.weeks, n = weeks.length;
  const W = 520, H = 190, L = 30, R = 8, T = 8, B = 22, x = i => L + (i * (W - L - R)) / (n - 1), y = v => H - B - (Math.min(6, v) / 6) * (H - B - T);
  const col = Object.fromEntries(crops.map(c => [c.label, c.colour]));
  const lines = Object.entries(med).map(([k, v]) => {
    const on = chipMatch(k);
    let d = "", pen = false;
    v.forEach((val, i) => { if (val == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(val).toFixed(1)}`; pen = true; });
    return `<path d="${d}" fill="none" stroke="${col[k]}" stroke-width="${on ? 2.2 : 1}" stroke-opacity="${on ? 1 : 0.25}"><title>${k}</title></path>`;
  }).join("");
  const months = weeks.map((w, i) => [w, i]).filter(([w]) => +w.slice(8, 10) <= 7 && ["11", "01", "03", "05", "07"].includes(w.slice(5, 7)))
    .map(([w, i]) => `<text x="${x(i)}" y="${H - 5}" text-anchor="middle">${["", "Jan", "", "Mar", "", "May", "", "Jul", "", "", "", "Nov"][+w.slice(5, 7)]}</text>`).join("");
  const ticks = [0, 2, 4, 6].map(t => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#8b96b4" stroke-opacity=".2"/><text x="${L - 4}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
  document.getElementById("mf-areachart").innerHTML = Object.keys(med).length
    ? `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Median green LAI per crop through the season">${ticks}${months}${lines}</svg>
       <p class="note">Median of all fields of each crop with at least 10 fields; values are weekly green LAI (m²/m²).</p>`
    : `<p class="note">No crop has at least 10 fields with values in this season.</p>`;
}

window.MYFIELDS = { add, render, renderAreas };

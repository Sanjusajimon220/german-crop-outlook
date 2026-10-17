// MY FIELDS: fields picked on the map (kept in this browser, no accounts), list + map, zones, rates, exports.
import { loadMyFields, saveMyFields, zones, fmtUnc, sourceLabel, withinNoise, nearestRgbDate,
         affineFromCorners, invertAffine, applyAffine } from "./field_core.js";
import { buildTaskData } from "./isoxml.js";
import { FV, loadSeason, attribution } from "./field_view.js";

const BASE = "field/";
const ARABLE = new Set(["wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"]);
const ZCOL = ["#d9534f", "#f0ad4e", "#2e9e5b"];
let ST = loadMyFields(safeStorage());
let CUR = null;                       // {area, year, id, px: Uint8Array [weeks x n], idx}
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
  if (!PXIDX[k]) PXIDX[k] = await fetch(`${BASE}${k}/px.json`).then(r => r.json());
  const e = PXIDX[k].fields[String(id)];
  const r = await fetch(`${BASE}${k}/px.bin`, { headers: { Range: `bytes=${e.o}-${e.o + e.l - 1}` } });
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

function drawPixels(canvas, pb, colourOf) {
  const { r, c } = pb.e, r0 = Math.min(...r), c0 = Math.min(...c), h = Math.max(...r) - r0 + 1, w = Math.max(...c) - c0 + 1;
  const sc = Math.max(1, Math.floor(Math.min(canvas.width / w, canvas.height / h)));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < pb.e.n; i++) {
    const col = colourOf(i);
    if (!col) continue;
    ctx.fillStyle = col; ctx.fillRect((c[i] - c0) * sc, (r[i] - r0) * sc, sc, sc);
  }
}

function listRow(f, data) {
  const s = data.series.fields[f.id], w = data.week;
  if (!s) return `<tr><td>${f.id}</td><td colspan="7">no data for ${f.year}</td></tr>`;
  const dv = w > 0 && s.lai[w] != null && s.lai[w - 1] != null ? s.lai[w] - s.lai[w - 1] : null;
  const ch = dv == null ? "–" : withinNoise(dv, s.lai_unc[w] ?? 0) ? "within noise" : (dv > 0 ? "▲ " : "▼ ") + Math.abs(dv).toFixed(1);
  const al = s.alert[w] === 1 ? "below median" : s.alert[w] === 2 ? "falling" : "";
  return `<tr data-k="${key(f)}" ${CUR && key(CUR) === key(f) ? 'style="background:var(--accent-soft)"' : ""}>
    <td>${f.id}</td><td>${s.group}</td><td>${f.year}</td><td>${fmtUnc(s.lai[w], s.lai_unc[w], 1)}</td><td>${ch}</td>
    <td>${sourceLabel(s.src[w])}</td><td class="lose">${al}</td><td><button data-rm="${key(f)}" aria-label="Remove field ${f.id}">✕</button></td></tr>`;
}

export async function render() {
  const box = document.getElementById("mf-list");
  if (!box) return;
  if (!ST.fields.length) {
    box.innerHTML = `<p class="note">No fields yet. In the Explorer, zoom into an NRW study area, click a field and press <b>Add to my fields</b>. Your choice stays in this browser only.</p>`;
    document.getElementById("mf-detail").innerHTML = "";
    return;
  }
  const rows = [];
  for (const f of ST.fields) {
    const d = await loadSeason(f.area, f.year);
    const wk = d.series.weeks, week = FV.active && FV.year === f.year ? FV.week : Math.max(0, wk.findIndex(w => w >= `${f.year}-05-20`));
    rows.push({ f, d: { series: d.series, week }, sort: d.series.fields[f.id]?.alert[week] ? 0 : 1 });
  }
  rows.sort((a, b) => a.sort - b.sort || a.f.id - b.f.id);
  box.innerHTML = `<div class="tbl"><table><thead><tr><th>Field</th><th>Crop</th><th>Season</th><th>Green LAI</th><th>Δ week</th><th>Source</th><th>Alert</th><th></th></tr></thead>
    <tbody>${rows.map(x => listRow(x.f, x.d)).join("")}</tbody></table></div><p class="note">Week shown: ${rows[0].d.series.weeks[rows[0].d.week]} (change it with the date slider in the Explorer). Click a row to open the field.</p>`;
  box.querySelectorAll("tr[data-k]").forEach(tr => tr.onclick = e => {
    if (e.target.dataset.rm) return;
    const f = ST.fields.find(x => key(x) === tr.dataset.k); openField(f);
  });
  box.querySelectorAll("button[data-rm]").forEach(b => b.onclick = () => remove(ST.fields.find(x => key(x) === b.dataset.rm)));
}

async function openField(f) {
  const d = await loadSeason(f.area, f.year), s = d.series.fields[f.id];
  const geo = d.geo.features.find(x => x.properties.id === f.id);
  const pb = await pixelBlock(f.area, f.year, f.id);
  CUR = { ...f, pb, s, geo, weeks: d.series.weeks, rgb: d.rgb };
  render();
  const arable = ARABLE.has(s.group), rates = ST.rates[f.id] || ["", "", ""];
  const det = document.getElementById("mf-detail");
  det.innerHTML = `<h2>Field ${f.id} · ${geo.properties.crop}</h2><div class="note">${s.group} · ${geo.properties.ha} ha · season ${f.year}</div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:10px">
      <div><canvas id="mf-rgb" width="220" height="220" style="width:100%"></canvas><div class="note" id="mf-rgb-cap"></div></div>
      <div><canvas id="mf-lai" width="220" height="220" style="width:100%"></canvas><div class="note" id="mf-lai-cap"></div></div>
      <div><canvas id="mf-zones" width="220" height="220" style="width:100%"></canvas><div class="note" id="mf-z-cap"></div></div>
    </div>
    ${arable ? `<h3>Zones for fertiliser</h3>
    <p><label>From <input type="date" id="mf-from" value="${f.year}-04-01"></label> <label>to <input type="date" id="mf-to" value="${f.year}-06-30"></label></p>
    <p class="note">Zones = thirds of the field by mean green LAI over the period. Rates are yours (kg/ha); the platform supplies the zones.</p>
    <p>${["weak", "mid", "strong"].map((n, i) => `<label style="margin-right:10px"><span style="display:inline-block;width:10px;height:10px;background:${ZCOL[i]};border-radius:2px"></span> ${n} <input type="number" min="0" max="400" step="1" style="width:70px" data-z="${i}" value="${rates[i]}"> kg/ha</label>`).join("")}</p>
    <p><button id="mf-iso" class="pill">Export ISO-XML</button> <button id="mf-gis">Export GIS (GeoJSON)</button> <button id="mf-print">Print report</button></p>
    <p class="note" id="mf-msg"></p>` : `<p class="note">Zones and exports are offered for arable crops only (this field: ${s.group}).</p>`}
    ${attribution()}`;
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
  const r = ST.rates[CUR.id] || [], ok = r.length === 3 && r.every(v => v !== "" && isFinite(v));
  ["mf-iso", "mf-gis"].forEach(i => { const b = document.getElementById(i); if (b) b.disabled = !ok; });
  const m = document.getElementById("mf-msg"); if (m) m.textContent = ok ? "" : "Set a rate for every zone to enable the exports.";
}

function currentZones() {
  const from = document.getElementById("mf-from")?.value || `${CUR.year}-04-01`, to = document.getElementById("mf-to")?.value || `${CUR.year}-06-30`;
  const mean = periodMean(CUR.pb, CUR.weeks, from, to);
  return { z: zones(Array.from(mean), CUR.pb.e.w), from, to };
}

function paint() {
  const pb = CUR.pb, w = Math.max(0, CUR.weeks.findIndex(x => x >= (FV.active && FV.year === CUR.year ? CUR.weeks[FV.week] : `${CUR.year}-05-20`)));
  const n = pb.e.n;
  drawPixels(document.getElementById("mf-lai"), pb, i => { const v = pb.raw[w * n + i]; return v === pb.nodata ? null : d3.interpolateYlGn(Math.min(1, v / pb.scale / 5)); });
  document.getElementById("mf-lai-cap").textContent = `Green LAI 10 m, week of ${CUR.weeks[w]}`;
  const { z } = currentZones();
  drawPixels(document.getElementById("mf-zones"), pb, i => (z.classes[i] < 0 ? null : ZCOL[z.n === 1 ? 1 : z.classes[i]]));
  document.getElementById("mf-z-cap").textContent = z.n === 1 ? "One zone (field too small or too uniform)" : `${z.n} zones`;
  const d = nearestRgbDate(CUR.rgb, CUR.weeks[w]) || CUR.rgb[Math.floor(CUR.rgb.length / 2)];
  const img = new Image(), c = document.getElementById("mf-rgb"), { r, c: cc } = pb.e;
  img.onload = () => {
    const r0 = Math.min(...r), c0 = Math.min(...cc), h = Math.max(...r) - r0 + 1, ww = Math.max(...cc) - c0 + 1;
    const sc = Math.max(1, Math.floor(Math.min(c.width / ww, c.height / h))), ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(img, c0, r0, ww, h, 0, 0, ww * sc, h * sc);
  };
  img.src = `${BASE}${CUR.area}_${CUR.year}/rgb/${d}.jpg`;
  document.getElementById("mf-rgb-cap").textContent = `Sentinel-2, ${d}`;
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

window.MYFIELDS = { add, render };

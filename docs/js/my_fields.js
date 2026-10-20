// MY FIELDS: fields picked on the map (kept in this browser, no accounts), list + map, zones, rates, exports.
import { loadMyFields, saveMyFields, zones, fmtUnc, sourceLabel, withinNoise, nearestRgbDate, clearRgbDate,
         affineFromCorners, invertAffine, applyAffine, groupMedian, exportCheck, noDataReason, fieldRgbDate,
         seriesChart, SOURCE_COLOURS, legendSpec, cropStyle, vsMedianText } from "./field_core.js";
const badge = (g, c) => { const k = cropStyle(g, c); return `<span class="crop" style="--c:${k.colour}"><i></i>${k.label}</span>`; };

import { buildTaskData } from "./isoxml.js";
import { FV, loadSeason, attribution, url } from "./field_view.js";

const ARABLE = new Set(["wheat", "barley", "other cereal", "rapeseed", "maize", "potato", "sugar beet"]);
const ZCOL = ["#d9534f", "#f0ad4e", "#2e9e5b"];
// example fields shown on a first visit (one per study area, season 2025); users can remove them or add their own
export const EXAMPLES = [{ area: "pilot", year: 2025, id: 8702226 }, { area: "selhausen", year: 2025, id: 8312694 },
                         { area: "north", year: 2025, id: 8910847 }, { area: "south", year: 2025, id: 8965510 }];
let ST = loadMyFields(safeStorage(), EXAMPLES);
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

const rm = f => `<button data-rm="${key(f)}" aria-label="Remove field ${f.id}">✕</button>`;

function listRow(f, data) {
  const s = data && data.series.fields[f.id], w = data && data.week;
  if (!s) return `<tr><td>${f.id}</td><td colspan="6">data not available for ${f.area} ${f.year}</td><td>${rm(f)}</td></tr>`;
  const med = ARABLE.has(s.group) ? groupMedian(data.series.fields, s.group, w, 10) : null, dm = s.lai[w] != null && med ? s.lai[w] - med : null;
  const vm = vsMedianText(s.lai[w], med, s.lai_unc[w]);
  const dv = w > 0 && s.lai[w] != null && s.lai[w - 1] != null ? s.lai[w] - s.lai[w - 1] : null;
  const ch = dv == null ? "–" : withinNoise(dv, s.lai_unc[w] ?? 0) ? "within noise" : (dv > 0 ? "▲ " : "▼ ") + Math.abs(dv).toFixed(1);
  const al = s.alert[w] === 1 ? "below median" : s.alert[w] === 2 ? "falling" : "";
  return `<tr data-k="${key(f)}" ${CUR && key(CUR) === key(f) ? 'style="background:var(--accent-soft)"' : ""}>
    <td>${f.id}${data.partial ? " ⚠" : ""}</td><td>${badge(s.group, data.crop)} <span class="note">${f.year}</span></td><td>${s.lai[w] == null ? noDataReason(s.src, w) || "no data" : fmtUnc(s.lai[w], s.lai_unc[w], 1)}</td>
    <td>${vm}</td><td>${ch}</td><td>${sourceLabel(s.src[w])}${s.season[w] ? "" : " · off-season"}</td><td class="lose">${al}</td><td>${rm(f)}</td></tr>`;
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
  box.innerHTML = `<div class="tbl"><table><thead><tr><th>Field</th><th>Crop · season</th><th>Green LAI</th><th>vs median</th><th>Δ week</th><th>Source</th><th>Alert</th><th></th></tr></thead>
    <tbody>${rows.map(x => listRow(x.f, x.d)).join("")}</tbody></table></div><p class="note">${rows.some(x => x.d) ? `Week shown: ${rows.find(x => x.d).d.series.weeks[rows.find(x => x.d).d.week]}` : ""}${rows.some(x => x.d?.partial) ? " · ⚠ = only partly inside the study area" : ""} (change it with the date slider in the Explorer). Click a row to open the field.</p>`;
  box.querySelectorAll("tr[data-k]").forEach(tr => tr.onclick = e => {
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
  CUR = { ...f, pb, s, geo, weeks: d.series.weeks, rgb: d.rgb, week: w0, layer: "lai" };
  render();
  const arable = ARABLE.has(s.group), rates = ST.rates[f.id] || ["", "", ""];
  const det = document.getElementById("mf-detail");
  const src = SOURCE_COLOURS.map((c, i) => i ? `<span style="margin-right:10px;white-space:nowrap"><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c}"></span> ${sourceLabel(i)}</span>` : "").join("");
  const cs = cropStyle(s.group, geo.properties.crop);
  det.innerHTML = `<h2>${badge(s.group, geo.properties.crop)} Field ${f.id}</h2><div class="note">${geo.properties.crop} (NRW register) · ${geo.properties.ha} ha · season ${f.year} · study area ${f.area}</div>
    ${geo.properties.partial ? '<p><span class="pill bad">only partly inside the study area: maps and zones cover the inside part only; exports disabled</span></p>' : ""}
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0">
      <span class="seg">${[["lai", "Green LAI"], ["ndvi", "NDVI"], ["zones", "Zones"]].map(([k, t]) => `<button data-layer="${k}" class="${k === "lai" ? "on" : ""}">${t}</button>`).join("")}</span>
      <input type="range" id="mf-week" min="0" max="${d.series.weeks.length - 1}" value="${w0}" style="flex:1 1 200px" aria-label="Week"><b id="mf-week-lbl"></b>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
      <div><canvas id="mf-map" width="300" height="300" style="width:100%;max-width:300px;background:var(--bg);border-radius:8px;border:2px solid ${cs.colour}"></canvas>
        <div class="legend" id="mf-legend"></div><div class="note" id="mf-map-cap"></div></div>
      <div id="mf-rgb-box"><canvas id="mf-rgb" width="300" height="300" style="width:100%;max-width:300px;border-radius:8px"></canvas><div class="note" id="mf-rgb-cap"></div></div>
    </div>
    <div id="mf-kpi" class="kpis" style="margin-top:10px"></div>
    <h3>Weekly green LAI</h3><div id="mf-lai-chart"></div>
    <h3>Weekly NDVI</h3><div id="mf-ndvi-chart"></div>
    <p class="note">${src}<br>Band: ± typical error of the value. Shaded weeks: before sowing / after harvest (previous crop, cover crop or bare soil). Dashed: selected week.</p>
    ${arable ? `<h3>Zones for fertiliser</h3>
    <p><label>From <input type="date" id="mf-from" value="${f.year}-04-01"></label> <label>to <input type="date" id="mf-to" value="${f.year}-06-30"></label></p>
    <p class="note">Zones = thirds of the field by mean green LAI over the period. Rates are yours (kg/ha); the platform supplies the zones.</p>
    <p>${["weak", "mid", "strong"].map((n, i) => `<label style="margin-right:10px"><span style="display:inline-block;width:10px;height:10px;background:${ZCOL[i]};border-radius:2px"></span> ${n} <input type="number" min="0" max="400" step="1" style="width:70px" data-z="${i}" value="${rates[i]}"> kg/ha</label>`).join("")}</p>
    <p><button id="mf-iso" class="pill">Export ISO-XML</button> <button id="mf-gis">Export GIS (GeoJSON)</button> <button id="mf-print">Print report</button></p>
    <p class="note" id="mf-msg"></p>` : `<p class="note">Zones and exports are offered for arable crops only (this field: ${s.group}).</p>`}
    ${attribution()}`;
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
    fieldCanvas(map, pb, i => (z.classes[i] < 0 ? null : ZCOL[z.n === 1 ? 1 : z.classes[i]]));
    leg.innerHTML = ["weak", "mid", "strong"].map((t, i) => `<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${ZCOL[i]}"></span> ${t}</span>`).join(" ");
    cap.textContent = z.n === 1 ? "One zone (field too small or too uniform)" : `${z.n} zones from the mean green LAI of the chosen period`;
  } else {
    const L = legendSpec(CUR.layer);
    if (CUR.layer === "lai") fieldCanvas(map, pb, i => { const v = pb.raw[w * n + i]; return v === pb.nodata ? null : d3.interpolateYlGn(0.05 + 0.95 * Math.min(1, v / pb.scale / L.max)); });
    else {
      const img = await areaPng(`${CUR.area}_${CUR.year}/ndvi/${wk}.png`).catch(() => null);
      fieldCanvas(map, pb, i => {
        if (!img) return null;
        const k = 4 * (pb.e.r[i] * img.width + pb.e.c[i]), a = img.data;
        return a[k + 3] ? `rgb(${a[k]},${a[k + 1]},${a[k + 2]})` : null;
      });
    }
    const stops = d3.range(0, 1.01, 0.1).map(t => d3.interpolateYlGn(0.05 + 0.95 * t));
    leg.innerHTML = `<span>${L.min}</span><span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span><span>${L.max} ${L.unit}</span>`;
    const sv = CUR.layer === "lai" ? s.src[w] : s.ndvi_src[w];
    cap.textContent = `${CUR.layer === "lai" ? "Green LAI" : "NDVI"} per 10 m pixel, week of ${wk} · ${sourceLabel(sv)}`;
  }
  document.getElementById("mf-kpi").innerHTML = `
    <div class="kpi"><div class="k">Green LAI</div><div class="v">${s.lai[w] == null ? "–" : fmtUnc(s.lai[w], s.lai_unc[w], 1)}</div><div class="s">${s.lai[w] == null ? noDataReason(s.src, w) || "no value" : sourceLabel(s.src[w])}</div></div>
    <div class="kpi"><div class="k">NDVI</div><div class="v">${s.ndvi[w] == null ? "–" : fmtUnc(s.ndvi[w], s.ndvi_unc[w], 2)}</div><div class="s">${s.ndvi[w] == null ? noDataReason(s.ndvi_src, w) || "no value" : sourceLabel(s.ndvi_src[w])}</div></div>
    <div class="kpi"><div class="k">Season</div><div class="v">${s.season[w] ? "in season" : "off-season"}</div><div class="s">${s.season[w] ? "the registered crop" : "previous crop / cover crop / bare soil"}</div></div>`;
  document.getElementById("mf-lai-chart").innerHTML = seriesChart({ values: s.lai, unc: s.lai_unc, src: s.src, season: s.season, week: w, weeks: CUR.weeks, max: 6, label: "green LAI" });
  document.getElementById("mf-ndvi-chart").innerHTML = seriesChart({ values: s.ndvi, unc: s.ndvi_unc, src: s.ndvi_src, season: s.season, week: w, weeks: CUR.weeks, max: 1, label: "NDVI" });
  // Sentinel-2 true colour only when >= 95 % of this field was clear within 10 days of the week; otherwise not shown
  const box = document.getElementById("mf-rgb-box"), d = fieldRgbDate(CUR.rgb, s.clr, wk);
  box.hidden = !d;
  if (!d) return;
  const img = new Image(), c = document.getElementById("mf-rgb"), { r, c: cc } = pb.e;
  img.onload = () => {
    const r0 = Math.min(...r), c0 = Math.min(...cc), h = Math.max(...r) - r0 + 1, ww = Math.max(...cc) - c0 + 1, pad = 3;
    const sx = Math.max(0, c0 - pad), sy = Math.max(0, r0 - pad), sw = Math.min(800 - sx, ww + 2 * pad), sh = Math.min(800 - sy, h + 2 * pad);
    const sc = Math.min(c.width / sw, c.height / sh), ox = (c.width - sw * sc) / 2, oy = (c.height - sh * sc) / 2, ctx = c.getContext("2d");
    ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(img, sx, sy, sw, sh, ox, oy, sw * sc, sh * sc);
    const inF = new Set(r.map((rr, i) => rr * 1000 + cc[i]));               // mark the field edge with small dots
    ctx.fillStyle = "rgba(242,169,59,.95)";
    for (let i = 0; i < r.length; i++) {
      const edge = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([a, b]) => !inF.has((r[i] + a) * 1000 + cc[i] + b));
      if (edge) ctx.fillRect(ox + (cc[i] - sx) * sc + sc * 0.35, oy + (r[i] - sy) * sc + sc * 0.35, sc * 0.3, sc * 0.3);
    }
  };
  img.src = url(`${CUR.area}_${CUR.year}/rgb/${d}.jpg`);
  document.getElementById("mf-rgb-cap").textContent = `Sentinel-2 true colour, ${d} (≥ 95 % of the field clear)`;
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

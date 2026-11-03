// Field level part of the Accuracy page: the logged, one-look benchmark results of the weekly 10 m series
// (data: field/benchmark.json, written by 06_field/fv2_benchmark_export.py). Pure string builders, tested in node.

const f1 = v => +v.toFixed(1), pct = v => `${(v * 100).toFixed(1)} %`, n2 = v => (v == null ? "–" : v.toFixed(2));
const SEASON_COLOURS = { 2021: "#2a78d6", 2022: "#1baf7a", 2023: "#eb6834", 2024: "#6250d6" };
const EVENT_COLOURS = { start: "#2a78d6", peak: "#1baf7a", end: "#e34948" };
const CROPS = ["wheat", "barley", "rapeseed", "maize", "potato", "sugar beet"];

// item 2: one dot per Landsat date; above the diagonal = our series ranks the fields closer to Landsat
export function item2Scatter(it) {
  const W = 340, H = 300, L = 44, R = 12, T = 12, B = 40, lo = 0.2;
  const x = v => f1(L + ((v - lo) / (1 - lo)) * (W - L - R)), y = v => f1(H - B - ((v - lo) / (1 - lo)) * (H - B - T));
  const grid = [0.2, 0.4, 0.6, 0.8, 1].map(t => `<line x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - B}" class="grid"/><line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" class="grid"/>`
    + `<text x="${x(t)}" y="${H - B + 14}" text-anchor="middle">${t}</text><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
  const dots = it.points.map(p => `<circle cx="${x(Math.max(lo, p.ppi))}" cy="${y(Math.max(lo, p.ours))}" r="${f1(Math.min(7, 2 + Math.sqrt(p.n) / 4))}" fill="${SEASON_COLOURS[p.season] || "#8b96b4"}" fill-opacity=".75"><title>${p.season}: ours ${n2(p.ours)}, HR-VPP ${n2(p.ppi)} (${p.n} fields)</title></circle>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Cloudy weeks: ours better on ${it.won} of ${it.of} Landsat dates">${grid}
    <line x1="${x(lo)}" y1="${y(lo)}" x2="${x(1)}" y2="${y(1)}" class="diag"/>${dots}
    <text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">HR-VPP vs Landsat (rank agreement)</text>
    <text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">ours vs Landsat</text></svg>
    <p class="note">${Object.entries(SEASON_COLOURS).map(([s, c]) => `<span style="color:${c}">●</span> ${s}`).join(" ")} · dot size = number of fields · ${it.won} of ${it.of} dates above the line</p>`;
}

// item 1: rank agreement with HR-VPP on clear days, crop x season
export function item1Table(rows) {
  const seasons = [...new Set(rows.map(r => r.season))].sort(), crops = CROPS.filter(c => rows.some(r => r.crop === c));
  const cell = (c, s) => rows.find(r => r.crop === c && r.season === s)?.rho ?? null;
  const shade = v => (v == null ? "" : ` style="background:rgba(76,175,125,${f1(Math.max(0, (v - 0.8) / 0.2) * 0.55)})"`);
  return `<table class="heat"><tr><th>Crop</th>${seasons.map(s => `<th>${s}</th>`).join("")}</tr>`
    + crops.map(c => `<tr><td>${c}</td>${seasons.map(s => `<td${shade(cell(c, s))}>${n2(cell(c, s))}</td>`).join("")}</tr>`).join("") + "</table>";
}

// item 3: our season dates minus HR-VPP's (days), middle 50 % of fields per crop and event
export function item3Boxes(boxes) {
  const W = 520, H = 230, L = 44, R = 10, T = 12, B = 34, lo = -80, hi = 80, ev = ["start", "peak", "end"];
  const crops = CROPS.filter(c => boxes.some(b => b.crop === c)), gw = (W - L - R) / Math.max(1, crops.length), bw = gw / 4.2;
  const y = v => f1(H - B - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (H - B - T));
  const grid = [-80, -40, 0, 40, 80].map(t => `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" class="${t === 0 ? "zero" : "grid"}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
  const rects = crops.map((c, i) => ev.map((e, k) => {
    const b = boxes.find(q => q.crop === c && q.event === e);
    if (!b) return "";
    const x0 = f1(L + i * gw + (k + 0.6) * bw);
    return `<rect class="box" x="${x0}" y="${y(b.q3)}" width="${f1(bw * 0.85)}" height="${f1(Math.max(1, y(b.q1) - y(b.q3)))}" fill="${EVENT_COLOURS[e]}" fill-opacity=".6"><title>${c} ${e}: median ${b.median} days (middle 50 %: ${b.q1} to ${b.q3})</title></rect>`
      + `<line x1="${x0}" x2="${f1(x0 + bw * 0.85)}" y1="${y(b.median)}" y2="${y(b.median)}" stroke="currentColor"/>`;
  }).join("") + `<text x="${f1(L + (i + 0.5) * gw)}" y="${H - 14}" text-anchor="middle">${c}</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Season dates, ours minus HR-VPP, in days">${grid}${rects}
    <text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">days (ours − HR-VPP)</text></svg>
    <p class="note">${ev.map(e => `<span style="color:${EVENT_COLOURS[e]}">■</span> ${e}`).join(" ")} · box = middle 50 % of fields, line = median, cut off at ±80 days · above 0: ours later</p>`;
}

// item 4: area mean of the 300 m cells, every 10 days: Copernicus LAI 300 m vs ours (before / after fix 2)
export function item4Chart(area, rows) {
  const W = 520, H = 140, L = 34, R = 8, T = 18, B = 22, max = 4.5;
  const t = rows.map(r => Date.parse(r[0])), t0 = Math.min(...t), t1 = Math.max(...t);
  const x = v => f1(L + ((v - t0) / Math.max(1, t1 - t0)) * (W - L - R)), y = v => f1(H - B - (Math.min(max, v) / max) * (H - B - T));
  const path = k => {
    let d = "", pen = false;
    rows.forEach((r, i) => { if (r[k] == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(t[i])},${y(r[k])}`; pen = true; });
    return d;
  };
  const years = []; for (let yr = new Date(t0).getUTCFullYear() + 1; yr <= new Date(t1).getUTCFullYear(); yr++) years.push(yr);
  const ax = years.map(yr => `<line x1="${x(Date.UTC(yr, 0, 1))}" x2="${x(Date.UTC(yr, 0, 1))}" y1="${T}" y2="${H - B}" class="grid"/><text x="${x(Date.UTC(yr, 0, 1))}" y="${H - 6}" text-anchor="middle">${yr}</text>`).join("")
    + [0, 2, 4].map(v => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 5}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${area}: our LAI averaged to 300 m and Copernicus LAI 300 m">${ax}
    <text x="${L}" y="12" font-weight="600">${area[0].toUpperCase() + area.slice(1)}</text>
    <path d="${path(1)}" fill="none" stroke="currentColor" stroke-width="1.4" stroke-opacity=".8"/>
    <path d="${path(2)}" fill="none" stroke="#e34948" stroke-width="1" stroke-opacity=".7"/>
    <path d="${path(3)}" fill="none" stroke="#1baf7a" stroke-width="1.8"/></svg>`;
}

export function fieldAccuracyHtml(d) {
  const at = d.area_test, u = d.uncertainty.lai, fx1 = d.fix1, fx2 = d.fix2, i2 = d.item2, s4 = d.item4.stats;
  const pooled = k => s4[k]["all | pooled"];
  const charts = Object.entries(d.item4.series).map(([a, rows]) => item4Chart(a, rows)).join("");
  return `
  <div class="prose" style="padding-block:24px 6px"><h2>Field level: weekly 10 m green LAI and NDVI (NRW study areas)</h2>
    <p>Four 8 × 8 km study areas in North Rhine-Westphalia, seasons 2021–2025, every arable field. Each test rule below was written down before its result existed, each result was looked at once, and nothing was tuned on it afterwards.</p></div>
  <div class="card"><h3>1. How a weekly value is made, and the test that chose the design</h3>
    <p>A clear Sentinel-2 image within 3 days is used as measured. Between clear images, gaps of up to 30 days are filled by a straight line; longer gaps by a model that uses the change seen by the Sentinel-1 radar, which sees through clouds. The test hid whole clear dates and compared each method with the hidden values: for gaps over 30 days the radar method was closer in <b>${at.lai.won} of ${at.lai.of}</b> area-seasons for LAI and <b>${at.ndvi.won} of ${at.ndvi.of}</b> for NDVI, in all four areas, so it was adopted.</p>
    <div class="tbl"><table><tr><th>Weekly LAI from</th><th>Typical error (± LAI)</th></tr>
      <tr><td>straight line, gap up to 15 days</td><td>${n2(u.line_le15)}</td></tr><tr><td>straight line, 16–30 days</td><td>${n2(u.line_16_30)}</td></tr>
      <tr><td>radar method, gap over 30 days</td><td>${n2(u.radar_gt30)}</td></tr><tr><td>after the last clear image (live), up to 30 days</td><td>${n2(u.live_le30)}</td></tr>
      <tr><td>after the last clear image, over 30 days</td><td>${n2(u.live_gt30)}</td></tr></table></div></div>
  <div class="card" style="margin-top:16px"><h3>2. Two quality fixes, tested the same way</h3>
    <p><b>Winter radar filter</b> (skip radar dates disturbed by frost, snow or wet soil): winter LAI better in ${fx1.verdict.lai.winter_better_areas} of 4 areas but NDVI in only ${fx1.verdict.ndvi.winter_better_areas} of 4 — <b>not adopted</b>.</p>
    <p><b>Single bad clear images</b> (haze or cloud edges the cloud mask missed): a clear value far below both neighbours while the radar shows no drop is treated as cloudy. Checked against Landsat on the same days, the removed LAI values were ${fx2.lai.ratio}× further from Landsat than the kept ones (${n2(fx2.lai.flagged_vs_landsat)} vs ${n2(fx2.lai.unflagged_vs_landsat)}) and only ${pct(fx2.lai.flagged_share)} of values were removed — <b>adopted for LAI</b>. For NDVI too many values were flagged (${pct(fx2.ndvi.flagged_share)}, limit 2 %), so NDVI is unchanged. The fix changed ${fx2.rebuild.changed_pct} % of weekly values, raising false dips by ${fx2.rebuild.mean_where_changed} LAI on average.</p></div>
  <div class="card" style="margin-top:16px"><h3>3. Compared with Copernicus HR-VPP (10 m, 10-daily, seasons 2021–2024)</h3>
    <div class="acc-grid"><div><h4>Cloudy weeks: which product ranks the fields closer to Landsat?</h4>${item2Scatter(i2)}
      <p>On weeks without a clear Sentinel-2 image, our weekly LAI ranked the fields closer to Landsat 8/9 on <b>${i2.won} of ${i2.of} dates</b> (median rank agreement ${n2(i2.median_ours)} vs ${n2(i2.median_ppi)}). The rule fixed in advance needed 60 % of dates.</p></div>
      <div><h4>Clear days: same ranking of fields?</h4>${item1Table(d.item1)}
      <p class="note">Rank agreement (Spearman) of our LAI with the HR-VPP plant index on weeks with a clear image; 1 = same order of fields.</p></div></div>
    <h4>Season dates: start, peak and end</h4>${item3Boxes(d.item3)}
    <p class="note">The peak agrees within 1–3 weeks. Start and end are defined differently (ours: half of the seasonal rise; HR-VPP: earlier in the green-up and later in the green-down), and our weekly series ends on 31 August while sugar beet and maize stay green into autumn. For maize, HR-VPP's first season is often the spring cover crop.</p></div>
  <div class="card" style="margin-top:16px"><h3>4. Compared with Copernicus LAI 300 m (10-daily, seasons 2021–2025)</h3>
    <p>Our 10 m LAI averaged over each 300 m cell inside the areas: typical difference ${n2(pooled("after").rmse)} LAI, ours ${n2(Math.abs(pooled("after").bias))} lower on average, correlation <b>${n2(pooled("after").r)}</b> (before the bad-image fix: ${n2(pooled("before").rmse)}, ${n2(Math.abs(pooled("before").bias))}, ${n2(pooled("before").r)}). In spring ours is about 0.3 lower; a 300 m cell also contains grass, hedges and trees that green up earlier than fields.</p>
    <div class="acc-grid wide">${charts}</div>
    <p class="note">— Copernicus LAI 300 m · <span style="color:#1baf7a">—</span> ours, after the fix · <span style="color:#e34948">—</span> ours, before · mean of all 300 m cells of each area, every 10 days</p></div>
  <div class="card" style="margin-top:16px"><h3>Limits</h3><ul>
    <li>The Landsat LAI used as truth for cloudy weeks is made with our own retrieval method (applied to Landsat images); a truth independent of our method (ground measurements) is still to come.</li>
    <li>All tests used seasons 2021–2025, which also guided the design; a final check on season 2026 follows once all methods are fixed.</li>
    <li>The weekly series runs from 1 October to 31 August; late harvests (sugar beet, maize) are not covered yet.</li>
    <li>Four study areas in North Rhine-Westphalia; other regions are not tested yet.</li></ul></div>`;
}

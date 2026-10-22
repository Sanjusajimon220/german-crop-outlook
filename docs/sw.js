// Offline support for the installable Crop Outlook (service worker).
// App shell and district data are stored at install; field files are stored as they are viewed (their URLs carry
// ?v=<data version>, so a data rebuild never mixes old and new files); px.bin range requests are answered from a stored
// full copy. A new APP_VERSION replaces the old caches on the next visit.
const APP_VERSION = "2026-10-22a";
const SHELL = `shell-${APP_VERSION}`, FIELD = "field-data", CDN = "cdn-libs";
const SHELL_FILES = ["./", "./index.html", "./manifest.webmanifest", "./js/field_core.js", "./js/field_view.js", "./js/my_fields.js",
  "./js/isoxml.js", "./js/field_accuracy.js", "./icons/icon-192.png", "./icons/icon-512.png", "./data/meta.json", "./data/districts.geojson",
  "./data/accuracy.json", "./data/ndvi.json"]
  .concat(["winter_wheat", "winter_barley", "grain_maize", "silage_maize", "potato"].flatMap(c => [`./data/fc_${c}.json`, `./data/water_${c}.json`]));
const CDN_FILES = ["https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js", "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"];

self.addEventListener("install", e => {
  e.waitUntil((async () => {
    await (await caches.open(SHELL)).addAll(SHELL_FILES);
    const cdn = await caches.open(CDN);
    await Promise.all(CDN_FILES.map(u => fetch(u, { mode: "cors" }).then(r => r.ok && cdn.put(u, r)).catch(() => {})));
    self.skipWaiting();
  })());
});

self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("shell-") && k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

// a byte range of a stored full file, answered as 206 Partial Content
async function rangeFrom(full, range) {
  const buf = await full.arrayBuffer(), m = /bytes=(\d+)-(\d+)?/.exec(range || "");
  if (!m) return new Response(buf, { status: 200, headers: { "Content-Type": "application/octet-stream" } });
  const a = +m[1], b = m[2] != null ? +m[2] : buf.byteLength - 1, part = buf.slice(a, b + 1);
  return new Response(part, { status: 206, headers: { "Content-Type": "application/octet-stream", "Content-Length": String(part.byteLength),
    "Content-Range": `bytes ${a}-${b}/${buf.byteLength}` } });
}

self.addEventListener("fetch", e => {
  const req = e.request, u = new URL(req.url);
  if (req.method !== "GET") return;
  if (CDN_FILES.includes(req.url)) {                                            // libraries: stored copy first
    e.respondWith(caches.match(req.url).then(r => r || fetch(req)));
    return;
  }
  if (u.origin !== location.origin) return;
  const path = u.pathname;
  if (path.includes("/field/") && path.endsWith("/index.json")) {             // data version: network first
    e.respondWith(fetch(req).then(r => { const c = r.clone(); caches.open(FIELD).then(x => x.put(req, c)); return r; })
      .catch(() => caches.match(req)));
    return;
  }
  if (path.includes("/field/") && path.endsWith("/px.bin")) {                 // pixel blocks: whole file stored once
    e.respondWith((async () => {
      const key = u.href, cache = await caches.open(FIELD);
      let full = await cache.match(key);
      if (!full) {
        try { const r = await fetch(key); if (r.ok) { await cache.put(key, r.clone()); full = r; } } catch (err) { /* offline */ }
      }
      if (!full) return fetch(req);
      return rangeFrom(full.clone(), req.headers.get("Range"));
    })());
    return;
  }
  if (path.includes("/field/")) {                                             // versioned field files: cache first
    e.respondWith(caches.match(req).then(r => r || fetch(req).then(res => {
      if (res.ok) { const c = res.clone(); caches.open(FIELD).then(x => x.put(req, c)); }
      return res;
    })));
    return;
  }
  // app shell and district data: network first (always the newest version online), stored copy offline
  e.respondWith(fetch(req).then(res => {
    if (res.ok && (req.mode === "navigate" || SHELL_FILES.some(f => path.endsWith(f.slice(1))))) { const c = res.clone(); caches.open(SHELL).then(x => x.put(req, c)); }
    return res;
  }).catch(async () => (await caches.match(req)) || (req.mode === "navigate" ? caches.match("./index.html") : Response.error())));
});

// the page asks to keep the files of a saved field available offline
self.addEventListener("message", e => {
  if (e.data?.type !== "keep") return;
  e.waitUntil((async () => {
    const cache = await caches.open(FIELD);
    for (const u of e.data.urls) { try { if (!(await cache.match(u))) { const r = await fetch(u); if (r.ok) await cache.put(u, r); } } catch (err) { /* offline */ } }
  })());
});

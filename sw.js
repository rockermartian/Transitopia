/* Correspondance — service worker : appli hors cache réseau + cache des photos aériennes */
const APP = 'corr-app-v3';
const TILES = 'corr-tiles-v1';
const MAX_TILES = 3000;
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(APP).then((c) => c.addAll(SHELL).catch(() => {})).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== APP && k !== TILES).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

async function trim() {
  const c = await caches.open(TILES);
  const keys = await c.keys();
  if (keys.length > MAX_TILES) for (const k of keys.slice(0, keys.length - MAX_TILES)) await c.delete(k);
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // tuiles IGN : cache d'abord
  if (url.hostname === 'data.geopf.fr' && url.searchParams.get('REQUEST') === 'GetTile') {
    e.respondWith(caches.open(TILES).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) { c.put(req, res.clone()); if (Math.random() < 0.02) trim(); }
      return res;
    }));
    return;
  }
  // appli : réseau d'abord (mises à jour), cache en secours
  if (url.origin === location.origin || url.hostname === 'cdnjs.cloudflare.com' || url.hostname.endsWith('gstatic.com') || url.hostname === 'fonts.googleapis.com') {
    e.respondWith(fetch(req).then((res) => {
      if (res.ok || res.type === 'opaque') { const cl = res.clone(); caches.open(APP).then((c) => c.put(req, cl)); }
      return res;
    }).catch(() => caches.match(req).then((r) => r || caches.match('./index.html'))));
  }
});

// Offline shell for Pocket Field Guide.
// Model weights are cached separately by transformers.js (Cache API), so this
// worker only needs to keep the app itself available with no network.
const CACHE = 'pfg-shell-v2';
const SCOPE = self.registration.scope;
const SHELL = ['', 'index.html', 'manifest.webmanifest', 'label-embeds.json', 'label-embeds.bin', 'ort/ort-wasm-simd-threaded.asyncify.mjs', 'ort/ort-wasm-simd-threaded.asyncify.wasm'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL.map((p) => new URL(p, SCOPE).href))).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('pfg-shell-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;

  // Pages: network first so updates show up, cached copy when offline.
  if (req.mode === 'navigate') {
    e.respondWith(
      fetch(req)
        .then((res) => { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(new URL('index.html', SCOPE).href, copy)); return res; })
        .catch(() => caches.match(new URL('index.html', SCOPE).href)),
    );
    return;
  }

  // Unhashed data files: network first so an updated field guide shows up.
  if (url.pathname.includes('label-embeds.')) {
    e.respondWith(
      fetch(req)
        .then((res) => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); } return res; })
        .catch(() => caches.match(req)),
    );
    return;
  }

  // Hashed assets and everything else: cache first, fill the cache as we go.
  e.respondWith(
    caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(req, copy)); }
      return res;
    })),
  );
});

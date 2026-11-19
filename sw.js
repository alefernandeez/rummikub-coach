// Caché offline: la app, el motor ONNX y los modelos quedan en el móvil tras la primera visita.
const V = 'rummi-v4-2';
const FILES = ['./', 'index.html', 'solver.js', 'vision.js', 'live.js', 'manifest.webmanifest', 'icon.png',
  'vendor/ort.wasm.min.js', 'vendor/ort-wasm-simd-threaded.wasm', 'vendor/ort-wasm-simd-threaded.mjs',
  'models/det.onnx', 'models/cls.onnx'];
self.addEventListener('install', e => e.waitUntil(caches.open(V).then(c => c.addAll(FILES)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== V).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  // red primero para la app (para recibir actualizaciones), caché si no hay conexión
  e.respondWith(fetch(e.request).then(r => { const cp = r.clone(); caches.open(V).then(c => c.put(e.request, cp)); return r; })
    .catch(() => caches.match(e.request, { ignoreSearch: true })));
});

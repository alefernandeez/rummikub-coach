// Visión offline: detector (centros + "arriba" + tamaño) y clasificador (número, color, girada).
// Misma lógica que ml/infer.py.
(function (root) {
  const CW = 48, CH = 64, MARGIN = 1.12, DET_SIZE = 960, THR = 0.35;
  let det = null, cls = null, ready = null;

  function init(base) {
    if (ready) return ready;
    ort.env.wasm.wasmPaths = base + 'vendor/';
    ort.env.wasm.numThreads = 1;
    ready = (async () => {
      const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      det = await ort.InferenceSession.create(base + 'models/det.onnx', opt);
      cls = await ort.InferenceSession.create(base + 'models/cls.onnx', opt);
    })();
    return ready;
  }

  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function softmax(arr, off, n) {
    let m = -Infinity; for (let i = 0; i < n; i++) m = Math.max(m, arr[off + i]);
    const e = []; let s = 0; for (let i = 0; i < n; i++) { e.push(Math.exp(arr[off + i] - m)); s += e[i]; }
    return e.map(v => v / s);
  }
  function toLab(r, g, b) {
    const f = c => { c /= 255; return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92; };
    const R = f(r), G = f(g), B = f(b);
    const X = (R * 0.4124 + G * 0.3576 + B * 0.1805) / 0.95047, Y = R * 0.2126 + G * 0.7152 + B * 0.0722, Z = (R * 0.0193 + G * 0.1192 + B * 0.9505) / 1.08883;
    const h = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
    return [116 * h(Y) - 16, 500 * (h(X) - h(Y)), 200 * (h(Y) - h(Z))];
  }
  const median = a => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0; };
  const argmax = a => a.reduce((bi, v, i, ar) => v > ar[bi] ? i : bi, 0);

  // src: canvas con la imagen a resolución completa
  async function detect(src) {
    const W = src.width, H = src.height, sc = DET_SIZE / Math.max(W, H);
    const w2 = Math.ceil(W * sc / 32) * 32, h2 = Math.ceil(H * sc / 32) * 32;
    const cv = document.createElement('canvas'); cv.width = w2; cv.height = h2;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.fillStyle = 'rgb(128,128,128)'; cx.fillRect(0, 0, w2, h2);
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(src, 0, 0, Math.round(W * sc), Math.round(H * sc));
    const px = cx.getImageData(0, 0, w2, h2).data;
    const N = w2 * h2, data = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { data[i] = px[4 * i] / 255 - 0.5; data[N + i] = px[4 * i + 1] / 255 - 0.5; data[2 * N + i] = px[4 * i + 2] / 255 - 0.5; }
    const out = (await det.run({ x: new ort.Tensor('float32', data, [1, 3, h2, w2]) })).y;
    const [, , oh, ow] = out.dims, o = out.data, P = oh * ow;
    const hm = new Float32Array(P); for (let i = 0; i < P; i++) hm[i] = sigmoid(o[i]);
    const dets = [];
    for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
      const v = hm[y * ow + x]; if (v < THR) continue;
      let isMax = true, sw = 0, sx = 0, sy = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx; if (yy < 0 || xx < 0 || yy >= oh || xx >= ow) continue;
        const u = hm[yy * ow + xx]; if (u > v) isMax = false; sw += u; sx += u * xx; sy += u * yy;
      }
      if (!isMax) continue;
      let ux = o[P + y * ow + x], uy = o[2 * P + y * ow + x]; const n = Math.hypot(ux, uy) + 1e-6; ux /= n; uy /= n;
      const w = Math.exp(o[3 * P + y * ow + x]) * 4;
      dets.push({ cx: (sx / sw) * 4 / sc, cy: (sy / sw) * 4 / sc, ux, uy, w: w / sc, score: v });
    }
    // supresión de duplicados: dos centros dentro de la misma ficha
    dets.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of dets) if (!keep.some(k => Math.hypot(k.cx - d.cx, k.cy - d.cy) < 0.5 * Math.max(k.w, d.w))) keep.push(d);
    return keep;
  }

  function cropInto(ctx, src, d, k) {
    const h = d.w * 1.4, sw = d.w * MARGIN, sh = h * MARGIN;
    const rx = -d.uy, ry = d.ux, dx = -d.ux, dy = -d.uy;
    const tlx = d.cx - rx * sw / 2 - dx * sh / 2, tly = d.cy - ry * sw / 2 - dy * sh / 2;
    // forward: img = tl + u*(r*sw/CW) + v*(dn*sh/CH)  -> invert
    const a = rx * sw / CW, b = ry * sw / CW, c = dx * sh / CH, e = dy * sh / CH;
    const det_ = a * e - b * c;
    const ia = e / det_, ib = -b / det_, ic = -c / det_, ie = a / det_;
    const itx = -(ia * tlx + ic * tly), ity = -(ib * tlx + ie * tly);
    ctx.save();
    ctx.beginPath(); ctx.rect(k * CW, 0, CW, CH); ctx.clip();
    ctx.setTransform(ia, ib, ic, ie, itx + k * CW, ity);
    ctx.drawImage(src, 0, 0);
    ctx.restore();
  }

  async function runCls(src, ds) {
    const cv = document.createElement('canvas'); cv.width = CW * ds.length; cv.height = CH;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingQuality = 'high';
    ds.forEach((d, k) => cropInto(ctx, src, d, k));
    const px = ctx.getImageData(0, 0, cv.width, CH).data;
    const n = ds.length, S = CW * CH, data = new Float32Array(n * 3 * S);
    for (let k = 0; k < n; k++) for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) {
      const p = 4 * (y * cv.width + k * CW + x), q = k * 3 * S + y * CW + x;
      data[q] = px[p] / 255 - 0.5; data[q + S] = px[p + 1] / 255 - 0.5; data[q + 2 * S] = px[p + 2] / 255 - 0.5;
    }
    const r = await cls.run({ x: new ort.Tensor('float32', data, [n, 3, CH, CW]) });
    // color medio (Lab) de la zona del hueco redondo: en una ficha real siempre es crema
    const base = ds.map((_, k) => {
      let R = 0, G = 0, B = 0, m = 0;
      for (let y = Math.floor(CH * 0.62); y < Math.floor(CH * 0.9); y++) for (let x = Math.floor(CW * 0.15); x < Math.floor(CW * 0.85); x++) {
        const p = 4 * (y * cv.width + k * CW + x); R += px[p]; G += px[p + 1]; B += px[p + 2]; m++;
      }
      return toLab(R / m, G / m, B / m);
    });
    return ds.map((_, k) => ({ num: softmax(r.num.data, k * 15, 15), col: softmax(r.col.data, k * 6, 6), flip: softmax(r.flip.data, k * 2, 2), lab: base[k] }));
  }

  async function classify(src, dets) {
    if (!dets.length) return [];
    const res = [];
    for (let i = 0; i < dets.length; i += 32) res.push(...await runCls(src, dets.slice(i, i + 32)));
    const flipped = dets.map((d, i) => i).filter(i => res[i].flip[1] > 0.5);
    if (flipped.length) {
      flipped.forEach(i => { dets[i].ux *= -1; dets[i].uy *= -1; });
      const r2 = await runCls(src, flipped.map(i => dets[i]));
      flipped.forEach((i, k) => res[i] = r2[k]);
    }
    const out = [];
    const mL = median(res.map(r => r.lab[0])), mA = median(res.map(r => r.lab[1])), mB = median(res.map(r => r.lab[2]));
    dets.forEach((d, i) => {
      const k = argmax(res[i].num); if (k === 14) return;
      if (dets.length >= 4) {
        const [L, A, B] = res[i].lab;
        if (L - mL < -16 || Math.hypot(A - mA, B - mB) > 15) return;   // no es una ficha (piel, ropa, objetos)
      }
      const c = argmax(res[i].col), pn = res[i].num[k], pc = res[i].col[c];
      out.push({ ...d, num: k, color: c, pn, pc, unsure: pn < 0.7 || (k !== 0 && pc < 0.7) });
    });
    return out;
  }

  function groupSets(tiles) {
    const n = tiles.length, parent = [...Array(n).keys()];
    const find = a => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const a = tiles[i], b = tiles[j], wm = (a.w + b.w) / 2;
      const dx = b.cx - a.cx, dy = b.cy - a.cy, rx = -a.uy, ry = a.ux;
      const along = Math.abs(dx * rx + dy * ry), across = Math.abs(dx * a.ux + dy * a.uy);
      const same = a.ux * b.ux + a.uy * b.uy > 0.7;
      if (same && along < wm * 1.45 && across < wm * 0.55) parent[find(i)] = find(j);
    }
    const groups = new Map();
    tiles.forEach((t, i) => { const r = find(i); if (!groups.has(r)) groups.set(r, []); groups.get(r).push(t); });
    return [...groups.values()].map(g => { const rx = -g[0].uy, ry = g[0].ux; return g.sort((p, q) => (p.cx * rx + p.cy * ry) - (q.cx * rx + q.cy * ry)); });
  }

  function rackFilter(tiles) {
    if (!tiles.length) return tiles;
    const ref = tiles.reduce((a, b) => b.w > a.w ? b : a), h = ref.w * 1.4;
    // fichas de pie, grandes (las más cercanas) y en la franja del atril (admite atril de dos pisos)
    const r = tiles.filter(t => t.w >= 0.7 * ref.w && t.uy < -0.7 && Math.abs(t.cy - ref.cy) < 1.5 * h);
    return r.sort((a, b) => Math.abs(a.cy - b.cy) > h * 0.6 ? a.cy - b.cy : a.cx - b.cx);
  }

  async function analyze(src, mode) {
    await ready;
    const t0 = performance.now();
    const tiles = await classify(src, await detect(src));
    const ms = Math.round(performance.now() - t0);
    if (mode === 'rack') return { rack: rackFilter(tiles), all: tiles, ms };
    // fuera detecciones dudosas y aisladas (suelen ser falsos positivos fuera de la mesa)
    const sets = groupSets(tiles).filter(g => !(g.length === 1 && g[0].pn < 0.7));
    return { sets, all: sets.flat(), ms };
  }

  root.Vision = { init, analyze, detect, classify, groupSets };
})(this);

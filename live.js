// Escaneo continuo: acumula fichas de muchos fotogramas en un "mapa" (coordenadas del primer fotograma).
// Cada fotograma se alinea con el mapa usando las fichas ya conocidas (RANSAC de similitud:
// escala + giro + desplazamiento) y cada ficha va sumando votos hasta quedar fijada.
(function (root) {
  const FIX_SCORE = 1.8, FIX_RATIO = 0.7, MIN_W = 34;   // px de ficha en el fotograma para poder leerla bien

  const keyOf = d => d.num === 0 ? 'J' : d.num + '-' + d.color;
  const median = a => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0; };

  // similitud como número complejo: q = a*p + t
  const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
  const cdiv = (a, b) => { const d = b[0] * b[0] + b[1] * b[1] + 1e-9; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
  const apply = (T, x, y) => { const r = cmul(T.a, [x, y]); return [r[0] + T.t[0], r[1] + T.t[1]]; };
  const rotate = (T, ux, uy) => { const s = Math.hypot(T.a[0], T.a[1]) + 1e-9; const r = cmul([T.a[0] / s, T.a[1] / s], [ux, uy]); return r; };

  function fit(pairs) {
    let px = 0, py = 0, qx = 0, qy = 0;
    for (const [p, q] of pairs) { px += p[0]; py += p[1]; qx += q[0]; qy += q[1]; }
    const n = pairs.length; px /= n; py /= n; qx /= n; qy /= n;
    let num = [0, 0], den = 0;
    for (const [p, q] of pairs) {
      const dp = [p[0] - px, p[1] - py], dq = [q[0] - qx, q[1] - qy];
      num = [num[0] + dq[0] * dp[0] + dq[1] * dp[1], num[1] + dq[1] * dp[0] - dq[0] * dp[1]];
      den += dp[0] * dp[0] + dp[1] * dp[1];
    }
    const a = [num[0] / (den + 1e-9), num[1] / (den + 1e-9)];
    const ap = cmul(a, [px, py]);
    return { a, t: [qx - ap[0], qy - ap[1]] };
  }

  class World {
    constructor(kind) { this.kind = kind; this.tiles = []; this.frames = 0; this.lost = 0; }
    reset() { this.tiles = []; this.frames = 0; this.lost = 0; }

    status(t) {
      let best = null, bs = 0, tot = 0;
      for (const [k, v] of t.votes) { tot += v; if (v > bs) { bs = v; best = k; } }
      return { key: best, score: bs, fixed: bs >= FIX_SCORE && bs / (tot + 1e-9) >= FIX_RATIO, total: tot };
    }

    vote(t, d) {
      const k = keyOf(d), s = d.num === 0 ? d.pn : d.pn * d.pc;
      t.votes.set(k, (t.votes.get(k) || 0) + s);
      t.seen++;
    }

    newTile(d, x, y, ux, uy, w) {
      const t = { x, y, ux, uy, w, votes: new Map(), seen: 0, small: false, id: Math.random().toString(36).slice(2, 8) };
      if (d.w >= MIN_W) this.vote(t, d); else t.small = true;
      this.tiles.push(t); return t;
    }

    // Alinea el fotograma con el mapa. Devuelve T (fotograma -> mapa) o null.
    register(dets) {
      const conf = this.tiles.map(t => ({ t, s: this.status(t) })).filter(o => o.s.score >= 0.9);
      const cand = [];
      for (const d of dets) {
        if (d.pn < 0.6) continue;
        const k = keyOf(d);
        for (const o of conf) if (o.s.key === k) cand.push([d, o.t]);
      }
      if (cand.length < 2) return null;
      const wm = median(this.tiles.map(t => t.w)) || 40;
      let best = null, bestIn = [];
      const pairs = [];
      if (cand.length <= 28) { for (let i = 0; i < cand.length; i++) for (let j = i + 1; j < cand.length; j++) pairs.push([i, j]); }
      else for (let k = 0; k < 400; k++) { const i = Math.floor(Math.random() * cand.length), j = Math.floor(Math.random() * cand.length); if (i !== j) pairs.push([i, j]); }
      for (const [i, j] of pairs) {
        const [d1, w1] = cand[i], [d2, w2] = cand[j];
        if (d1 === d2 || w1 === w2) continue;
        const dp = [d2.cx - d1.cx, d2.cy - d1.cy]; if (Math.hypot(dp[0], dp[1]) < 5) continue;
        const a = cdiv([w2.x - w1.x, w2.y - w1.y], dp);
        const sc = Math.hypot(a[0], a[1]); if (sc < 0.3 || sc > 3.5) continue;
        const ap = cmul(a, [d1.cx, d1.cy]); const T = { a, t: [w1.x - ap[0], w1.y - ap[1]] };
        const inl = [], usedW = new Set(), usedD = new Set();
        for (const [d, w] of cand) {
          if (usedW.has(w) || usedD.has(d)) continue;
          const [x, y] = apply(T, d.cx, d.cy);
          if (Math.hypot(x - w.x, y - w.y) > 0.4 * wm) continue;
          const u = rotate(T, d.ux, d.uy); if (u[0] * w.ux + u[1] * w.uy < 0.7) continue;
          inl.push([[d.cx, d.cy], [w.x, w.y]]); usedW.add(w); usedD.add(d);
        }
        if (inl.length > bestIn.length) { bestIn = inl; best = T; }
      }
      const need = this.tiles.length < 6 ? 2 : 3;
      if (!best || bestIn.length < need) return null;
      return fit(bestIn);
    }

    // Añade un fotograma. dets en coordenadas del fotograma.
    addFrame(dets) {
      if (!dets.length) return { ok: false, reason: 'vacio' };
      let T;
      if (!this.tiles.length) T = { a: [1, 0], t: [0, 0] };
      else { T = this.register(dets); if (!T) { this.lost++; return { ok: false, reason: 'perdido' }; } }
      this.lost = 0; this.frames++;
      const sc = Math.hypot(T.a[0], T.a[1]);
      const out = [];
      const taken = new Set();
      // asociar por cercanía (primero las detecciones más seguras)
      const order = dets.map((d, i) => i).sort((i, j) => dets[j].pn - dets[i].pn);
      for (const i of order) {
        const d = dets[i];
        const [x, y] = apply(T, d.cx, d.cy), u = rotate(T, d.ux, d.uy), w = d.w * sc;
        let best = null, bd = Infinity, near = Infinity, flipped = false;
        for (const t of this.tiles) {
          const dd = Math.hypot(t.x - x, t.y - y);
          if (dd >= 0.65 * Math.max(t.w, w)) continue;
          near = Math.min(near, dd);
          if (taken.has(t)) continue;
          if ((t.ux * u[0] + t.uy * u[1]) <= 0.5) { flipped = true; continue; }
          if (dd < bd) { bd = dd; best = t; }
        }
        let t;
        if (best) {
          t = best;
          if (d.w < MIN_W) { taken.add(t); out[i] = t; continue; }   // demasiado lejos para leerla: solo se marca
          this.vote(t, d); t.small = false;
          // posición media (absorbe la deriva de perspectiva)
          const k = Math.min(0.5, 1 / t.seen);
          t.x += (x - t.x) * k; t.y += (y - t.y) * k; t.w += (w - t.w) * k;
          const nx = t.ux + (u[0] - t.ux) * k, ny = t.uy + (u[1] - t.uy) * k, nn = Math.hypot(nx, ny) + 1e-9; t.ux = nx / nn; t.uy = ny / nn;
        } else if (near < Infinity || flipped) {
          continue;   // encima de una ficha ya conocida (lectura doble o girada): se ignora
        } else t = this.newTile(d, x, y, u[0], u[1], w);
        taken.add(t);
        out[i] = t;
      }
      return { ok: true, T, matched: out };
    }

    // Fichas finales (descarta las vistas pocas veces y con poca confianza)
    result() {
      return this.tiles.map(t => ({ t, s: this.status(t) }))
        .filter(o => o.s.key && o.s.total >= 0.6 && (o.t.seen >= 2 || o.s.score >= 0.9))
        .map(({ t, s }) => {
          const [n, c] = s.key === 'J' ? [0, 0] : s.key.split('-').map(Number);
          return { cx: t.x, cy: t.y, ux: t.ux, uy: t.uy, w: t.w, num: n, color: c, pn: s.score / (s.total + 1e-9), unsure: !s.fixed };
        });
    }

    counts() {
      let fixed = 0, pending = 0;
      for (const t of this.tiles) { const s = this.status(t); if (t.small && !t.seen) { pending++; continue; } if (s.total < 0.6 || (t.seen < 2 && s.score < 0.9)) continue; if (s.fixed) fixed++; else pending++; }
      return { fixed, pending };
    }
  }

  const api = { World, keyOf };
  if (typeof module !== 'undefined') module.exports = api; else root.LiveScan = api;
})(this);

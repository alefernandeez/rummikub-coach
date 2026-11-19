// Rummikub solver — reglas oficiales, generalizado a C colores, K copias y J comodines.
// Objetivo: maximizar fichas de la mano jugadas (desempate: puntos de penalización quitados,
// comodín = 30). Antes de la primera bajada solo cuentan fichas propias y hay que sumar >= 30.
// DP exacto por valor (1..13) y color. Estado por color: nº de escaleras abiertas de
// longitud 1, 2 y >=3 (máx. K abiertas); además comodines usados, fichas para grupos y puntos.

(function (root) {
  const MAXN = 13, BIG = 100000, MELD = 30;

  function makeColorStates(K) {
    const list = [], idx = new Map();
    for (let a = 0; a <= K; a++) for (let b = 0; a + b <= K; b++) for (let c = 0; a + b + c <= K; c++) {
      idx.set(a + ',' + b + ',' + c, list.length); list.push([a, b, c]);
    }
    return { list, idx };
  }

  // grupos: m grupos de 3..C fichas de colores distintos
  function groupsFeasible(gsum, gmax, C, M) {
    if (gsum === 0) return 0;
    for (let m = Math.max(1, gmax); m <= M; m++) if (gsum >= 3 * m && gsum <= C * m) return m;
    return -1;
  }

  function solveCounts(hand, table, handJ, tableJ, meldMode, cfg) {
    const C = cfg.colors, K = cfg.copies;
    const KMAX = K + 1;                       // fichas efectivas por casilla (reales + comodines)
    const totalJ = handJ + tableJ;
    const CS = makeColorStates(K), NS = CS.list.length;
    const MAXG = C * KMAX, M = KMAX;
    // empaquetado: colores en base NS, luego ju, gsum, gmax, pts
    const pack = (cs, ju, gsum, gmax, pts) => {
      let k = 0; for (let i = 0; i < C; i++) k = k * NS + cs[i];
      return (((k * (totalJ + 1) + ju) * (MAXG + 1) + gsum) * (KMAX + 1) + gmax) * 31 + pts;
    };
    const unpack = key => {
      const pts = key % 31; key = (key - pts) / 31;
      const gmax = key % (KMAX + 1); key = (key - gmax) / (KMAX + 1);
      const gsum = key % (MAXG + 1); key = (key - gsum) / (MAXG + 1);
      const ju = key % (totalJ + 1); key = (key - ju) / (totalJ + 1);
      const cs = Array(C);
      for (let i = C - 1; i >= 0; i--) { cs[i] = key % NS; key = (key - cs[i]) / NS; }
      return { cs, ju, gsum, gmax, pts };
    };
    // transiciones por color
    const tcache = new Map();
    const trans = (code, k, last) => {
      const tk = code * 64 + k * 2 + (last ? 1 : 0);
      if (tcache.has(tk)) return tcache.get(tk);
      const [n1, n2, n3] = CS.list[code];
      const out = [];
      for (let e3 = 0; e3 <= n3; e3++) {
        const ext = n1 + n2 + e3;
        for (let nn = 0; ext + nn <= k; nn++) {
          const g = k - ext - nn;
          if (g > KMAX) continue;
          if (nn + n1 + n2 + e3 > K) continue;
          if (last && (nn > 0 || n1 > 0)) continue;
          out.push({ ns: CS.idx.get(nn + ',' + n1 + ',' + (n2 + e3)), g, e3, nn });
        }
      }
      tcache.set(tk, out);
      return out;
    };

    let layer = new Map();
    layer.set(pack(Array(C).fill(0), 0, 0, 0, 0), { val: 0, prev: null, dec: null });
    const layers = [];
    for (let v = 1; v <= MAXN; v++) {
      for (let c = 0; c < C; c++) {
        const next = new Map();
        const tb = table[c][v], hd = hand[c][v];
        for (const [key, node] of layer) {
          const s = unpack(key);
          for (let r = tb; r <= tb + hd; r++) {
            for (let j = 0; s.ju + j <= totalJ; j++) {
              const k = r + j;
              if (k > KMAX) break;
              for (const t of trans(s.cs[c], k, v === MAXN)) {
                const gsum = s.gsum + t.g;
                if (gsum > MAXG) continue;
                const gmax = Math.max(s.gmax, t.g);
                const cs = s.cs.slice(); cs[c] = t.ns;
                const pts = meldMode ? Math.min(MELD, s.pts + k * v) : 0;
                const ht = r - tb;
                const val = node.val + ht * BIG + ht * v;
                const nk = pack(cs, s.ju + j, gsum, gmax, pts);
                const ex = next.get(nk);
                if (!ex || ex.val < val) next.set(nk, { val, prev: key, dec: { v, c, r, j, t } });
              }
            }
          }
        }
        layers.push(next); layer = next;
      }
      const next = new Map();
      for (const [key, node] of layer) {
        const s = unpack(key);
        const m = groupsFeasible(s.gsum, s.gmax, C, M);
        if (m < 0) continue;
        const nk = pack(s.cs, s.ju, 0, 0, s.pts);
        const ex = next.get(nk);
        if (!ex || ex.val < node.val) next.set(nk, { val: node.val, prev: key, dec: { groupsEnd: v, m } });
      }
      layers.push(next); layer = next;
    }
    let best = null, bestKey = null;
    for (const [key, node] of layer) {
      const s = unpack(key);
      if (s.ju < tableJ) continue;
      if (meldMode && s.pts < MELD) continue;
      const hj = s.ju - tableJ;
      if (hj > handJ) continue;
      const total = node.val + hj * (BIG + 30);
      if (best === null || total > best) { best = total; bestKey = key; }
    }
    if (bestKey === null) return null;
    const decs = [];
    let key = bestKey;
    for (let li = layers.length - 1; li >= 0; li--) { const nd = layers[li].get(key); decs.push(nd.dec); key = nd.prev; }
    decs.reverse();
    return { decs, tableJ, tableCounts: table, C };
  }

  function buildSets(res) {
    const C = res.C, sets = [];
    const open = Array.from({ length: C }, () => []);
    let tablePoolJ = res.tableJ, groupBuf = [];
    for (const d of res.decs) {
      if (d.groupsEnd !== undefined) {
        if (d.m > 0) {
          const groups = Array.from({ length: d.m }, () => []);
          const byColor = Array.from({ length: C }, () => []);
          for (const t of groupBuf) byColor[t.c].push(t);
          const order = [...Array(C).keys()].sort((x, y) => byColor[y].length - byColor[x].length);
          for (const c of order) {
            const used = new Set();
            for (const t of byColor[c]) {
              let bi = -1;
              for (let i = 0; i < groups.length; i++) if (!used.has(i) && (bi < 0 || groups[i].length < groups[bi].length)) bi = i;
              used.add(bi); groups[bi].push(t);
            }
          }
          for (const g of groups) sets.push({ type: 'grupo', tiles: g });
        }
        groupBuf = []; continue;
      }
      const { v, c, r, j, t } = d;
      const tb = res.tableCounts[c][v];
      const tiles = [];
      for (let i = 0; i < r; i++) tiles.push({ joker: false, c, n: v, fromHand: i >= tb });
      for (let i = 0; i < j; i++) { const fh = tablePoolJ <= 0; if (!fh) tablePoolJ--; tiles.push({ joker: true, c, n: v, fromHand: fh }); }
      const runs = open[c];
      const short = runs.filter(x => x.tiles.length < 3);
      const long = runs.filter(x => x.tiles.length >= 3);
      const ext = short.concat(long.slice(0, t.e3));
      for (const x of long.slice(t.e3)) sets.push({ type: 'escalera', tiles: x.tiles });
      open[c] = [];
      let ti = 0;
      for (const x of ext) { x.tiles.push(tiles[ti++]); open[c].push(x); }
      for (let i = 0; i < t.nn; i++) open[c].push({ tiles: [tiles[ti++]] });
      while (ti < tiles.length) groupBuf.push(tiles[ti++]);
    }
    for (let c = 0; c < C; c++) for (const x of open[c]) sets.push({ type: 'escalera', tiles: x.tiles });
    return sets;
  }

  function toCounts(tiles, C) {
    const cnt = Array.from({ length: C }, () => Array(MAXN + 1).fill(0));
    let J = 0;
    for (const t of tiles) { if (t.joker) J++; else cnt[t.c][t.n]++; }
    return { cnt, J };
  }

  // cfg: {colors, copies}
  function solve({ hand, table, hasMelded, cfg }) {
    cfg = cfg || { colors: 4, copies: 2 };
    const C = cfg.colors;
    // copias efectivas = máximo observado (por si el juego trae más de las configuradas)
    const K = cfg.copies;
    const H = toCounts(hand, C), T = toCounts(table || [], C);
    // lectura imposible: más copias de una ficha de las que existen -> no se calcula (el cálculo se dispararía)
    const over = [];
    for (let c = 0; c < C; c++) for (let n = 1; n <= MAXN; n++) if (H.cnt[c][n] + T.cnt[c][n] > K) over.push({ c, n, count: H.cnt[c][n] + T.cnt[c][n] });
    if (over.length) return { possible: false, played: 0, sets: [], meldMode: !hasMelded, over };
    const conf = { colors: C, copies: K };
    const empty = Array.from({ length: C }, () => Array(MAXN + 1).fill(0));
    const meldMode = !hasMelded;
    const res = meldMode ? solveCounts(H.cnt, empty, H.J, 0, true, conf)
                         : solveCounts(H.cnt, T.cnt, H.J, T.J, false, conf);
    if (!res) return { possible: false, played: 0, sets: [], meldMode, tableInvalid: !meldMode };
    const sets = buildSets(res);
    const played = sets.reduce((s, x) => s + x.tiles.filter(t => t.fromHand).length, 0);
    return { possible: played > 0, played, sets, meldMode,
      points: sets.reduce((s, x) => s + x.tiles.filter(t => t.fromHand).reduce((a, t) => a + (t.joker ? 30 : t.n), 0), 0) };
  }

  function validSet(tiles, C) {
    C = C || 6;
    if (tiles.length < 3) return false;
    const real = tiles.filter(t => !t.joker);
    if (real.length === 0) return true; // solo comodines: raro pero no lo marcamos
    const n0 = real[0].n;
    if (tiles.length <= C && real.every(t => t.n === n0) && new Set(real.map(t => t.c)).size === real.length) return true;
    const c0 = real[0].c;
    if (!real.every(t => t.c === c0)) return false;
    // escalera: los números reales distintos y caben en un tramo de longitud = nº fichas dentro de 1..13
    const nums = real.map(t => t.n).sort((a, b) => a - b);
    for (let i = 1; i < nums.length; i++) if (nums[i] === nums[i - 1]) return false;
    const span = nums[nums.length - 1] - nums[0] + 1;
    if (span > tiles.length) return false;
    return tiles.length <= 13;
  }

  // ---------- Jugada "sencilla": no se rompe ningún set de la mesa ----------
  // Las fichas de la mano se añaden a los extremos de escaleras o a grupos incompletos,
  // y con el resto se forman sets nuevos (DP solo con la mano).
  const tkey = t => t.joker ? 'J' : t.c + '-' + t.n;

  function setInfo(tiles, C) {
    const real = tiles.filter(t => !t.joker);
    if (!real.length) return null;
    const isGroup = tiles.length <= C && real.every(t => t.n === real[0].n) && new Set(real.map(t => t.c)).size === real.length;
    if (isGroup && !(real.length >= 2 && real.every(t => t.c === real[0].c))) {
      return { type: 'grupo', n: real[0].n, colors: new Set(real.map(t => t.c)), size: tiles.length };
    }
    // escalera: deducir inicio
    const nums = real.map(t => t.n).sort((a, b) => a - b);
    let start = Math.max(1, nums[nums.length - 1] - tiles.length + 1);
    start = Math.min(start, nums[0]);
    return { type: 'escalera', c: real[0].c, start, end: start + tiles.length - 1 };
  }

  function canAttach(info, t, C) {
    if (info.type === 'grupo') return !t.joker && t.n === info.n && !info.colors.has(t.c) && info.size < C;
    return !t.joker && t.c === info.c && (t.n === info.start - 1 || t.n === info.end + 1);
  }
  function attach(info, t) {
    if (info.type === 'grupo') { info.colors.add(t.c); info.size++; }
    else if (t.n === info.start - 1) info.start--; else info.end++;
  }

  function solveSimple({ hand, tableSets, cfg }) {
    const C = cfg.colors;
    const attempt = (orderFirst) => {
      const sets = tableSets.map(s => ({ tiles: s.map(t => ({ ...t, fromHand: false })), info: setInfo(s, C), added: [] }));
      let rest = hand.slice();
      const greedyAttach = () => {
        let changed = true;
        while (changed) {
          changed = false;
          for (let i = 0; i < rest.length; i++) {
            const t = rest[i];
            const s = sets.find(x => x.info && canAttach(x.info, t, C));
            if (s) { attach(s.info, t); s.added.push(t); rest.splice(i, 1); changed = true; break; }
          }
        }
      };
      let newSets = [];
      const formNew = () => {
        const r = solve({ hand: rest, table: [], hasMelded: true, cfg });
        newSets = r.sets;
        const used = r.sets.flatMap(s => s.tiles);
        const pool = rest.slice();
        for (const u of used) { const i = pool.findIndex(x => tkey(x) === (u.joker ? 'J' : u.c + '-' + u.n) || (u.joker && x.joker)); if (i >= 0) pool.splice(i, 1); }
        rest = pool;
      };
      if (orderFirst === 'attach') { greedyAttach(); formNew(); greedyAttach(); }
      else { formNew(); greedyAttach(); }
      // comodines sobrantes: a cualquier escalera que pueda crecer o grupo con hueco
      for (let i = rest.length - 1; i >= 0; i--) if (rest[i].joker) {
        const s = sets.find(x => x.info && (x.info.type === 'grupo' ? x.info.size < C : (x.info.start > 1 || x.info.end < 13)));
        if (s) {
          if (s.info.type === 'grupo') s.info.size++; else if (s.info.end < 13) s.info.end++; else s.info.start--;
          s.added.push(rest[i]); rest.splice(i, 1);
        }
      }
      const outSets = [];
      for (const s of sets) {
        if (!s.added.length) { outSets.push({ type: s.info ? s.info.type : 'set', tiles: s.tiles, unchanged: true }); continue; }
        const all = s.tiles.concat(s.added.map(t => ({ ...t, fromHand: true })));
        outSets.push({ type: s.info.type, tiles: orderSet(all, s.info), extended: true });
      }
      for (const s of newSets) outSets.push({ ...s, isNew: true });
      const played = hand.length - rest.length;
      return { played, sets: outSets, points: hand.filter(t => !rest.includes(t)).reduce((a, t) => a + (t.joker ? 30 : t.n), 0) };
    };
    const a = attempt('attach'), b = attempt('new');
    return a.played >= b.played ? a : b;
  }

  function orderSet(tiles, info) {
    if (info.type === 'grupo') return tiles;
    const real = tiles.filter(t => !t.joker), jk = tiles.filter(t => t.joker);
    const out = [];
    for (let n = info.start; n <= info.end; n++) {
      const i = real.findIndex(t => t.n === n);
      if (i >= 0) out.push(real.splice(i, 1)[0]);
      else { const j = jk.shift(); if (j) out.push({ ...j, c: info.c, n }); }
    }
    return out.concat(real, jk);
  }

  // Solución final: máximo de fichas; si la jugada sencilla iguala el máximo, se prefiere.
  function recommend({ hand, tableSets, hasMelded, cfg }) {
    const table = tableSets.flat();
    const full = solve({ hand, table, hasMelded, cfg });
    if (full.over) return { ...full, mode: 'over' };
    if (hasMelded && !full.possible && full.tableInvalid) {
      // mesa mal leída: jugada provisional sin tocar los sets inválidos
      const C = cfg.colors, ok = tableSets.filter(s => validSet(s, C)), bad = tableSets.filter(s => !validSet(s, C));
      const simple = solveSimple({ hand, tableSets: ok, cfg });
      return { possible: simple.played > 0, ...simple, mode: 'provisional', keepSets: bad, tableInvalid: true };
    }
    if (!hasMelded || !full.possible) return { ...full, mode: hasMelded ? 'full' : 'meld' };
    const simple = solveSimple({ hand, tableSets, cfg });
    if (simple.played >= full.played) return { possible: true, ...simple, mode: 'simple' };
    // marcar sets sin cambios en la solución completa
    const sig = s => s.map(t => tkey(t)).sort().join('|');
    const orig = new Map();
    for (const s of tableSets) { const k = sig(s); orig.set(k, (orig.get(k) || 0) + 1); }
    for (const s of full.sets) {
      const k = sig(s.tiles);
      if (!s.tiles.some(t => t.fromHand) && orig.get(k) > 0) { s.unchanged = true; orig.set(k, orig.get(k) - 1); }
    }
    return { ...full, mode: 'full', simplePlayed: simple.played };
  }

  const api = { solve, solveCounts, solveSimple, recommend, validSet, setInfo };
  if (typeof module !== 'undefined') module.exports = api; else root.RummiSolver = api;
})(this);

"""Pipeline de inferencia (referencia para la versión JS). Usa los modelos ONNX."""
import sys, math, json, numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, '.')
from models import crop_tile, CW, CH

COLORS = ['negro', 'rojo', 'azul', 'naranja', 'amarillo', 'verde']
DET_SIZE = 960
THR = 0.35

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

class Pipeline:
    def __init__(s, det='det.onnx', cls='cls.onnx'):
        s.det = ort.InferenceSession(det); s.cls = ort.InferenceSession(cls)

    def detect(s, img, size=DET_SIZE):
        H, W = img.shape[:2]; sc = size / max(H, W)
        w2, h2 = int(math.ceil(W * sc / 32) * 32), int(math.ceil(H * sc / 32) * 32)
        canvas = np.full((h2, w2, 3), 128, np.uint8)
        small = cv2.resize(img, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
        canvas[:small.shape[0], :small.shape[1]] = small
        x = (canvas.astype(np.float32) / 255 - 0.5).transpose(2, 0, 1)[None]
        out = s.det.run(None, {'x': x})[0][0]
        hm = 1 / (1 + np.exp(-out[0]))
        mx = cv2.dilate(hm, np.ones((3, 3), np.uint8))
        ys, xs = np.where((hm >= THR) & (hm >= mx))
        dets = []
        for y, x_ in zip(ys, xs):
            # refinamiento subpíxel
            y0, y1, x0, x1 = max(0, y - 1), min(hm.shape[0], y + 2), max(0, x_ - 1), min(hm.shape[1], x_ + 2)
            patch = hm[y0:y1, x0:x1]; gy, gx = np.mgrid[y0:y1, x0:x1]
            cy = (patch * gy).sum() / patch.sum(); cx = (patch * gx).sum() / patch.sum()
            ux, uy = out[1, y, x_], out[2, y, x_]; n = math.hypot(ux, uy) + 1e-6
            w = math.exp(out[3, y, x_]) * 4
            dets.append(dict(cx=cx * 4 / sc, cy=cy * 4 / sc, ux=ux / n, uy=uy / n, w=w / sc, score=float(hm[y, x_])))
        dets.sort(key=lambda d: -d['score']); keep = []
        for d in dets:
            if not any(math.hypot(k['cx'] - d['cx'], k['cy'] - d['cy']) < 0.5 * max(k['w'], d['w']) for k in keep): keep.append(d)
        return keep

    def classify(s, img, dets):
        if not dets: return dets
        def run(ds):
            X = np.stack([crop_tile(img, d['cx'], d['cy'], d['ux'], d['uy'], d['w']) for d in ds])
            x = (X.astype(np.float32) / 255 - 0.5).transpose(0, 3, 1, 2)
            n, c, f = s.cls.run(None, {'x': x})
            reg = X[:, int(CH * 0.62):int(CH * 0.9), int(CW * 0.15):int(CW * 0.85)].reshape(len(ds), -1, 3).mean(1).astype(np.uint8)
            lab = cv2.cvtColor(reg[None], cv2.COLOR_RGB2LAB)[0].astype(float)
            lab[:, 0] *= 100 / 255; lab[:, 1:] -= 128
            return softmax(n), softmax(c), softmax(f), lab
        n, c, f, lab = run(dets)
        flipped = [i for i in range(len(dets)) if f[i, 1] > 0.5]
        if flipped:
            for i in flipped: dets[i]['ux'] *= -1; dets[i]['uy'] *= -1
            n2, c2, f2, l2 = run([dets[i] for i in flipped])
            for k, i in enumerate(flipped): n[i], c[i], f[i], lab[i] = n2[k], c2[k], f2[k], l2[k]
        out = []
        med = np.median(lab, 0)
        for i, d in enumerate(dets):
            k = int(n[i].argmax())
            if k == 14: continue
            if len(dets) >= 4 and (lab[i, 0] - med[0] < -16 or np.hypot(*(lab[i, 1:] - med[1:])) > 15): continue
            d.update(num=k, color=int(c[i].argmax()), pn=float(n[i].max()), pc=float(c[i].max()), pnum=n[i].tolist(), pcol=c[i].tolist())
            d['unsure'] = bool(d['pn'] < 0.7 or (k != 0 and d['pc'] < 0.7))
            out.append(d)
        return out

def correct_set(g, C=6, max_j=4, margin=10):
    """Corrección por reglas (Vandevelde et al., 2024): lectura válida más probable del set."""
    k = len(g)
    if k < 3: return None
    eps = 1e-4
    LN = [np.log(np.array(t['pnum'][:14]) / max(sum(t['pnum'][:14]), eps) + eps) for t in g]   # 0=comodín, 1..13
    LC = [np.log(np.array(t['pcol'][:C]) + eps) for t in g]
    indep = sum(max(LN[i][1:].max() + LC[i].max(), LN[i][0]) for i in range(k))
    best = (-1e9, None)
    # grupos: mismo número, colores distintos (DP sobre colores usados y comodines)
    if k <= C:
        for n in range(1, 14):
            dp = {(0, 0): (0.0, [])}
            for i in range(k):
                nd = {}
                for (mask, j), (s, asg) in dp.items():
                    if j < max_j:
                        key = (mask, j + 1); v = s + LN[i][0]
                        if key not in nd or nd[key][0] < v: nd[key] = (v, asg + [(0, 0)])
                    for c in range(C):
                        if mask >> c & 1: continue
                        key = (mask | 1 << c, j); v = s + LN[i][n] + LC[i][c]
                        if key not in nd or nd[key][0] < v: nd[key] = (v, asg + [(n, c)])
                dp = nd
            for (mask, j), (s, asg) in dp.items():
                if mask and s > best[0]: best = (s, asg)
    # escaleras: un color, consecutivos en el orden del set (o al revés)
    for c in range(C):
        for rev in (False, True):
            for s0 in range(1, 15 - k):
                vals = [s0 + (k - 1 - i if rev else i) for i in range(k)]
                real = [LN[i][vals[i]] + LC[i][c] for i in range(k)]
                delta = sorted(((LN[i][0] - real[i], i) for i in range(k)), reverse=True)
                jokers = {i for dlt, i in delta[:max_j] if dlt > 0}
                if len(jokers) == k: jokers.discard(delta[-1][1])
                tot = sum(LN[i][0] if i in jokers else real[i] for i in range(k))
                if tot > best[0]: best = (tot, [(0, 0) if i in jokers else (vals[i], c) for i in range(k)])
    if best[1] is None or best[0] < indep - margin: return None
    return best[1]

def apply_corrections(sets, C=6):
    changed = 0
    for g in sets:
        asg = correct_set(g, C)
        if asg is None:
            for t in g: t['invalid'] = len(g) >= 3
            continue
        for t, (n, c) in zip(g, asg):
            if t['num'] != n or (n != 0 and t['color'] != c):
                changed += 1; t['corrected'] = True; t['unsure'] = True
            t['num'], t['color'] = n, c
    return changed

def group_sets(tiles):
    n = len(tiles); parent = list(range(n))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tiles[i], tiles[j]
            wm = (a['w'] + b['w']) / 2
            dx, dy = b['cx'] - a['cx'], b['cy'] - a['cy']
            # vecino a lo largo del eje "derecha" de la ficha (lado largo con lado largo)
            rx, ry = -a['uy'], a['ux']
            along = abs(dx * rx + dy * ry); across = abs(dx * a['ux'] + dy * a['uy'])
            same_dir = a['ux'] * b['ux'] + a['uy'] * b['uy'] > 0.7
            if same_dir and along < wm * 1.45 and across < wm * 0.55: parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(tiles[i])
    sets = []
    for g in groups.values():
        rx, ry = -g[0]['uy'], g[0]['ux']
        g.sort(key=lambda t: t['cx'] * rx + t['cy'] * ry)
        sets.append(g)
    return sets

def rack_filter(tiles):
    if not tiles: return tiles
    ref = max(tiles, key=lambda t: t['w']); h = ref['w'] * 1.4
    r = [t for t in tiles if t['w'] >= 0.7 * ref['w'] and t['uy'] < -0.7 and abs(t['cy'] - ref['cy']) < 1.5 * h]
    return sorted(r, key=lambda t: t['cx'])

def fmt(t):
    return '☺' if t['num'] == 0 else f"{t['num']}{COLORS[t['color']][:2]}" + ('?' if t['unsure'] else '')

if __name__ == '__main__':
    P = Pipeline()
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
    size = int(sys.argv[2]) if len(sys.argv) > 2 else DET_SIZE
    dets = P.classify(img, P.detect(img, size))
    sets = group_sets(dets)
    print(len(dets), 'fichas,', len(sets), 'sets')
    for s_ in sorted(sets, key=lambda g: (g[0]['cy'], g[0]['cx'])): print('  ', ' '.join(fmt(t) for t in s_))
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for t in dets:
        c = (0, 200, 255) if t['unsure'] else (0, 220, 0)
        cv2.circle(vis, (int(t['cx']), int(t['cy'])), 5, c, -1)
        cv2.line(vis, (int(t['cx']), int(t['cy'])), (int(t['cx'] + t['ux'] * t['w'] * 0.6), int(t['cy'] + t['uy'] * t['w'] * 0.6)), (255, 0, 255), 2)
        cv2.putText(vis, fmt(t).replace('☺', 'J'), (int(t['cx']) + 6, int(t['cy']) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, fmt(t).replace('☺', 'J'), (int(t['cx']) + 6, int(t['cy']) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1)
    cv2.imwrite(sys.argv[3] if len(sys.argv) > 3 else 'pred.png', vis)

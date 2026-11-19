"""Generador de escenas sintéticas de Rummikub (6 colores) con etiquetas por ficha:
centro (x,y), dirección 'arriba' (ángulo), anchura de ficha, número (0=comodín) y color."""
import math, random, os, sys, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = [f for f in [
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf',
    '/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
] if os.path.exists(f)]

# negro, rojo, azul, naranja, amarillo, verde (RGB)
PALETTE = [(28, 28, 30), (205, 35, 48), (28, 92, 196), (236, 118, 28), (232, 186, 12), (34, 142, 72)]
TW, TH = 120, 168  # ficha base en px (vertical)

def jit(c, s=18):
    return tuple(int(np.clip(v + random.gauss(0, s), 0, 255)) for v in c)

_font_cache = {}
def font(path, size):
    k = (path, size)
    if k not in _font_cache: _font_cache[k] = ImageFont.truetype(path, size)
    return _font_cache[k]

def render_tile(num, color, joker_style=None):
    """Devuelve RGBA TWxTH de la cara superior de la ficha, derecha."""
    base = jit((240, 235, 220), 8)
    im = Image.new('RGBA', (TW, TH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = random.randint(8, 16)
    d.rounded_rectangle([0, 0, TW - 1, TH - 1], r, fill=base + (255,))
    # leve degradado
    g = np.linspace(random.uniform(-10, 4), random.uniform(-4, 10), TH)[:, None, None]
    arr = np.array(im).astype(np.float32); arr[..., :3] += g; im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # hueco redondo inferior
    cy = TH * random.uniform(0.68, 0.74); cx = TW / 2; rr = TW * random.uniform(0.2, 0.25)
    ring = tuple(max(0, v - random.randint(10, 28)) for v in base)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=tuple(min(255, v + 4) for v in base), outline=ring, width=random.randint(2, 4))
    col = jit(PALETTE[color], 14)
    if num == 0:
        # comodín: cara dentro de un círculo
        jx, jy, jr = TW / 2, TH * 0.32, TW * random.uniform(0.3, 0.36)
        w = random.randint(3, 5)
        d.ellipse([jx - jr, jy - jr, jx + jr, jy + jr], outline=col, width=w)
        fr = jr * 0.62
        d.ellipse([jx - fr, jy - fr * 0.9, jx + fr, jy + fr], outline=col, width=w - 1)
        for sx in (-1, 1):
            d.ellipse([jx + sx * fr * 0.4 - 3, jy - fr * 0.25 - 3, jx + sx * fr * 0.4 + 3, jy - fr * 0.25 + 3], fill=col)
        d.arc([jx - fr * 0.55, jy - fr * 0.2, jx + fr * 0.55, jy + fr * 0.6], 20, 160, fill=col, width=w - 1)
        for k in range(random.randint(3, 6)):  # pelo
            a = random.uniform(200, 340)
            d.arc([jx - jr * 0.9, jy - jr * 0.95, jx + jr * 0.9, jy + jr * 0.6], a, a + 25, fill=col, width=2)
    else:
        txt = str(num)
        fp = random.choice(FONTS)
        size = int(TH * (random.uniform(0.40, 0.48) if num < 10 else random.uniform(0.34, 0.40)))
        f = font(fp, size)
        bb = d.textbbox((0, 0), txt, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        if tw > TW * 0.86:
            size = int(size * TW * 0.86 / tw); f = font(fp, size); bb = d.textbbox((0, 0), txt, font=f); tw, th = bb[2] - bb[0], bb[3] - bb[1]
        tx = (TW - tw) / 2 - bb[0] + random.uniform(-3, 3)
        ty = TH * random.uniform(0.28, 0.33) - th / 2 - bb[1]
        d.text((tx, ty), txt, font=f, fill=col)
    return im

def background(S):
    kind = random.random()
    if kind < 0.55:  # mantel gris claro con ondas (como el real)
        base = np.array(jit((200, 200, 198), 22), np.float32)
        y, x = np.mgrid[0:S, 0:S].astype(np.float32)
        f1, f2 = random.uniform(0.03, 0.12), random.uniform(0.005, 0.03)
        ph = np.sin(x * f2 + random.uniform(0, 6)) * random.uniform(10, 60)
        waves = np.sin((y + ph) * f1 + np.sin(x * f2 * 3) * 2)
        img = base[None, None, :] + (waves * random.uniform(4, 14))[..., None]
    elif kind < 0.75:  # madera
        base = np.array(jit((150, 105, 70), 30), np.float32)
        y, x = np.mgrid[0:S, 0:S].astype(np.float32)
        grain = np.sin(x * random.uniform(0.05, 0.2) + np.sin(y * 0.02) * 4)
        img = base[None, None, :] + (grain * 12)[..., None]
    else:  # liso de color
        g0 = random.randint(60, 235); img = np.ones((S, S, 3), np.float32) * np.array(jit((g0,) * 3, 12), np.float32)
    img += np.random.normal(0, random.uniform(2, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # distractores: barras azules (atriles), móviles, círculos
    for _ in range(random.randint(0, 4)):
        t = random.random(); c = jit((30, 70, 170), 25) if t < 0.5 else jit((30, 30, 35), 15)
        cx, cy = random.randint(0, S), random.randint(0, S)
        if t < 0.7:
            w, h, a = random.randint(80, 400), random.randint(15, 60), random.uniform(0, 180)
            box = cv2.boxPoints(((cx, cy), (w, h), a)).astype(np.int32)
            cv2.fillPoly(img, [box], c)
        else:
            cv2.circle(img, (cx, cy), random.randint(15, 60), c, random.randint(2, 6))
    return img

def random_set():
    """Devuelve lista de (num, color) de un set (válido o casi)."""
    if random.random() < 0.5:
        c = random.randrange(6); L = random.randint(3, 6); s = random.randint(1, 14 - L)
        tiles = [(s + i, c) for i in range(L)]
    else:
        n = random.randint(1, 13); k = random.randint(3, 6)
        tiles = [(n, c) for c in random.sample(range(6), k)]
    if random.random() < 0.15:
        i = random.randrange(len(tiles)); tiles[i] = (0, random.choice([0, 1]))
    return tiles

def paste_tile(canvas, tile_rgba, cx, cy, scale, ang_deg, thick):
    """Pega la ficha rotada; ang_deg = rotación de la ficha (0 = derecha). Devuelve vector 'arriba'."""
    t = tile_rgba.resize((max(4, int(TW * scale)), max(4, int(TH * scale))), Image.BILINEAR)
    rot = t.rotate(ang_deg, expand=True, resample=Image.BILINEAR)
    # canto/sombra (grosor) hacia abajo
    a = np.array(rot)
    alpha = a[..., 3]
    w, h = rot.size
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    H, W = canvas.shape[:2]
    def blend(img_rgb, al, ox, oy):
        xs, ys = max(0, ox), max(0, oy); xe, ye = min(W, ox + w), min(H, oy + h)
        if xe <= xs or ye <= ys: return
        sub = img_rgb[ys - oy:ye - oy, xs - ox:xe - ox].astype(np.float32)
        al2 = (al[ys - oy:ye - oy, xs - ox:xe - ox].astype(np.float32) / 255)[..., None]
        canvas[ys:ye, xs:xe] = (canvas[ys:ye, xs:xe] * (1 - al2) + sub * al2).astype(np.uint8)
    shadow = np.zeros_like(a[..., :3]); shadow[:] = (60, 60, 60)
    blend(shadow, (alpha * 0.35).astype(np.uint8), x0 + thick, y0 + thick * 2)
    side = np.zeros_like(a[..., :3]); side[:] = jit((200, 192, 172), 8)
    for k in range(thick, 0, -1): blend(side, alpha, x0, y0 + k)
    blend(a[..., :3], alpha, x0, y0)
    th = math.radians(ang_deg)
    return (-math.sin(th), -math.cos(th))  # vector arriba en imagen (y hacia abajo)

def make_scene(S=512):
    img = background(S)
    labels = []  # cx, cy, upx, upy, width, num, color
    occupied = np.zeros((S, S), bool)
    mode = random.random()
    tw_px = random.uniform(16, 70) if mode < 0.75 else random.uniform(40, 110)
    scale = tw_px / TW
    nsets = random.randint(2, 12) if tw_px < 50 else random.randint(1, 5)
    rack = random.random() < 0.35
    if rack:  # fila de fichas de pie en un atril abajo
        ry = random.uniform(0.6, 0.9) * S
        n = random.randint(4, 16)
        sc = scale * random.uniform(1.0, 1.4)
        x = random.uniform(0, S * 0.2)
        cv2.rectangle(img, (0, int(ry + TH * sc * 0.45)), (S, int(ry + TH * sc * 0.45 + random.randint(10, 40))), jit((30, 70, 170), 20), -1)
        tilt = random.uniform(-6, 6)
        for i in range(n):
            num, col = (0, random.choice([0, 1])) if random.random() < 0.05 else (random.randint(1, 13), random.randrange(6))
            cx = x + TW * sc * (0.5 + i * random.uniform(1.0, 1.08)); cy = ry + math.sin(math.radians(tilt)) * (cx - S / 2)
            if cx > S - 5: break
            up = paste_tile(img, render_tile(num, col), cx, cy, sc, tilt + random.uniform(-3, 3), max(1, int(3 * sc)))
            labels.append([cx, cy, up[0], up[1], TW * sc, num, col])
    for _ in range(nsets):
        tiles = random_set()
        ang = random.choice([0, 90, 180, 270]) + random.gauss(0, 12) if random.random() < 0.8 else random.uniform(0, 360)
        th = math.radians(ang)
        ex = (math.cos(th), -math.sin(th))  # eje x de la ficha en imagen
        gap = random.uniform(1.0, 1.12)
        L = len(tiles) * TW * scale * gap
        for attempt in range(15):
            sx, sy = random.uniform(0, S), random.uniform(0, S)
            cxs = [sx + ex[0] * TW * scale * gap * i for i in range(len(tiles))]
            cys = [sy + ex[1] * TW * scale * gap * i for i in range(len(tiles))]
            m = TH * scale * 0.6
            if min(cxs) < -m * 0.3 or max(cxs) > S + m * 0.3 or min(cys) < -m * 0.3 or max(cys) > S + m * 0.3: continue
            ok = all(not occupied[int(np.clip(cy, 0, S - 1)), int(np.clip(cx, 0, S - 1))] for cx, cy in zip(cxs, cys))
            if not ok: continue
            stag = random.uniform(0, 0.08)
            for (num, col), cx, cy in zip(tiles, cxs, cys):
                cx += random.gauss(0, TW * scale * stag); cy += random.gauss(0, TW * scale * stag)
                a2 = ang + random.gauss(0, 3)
                up = paste_tile(img, render_tile(num, col), cx, cy, scale, a2, max(1, int(5 * scale)))
                labels.append([cx, cy, up[0], up[1], TW * scale, num, col])
                cv2.circle(occupied.view(np.uint8), (int(cx), int(cy)), int(TH * scale * 0.75), 1, -1)
            break
    # perspectiva global
    if random.random() < 0.8:
        d = S * random.uniform(0.02, 0.18)
        src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        top = random.uniform(0, d)
        dst = np.float32([[top + random.uniform(-d * .3, d * .3), random.uniform(0, d * .5)], [S - top + random.uniform(-d * .3, d * .3), random.uniform(0, d * .5)],
                          [S + random.uniform(-d * .3, d * .3), S], [random.uniform(-d * .3, d * .3), S]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, Hm, (S, S), borderMode=cv2.BORDER_REFLECT)
        new = []
        for cx, cy, ux, uy, w, num, col in labels:
            p = cv2.perspectiveTransform(np.float32([[[cx, cy], [cx + ux * w * 0.5, cy + uy * w * 0.5], [cx + w * 0.5, cy]]]), Hm)[0]
            v = p[1] - p[0]; n = np.linalg.norm(v) + 1e-6
            new.append([p[0][0], p[0][1], v[0] / n, v[1] / n, n + np.linalg.norm(p[2] - p[0]), num, col])
        labels = new
    # fotometría
    imgf = img.astype(np.float32)
    gx = np.linspace(random.uniform(0.75, 1.1), random.uniform(0.9, 1.2), S)[None, :, None]
    gy = np.linspace(random.uniform(0.8, 1.1), random.uniform(0.9, 1.2), S)[:, None, None]
    imgf *= gx * gy
    imgf *= np.array([random.uniform(0.94, 1.06), random.uniform(0.97, 1.03), random.uniform(0.9, 1.06)])[None, None, :]
    imgf += np.random.normal(0, random.uniform(1, 6), imgf.shape)
    img = np.clip(imgf, 0, 255).astype(np.uint8)
    if random.random() < 0.6:
        k = random.choice([3, 5]); img = cv2.GaussianBlur(img, (k, k), random.uniform(0.3, 1.4))
    if random.random() < 0.3:  # desenfoque de movimiento
        k = random.randint(3, 7); ker = np.zeros((k, k)); ker[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), random.uniform(0, 180), 1); ker = cv2.warpAffine(ker, M, (k, k)); ker /= ker.sum() + 1e-6
        img = cv2.filter2D(img, -1, ker)
    q = random.randint(45, 95)
    img = cv2.imdecode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    labels = [l for l in labels if 0 <= l[0] < S and 0 <= l[1] < S]
    return img, np.array(labels, np.float32).reshape(-1, 7)

if __name__ == '__main__':
    out = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
    random.seed(seed); np.random.seed(seed)
    imgs, labs = [], []
    for i in range(n):
        im, lb = make_scene()
        imgs.append(im); labs.append(lb)
    np.savez_compressed(out, imgs=np.stack(imgs), labs=np.array(labs, dtype=object), allow_pickle=True)
    print('ok', out, n)

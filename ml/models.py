import math, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

def cbr(i, o, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, d, dilation=d, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class Det(nn.Module):
    """CenterNet mínima: salida a stride 4 -> [heatmap, up_x, up_y, log_w]"""
    def __init__(s):
        super().__init__()
        s.e1 = nn.Sequential(cbr(3, 16, 2), cbr(16, 16))            # /2
        s.e2 = nn.Sequential(cbr(16, 32, 2), cbr(32, 32))           # /4
        s.e3 = nn.Sequential(cbr(32, 64, 2), cbr(64, 64))           # /8
        s.e4 = nn.Sequential(cbr(64, 96, 2), cbr(96, 96), cbr(96, 96, d=2))   # /16
        s.e5 = nn.Sequential(cbr(96, 128, 2), cbr(128, 128, d=2))   # /32
        s.d4 = cbr(128 + 96, 96); s.d3 = cbr(96 + 64, 64); s.d2 = cbr(64 + 32, 48)
        s.head = nn.Sequential(cbr(48, 48), nn.Conv2d(48, 4, 1))
        s.head[-1].bias.data[0] = -2.2
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(e1); e3 = s.e3(e2); e4 = s.e4(e3); e5 = s.e5(e4)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False)
        d4 = s.d4(torch.cat([up(e5, e4), e4], 1))
        d3 = s.d3(torch.cat([up(d4, e3), e3], 1))
        d2 = s.d2(torch.cat([up(d3, e2), e2], 1))
        return s.head(d2)

CW, CH = 48, 64     # recorte de ficha enderezada
MARGIN = 1.12
NUM_CLASSES = 15    # 0 = comodín, 1..13, 14 = no es ficha

class Cls(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(cbr(3, 32), nn.MaxPool2d(2), cbr(32, 48), nn.MaxPool2d(2), cbr(48, 64), nn.MaxPool2d(2),
                            cbr(64, 96), nn.MaxPool2d(2), cbr(96, 96))   # 3x4
        s.fc = nn.Sequential(nn.Flatten(), nn.Linear(96 * 3 * 4, 192), nn.ReLU(inplace=True), nn.Dropout(0.2))
        s.num = nn.Linear(192, NUM_CLASSES); s.col = nn.Linear(192, 6); s.flip = nn.Linear(192, 2)
    def forward(s, x):
        h = s.fc(s.f(x))
        return s.num(h), s.col(h), s.flip(h)

def crop_tile(img, cx, cy, ux, uy, w):
    """Recorta la ficha enderezada (arriba = hacia arriba) a CWxCH."""
    h = w * 1.4
    rx, ry = -uy, ux          # vector derecha
    dx, dy = -ux, -uy         # vector abajo
    sw, sh = w * MARGIN, h * MARGIN
    tl = (cx - rx * sw / 2 - dx * sh / 2, cy - ry * sw / 2 - dy * sh / 2)
    tr = (tl[0] + rx * sw, tl[1] + ry * sw)
    bl = (tl[0] + dx * sh, tl[1] + dy * sh)
    M = cv2.getAffineTransform(np.float32([tl, tr, bl]), np.float32([[0, 0], [CW, 0], [0, CH]]))
    return cv2.warpAffine(img, M, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def to_tensor(imgs):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).float().permute(0, 3, 1, 2) / 255.0 - 0.5
    return x

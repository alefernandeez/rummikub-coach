import sys, time, math, random, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from models import Det, to_tensor
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
R = 4

def load(paths):
    L, chunks = [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as d: L += list(d['labs'])
    I = None; k = 0
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            a = d['imgs']
            if I is None: I = np.empty((len(L),) + a.shape[1:], np.uint8)
            I[k:k + len(a)] = a; k += len(a); del a
    return I, L

def rot90(img, lab, k):
    S = img.shape[0]
    img = np.rot90(img, k).copy(); lab = lab.copy()
    for _ in range(k):  # rotación 90° antihoraria: (x,y)->(y, S-1-x); vector (ux,uy)->(uy,-ux)
        x, y, ux, uy = lab[:, 0].copy(), lab[:, 1].copy(), lab[:, 2].copy(), lab[:, 3].copy()
        lab[:, 0], lab[:, 1], lab[:, 2], lab[:, 3] = y, S - 1 - x, uy, -ux
    return img, lab

def targets(lab, S):
    O = S // R
    hm = np.zeros((O, O), np.float32); reg = np.zeros((3, O, O), np.float32); msk = np.zeros((O, O), np.float32)
    yy, xx = np.mgrid[0:O, 0:O]
    for cx, cy, ux, uy, w, num, col in lab:
        x, y = cx / R, cy / R; sg = max(0.8, (w / R) * 0.22)
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sg * sg))
        hm = np.maximum(hm, g)
        m = g > 0.5
        reg[0][m] = ux; reg[1][m] = uy; reg[2][m] = math.log(w / R)
        msk = np.maximum(msk, m.astype(np.float32))
        ix, iy = int(np.clip(round(x), 0, O - 1)), int(np.clip(round(y), 0, O - 1))
        hm[iy, ix] = 1.0
    return hm, reg, msk

def focal(pred, gt):
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float(); neg = 1 - pos
    pl = torch.log(p) * (1 - p) ** 2 * pos
    nl = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pl.sum() + nl.sum()) / n

def batch(imgs, labs, idx, aug=True):
    X, H, G, M = [], [], [], []
    for i in idx:
        im, lb = imgs[i], labs[i].reshape(-1, 7)
        if aug: im, lb = rot90(im, lb, random.randrange(4))
        hm, reg, msk = targets(lb, im.shape[0])
        X.append(im); H.append(hm); G.append(reg); M.append(msk)
    return to_tensor(np.stack(X)), torch.from_numpy(np.stack(H)), torch.from_numpy(np.stack(G)), torch.from_numpy(np.stack(M))

def loss_fn(out, H, G, M):
    lh = focal(out[:, 0], H)
    up = out[:, 1:3]; up = up / (up.norm(dim=1, keepdim=True) + 1e-6)
    m = M.unsqueeze(1)
    lu = ((up - G[:, :2]).abs() * m).sum() / (m.sum() * 2 + 1)
    lw = ((out[:, 3:4] - G[:, 2:3]).abs() * m).sum() / (m.sum() + 1)
    return lh + 2 * lu + lw, (lh.item(), lu.item(), lw.item())

if __name__ == '__main__':
    import pickle, cv2
    LB = pickle.load(open('data/jpg/labels.pkl', 'rb'))
    class Lazy:
        def __init__(s, names): s.n = names
        def __len__(s): return len(s.n)
        def __getitem__(s, i): return cv2.cvtColor(cv2.imread('data/jpg/' + s.n[i]), cv2.COLOR_BGR2RGB)
    tn = sorted(k for k in LB if not k.startswith('val')); vn = sorted(k for k in LB if k.startswith('val'))
    imgs, labs = Lazy(tn), [LB[k] for k in tn]
    vimgs, vlabs = Lazy(vn), [LB[k] for k in vn]
    print('train', len(imgs), 'val', len(vimgs), flush=True)
    net = Det(); print('params', sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    BS = 8; steps = EPOCHS * (len(imgs) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    best = 1e9
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(len(imgs)); t0 = time.time(); acc = np.zeros(3)
        for b in range(len(imgs) // BS):
            X, H, G, M = batch(imgs, labs, perm[b * BS:(b + 1) * BS])
            out = net(X); l, parts = loss_fn(out, H, G, M)
            opt.zero_grad(); l.backward(); opt.step(); sched.step(); acc += parts
            if b % 50 == 0: print(f'ep{ep} b{b} {acc / (b + 1)} {time.time() - t0:.0f}s', flush=True)
        net.eval(); vl = 0
        with torch.no_grad():
            for b in range(0, len(vimgs), 16):
                X, H, G, M = batch(vimgs, vlabs, range(b, min(b + 16, len(vimgs))), aug=False)
                vl += loss_fn(net(X), H, G, M)[0].item()
        print(f'EPOCH {ep} val {vl:.3f} time {time.time() - t0:.0f}s', flush=True)
        if vl < best: best = vl; torch.save(net.state_dict(), 'det.pt')

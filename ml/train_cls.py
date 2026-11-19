import sys, time, math, random, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '.')
from models import Cls, crop_tile, to_tensor, CW, CH
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def make_crops(paths, per_bg=4, jitter=True, seed=0):
    rng = random.Random(seed)
    cap = 0
    for p in paths:
        with np.load(p, allow_pickle=True) as d: cap += sum(len(l.reshape(-1, 7)) for l in d['labs']) * 1 + len(d['labs']) * per_bg
    X = np.empty((cap, CH, CW, 3), np.uint8); N, Cc, Fl = [], [], []
    k = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        for im, lb in zip(d['imgs'], d['labs']):
            lb = lb.reshape(-1, 7)
            for cx, cy, ux, uy, w, num, col in lb:
                for rep in range(1):
                    a = math.atan2(uy, ux) + (rng.gauss(0, 0.12) if jitter else 0)
                    flip = jitter and rng.random() < 0.2
                    if flip: a += math.pi
                    ww = w * (math.exp(rng.gauss(0, 0.12)) if jitter else 1)
                    jx = rng.gauss(0, 0.08 * w) if jitter else 0; jy = rng.gauss(0, 0.08 * w) if jitter else 0
                    X[k] = crop_tile(im, cx + jx, cy + jy, math.cos(a), math.sin(a), ww); k += 1; N.append(int(num)); Cc.append(int(col)); Fl.append(int(flip))
            # negativos: zonas sin ficha o entre fichas
            S = im.shape[0]
            wm = float(np.median(lb[:, 4])) if len(lb) else 40
            for _ in range(per_bg):
                cx, cy = rng.uniform(0, S), rng.uniform(0, S)
                if len(lb) and np.min(np.hypot(lb[:, 0] - cx, lb[:, 1] - cy)) < wm * 0.45: continue
                a = rng.uniform(0, 2 * math.pi)
                X[k] = crop_tile(im, cx, cy, math.cos(a), math.sin(a), wm * math.exp(rng.gauss(0, 0.3))); k += 1; N.append(14); Cc.append(-1); Fl.append(-1)
        del d
    return X[:k], np.array(N), np.array(Cc), np.array(Fl)

if __name__ == '__main__':
    X, N, Cc, Fl = make_crops(sorted(glob.glob('data/det_*.npz')))
    vX, vN, vC, vF = make_crops(['data/val.npz'], jitter=True, seed=1)
    print('crops', len(X), 'val', len(vX), np.bincount(N), flush=True)
    net = Cls()
    import os
    if os.path.exists('cls.pt'): net.load_state_dict(torch.load('cls.pt')); print('resume', flush=True)
    print('params', sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    BS = 128; steps = EPOCHS * (len(X) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    def run(idx, Xs, Ns, Cs, Fs, train):
        x = to_tensor(Xs[idx])
        if train:  # aumento fotométrico
            x = x * torch.empty(len(idx), 1, 1, 1).uniform_(0.75, 1.25) + torch.empty(len(idx), 3, 1, 1).uniform_(-0.08, 0.08)
        n, c, f = net(x)
        tn = torch.from_numpy(Ns[idx]); tc = torch.from_numpy(Cs[idx]); tf = torch.from_numpy(Fs[idx])
        l = F.cross_entropy(n, tn) + F.cross_entropy(c, tc, ignore_index=-1) + 0.5 * F.cross_entropy(f, tf, ignore_index=-1)
        return l, n, c, f
    best = 0.919
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(len(X)); t0 = time.time()
        for b in range(len(X) // BS):
            l, *_ = run(perm[b * BS:(b + 1) * BS], X, N, Cc, Fl, True)
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
        net.eval(); cn = cc = cf = tot = totc = totf = 0
        with torch.no_grad():
            for b in range(0, len(vX), 512):
                idx = np.arange(b, min(b + 512, len(vX)))
                _, n, c, f = run(idx, vX, vN, vC, vF, False)
                cn += (n.argmax(1).numpy() == vN[idx]).sum(); tot += len(idx)
                m = vC[idx] >= 0; cc += (c.argmax(1).numpy()[m] == vC[idx][m]).sum(); totc += m.sum()
                m2 = vF[idx] >= 0; cf += (f.argmax(1).numpy()[m2] == vF[idx][m2]).sum(); totf += m2.sum()
        accn = cn / tot
        print(f'EPOCH {ep} num {accn:.4f} col {cc / totc:.4f} flip {cf / totf:.4f} {time.time() - t0:.0f}s', flush=True)
        if accn > best: best = accn; torch.save(net.state_dict(), 'cls.pt')

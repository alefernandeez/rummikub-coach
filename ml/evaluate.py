"""Evaluación de extremo a extremo en escenas sintéticas de validación."""
import sys, pickle, numpy as np, cv2
sys.path.insert(0, '.')
from infer import Pipeline

P = Pipeline()
LB = pickle.load(open('data/jpg/labels.pkl', 'rb'))
names = sorted(k for k in LB if k.startswith('val'))
tp = fp = fn = ok_num = ok_all = ok_up = 0
for n in names:
    img = cv2.cvtColor(cv2.imread('data/jpg/' + n), cv2.COLOR_BGR2RGB)
    gt = LB[n]
    dets = P.classify(img, P.detect(img, 512))
    used = set()
    for d in dets:
        dist = np.hypot(gt[:, 0] - d['cx'], gt[:, 1] - d['cy']) / gt[:, 4]
        j = int(np.argmin(dist)) if len(gt) else -1
        if j >= 0 and dist[j] < 0.5 and j not in used:
            used.add(j); tp += 1
            g = gt[j]
            ok_up += (d['ux'] * g[2] + d['uy'] * g[3]) > 0.7
            if d['num'] == int(g[5]):
                ok_num += 1
                if d['num'] == 0 or d['color'] == int(g[6]): ok_all += 1
        else:
            fp += 1
    fn += len(gt) - len(used)
print(f'fichas reales {tp + fn}: detectadas {tp / (tp + fn):.3f}, falsos positivos {fp}')
print(f'de las detectadas: orientación {ok_up / tp:.3f}, número {ok_num / tp:.3f}, número+color {ok_all / tp:.3f}')

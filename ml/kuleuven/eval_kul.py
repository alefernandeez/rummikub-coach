import sys, json, cv2, numpy as np, time
sys.path.insert(0,'/home/claude/rummi/ml')
from infer import Pipeline, group_sets, apply_corrections
CMAP={'Black':[0],'Red':[1],'Blue':[2],'Orange':[3,4]}
A=json.load(open('rummikub.json')); M=json.load(open('manifest.json'))
det_onnx=sys.argv[1] if len(sys.argv)>1 and sys.argv[1] else '/home/claude/rummi/ml/det.onnx'
cls_onnx=sys.argv[2] if len(sys.argv)>2 and sys.argv[2] else '/home/claude/rummi/ml/cls.onnx'
split=sys.argv[3] if len(sys.argv)>3 else 'all'
P=Pipeline(det_onnx,cls_onnx)
names=sorted(v['filename'] for v in A.values() if v['filename'] in M)
if split=='test': names=json.load(open('ft/test_names.json'))
tp=fn=fp=0; num_ok=col_ok=all_ok=all_ok_c=0; t0=time.time()
for fn_ in names:
    v=[x for x in A.values() if x['filename']==fn_][0]; m=M[fn_]; sc=m['scale']
    img=cv2.cvtColor(cv2.imread('images/'+fn_),cv2.COLOR_BGR2RGB)
    gts=[]
    for r in v['regions']:
        s=r['shape_attributes']; a=r['region_attributes']
        n=0 if a['number']=='J' else int(a['number'])
        gts.append((s['x']*sc,s['y']*sc,s['width']*sc,s['height']*sc,n,a['color']))
    dets=P.classify(img,P.detect(img,960))
    sets=[g for g in group_sets(dets)]
    import copy; cdets=copy.deepcopy(sets); apply_corrections(cdets); cflat={(round(t['cx']),round(t['cy'])):t for g in cdets for t in g}
    used=set()
    for d in dets:
        hit=None
        for j,(x,y,w,h,n,c) in enumerate(gts):
            if j in used: continue
            if x<=d['cx']<=x+w and y<=d['cy']<=y+h: hit=j; break
        if hit is None: fp+=1; continue
        used.add(hit); tp+=1
        x,y,w,h,n,c=gts[hit]
        nok = d['num']==n; cok = n==0 or d['color'] in CMAP[c]
        num_ok+=nok; col_ok+=cok; all_ok+=nok and cok
        dc=cflat.get((round(d['cx']),round(d['cy'])),d)
        all_ok_c+= dc['num']==n and (n==0 or dc['color'] in CMAP[c])
    fn+=len(gts)-len(used)
print(f'{split}: {len(names)} fotos, {tp+fn} fichas | detectadas {tp/(tp+fn):.3f} | falsos pos {fp} | número {num_ok/tp:.3f} | color {col_ok/tp:.3f} | ficha completa {all_ok/tp:.3f} | con corrección {all_ok_c/tp:.3f} | {time.time()-t0:.0f}s')

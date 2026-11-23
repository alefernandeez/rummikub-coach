import sys, json, math, cv2, numpy as np, pickle
sys.path.insert(0,'/home/claude/rummi/ml')
from infer import Pipeline, softmax
from models import crop_tile, CW, CH
A=json.load(open('rummikub.json')); M=json.load(open('manifest.json'))
P=Pipeline('/home/claude/rummi/ml/det.onnx','/home/claude/rummi/ml/cls.onnx')
names=sorted(v['filename'] for v in A.values() if v['filename'] in M)
byname={v['filename']:v for v in A.values()}
CI={'Black':0,'Red':1,'Blue':2,'Orange':3}
out={}; stats=dict(det=0,box=0,low=0)
def probs(img, cands):
    X=np.stack([crop_tile(img,*c) for c in cands]); x=(X.astype(np.float32)/255-0.5).transpose(0,3,1,2)
    n,c,f=P.cls.run(None,{'x':x}); return softmax(n)
for k,fn in enumerate(names):
    v=byname[fn]; sc=M[fn]['scale']
    img=cv2.cvtColor(cv2.imread('images/'+fn),cv2.COLOR_BGR2RGB)
    dets=P.detect(img,960)
    tiles=[]
    for r in v['regions']:
        s=r['shape_attributes']; a=r['region_attributes']
        x,y,w,h=s['x']*sc,s['y']*sc,s['width']*sc,s['height']*sc
        n=0 if a['number']=='J' else int(a['number'])
        inside=[d for d in dets if x<=d['cx']<=x+w and y<=d['cy']<=y+h]
        if inside:
            d=min(inside,key=lambda d:math.hypot(d['cx']-(x+w/2),d['cy']-(y+h/2)))
            cx,cy,tw=d['cx'],d['cy'],d['w']; base=math.atan2(d['uy'],d['ux']); stats['det']+=1
        else:
            cx,cy,tw=x+w/2,y+h/2,min(w,h)*0.85; base=-math.pi/2; stats['box']+=1
        cands=[(cx,cy,math.cos(base+q*math.pi/2),math.sin(base+q*math.pi/2),tw) for q in range(4)]
        pn=probs(img,cands)[:,n]; q=int(pn.argmax())
        if pn[q]<0.2: stats['low']+=1
        c=cands[q]
        tiles.append(dict(cx=c[0],cy=c[1],ux=c[2],uy=c[3],w=c[4],num=n,color=CI[a['color']],conf=float(pn[q])))
    out[fn]=tiles
    if k%50==0: print(k,stats,flush=True)
pickle.dump(out,open('labels_oriented.pkl','wb')); print('fin',stats)

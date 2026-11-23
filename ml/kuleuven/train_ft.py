import sys, json, math, random, pickle, time, numpy as np, cv2, torch, torch.nn.functional as F
sys.path.insert(0,'/home/claude/rummi/ml')
from models import Det, Cls, crop_tile, to_tensor, CW, CH
from train_det import targets, focal
torch.set_num_threads(2); random.seed(0); np.random.seed(0); torch.manual_seed(0)
L=pickle.load(open('labels_oriented.pkl','rb')); names=sorted(L)
test=[n for i,n in enumerate(names) if (i//10)%5==0]; train=[n for n in names if n not in test]
json.dump(test,open('ft/test_names.json','w'))
SL=pickle.load(open('/home/claude/rummi/ml/data/jpg/labels.pkl','rb')); syn=sorted(k for k in SL if not k.startswith('val'))
print('real train',len(train),'test',len(test),flush=True)
# ---------- clasificador ----------
AMBIG={1,6,8,9,11}
X=[];N=[];C=[];FL=[]
for fn in train:
    img=cv2.cvtColor(cv2.imread('images/'+fn),cv2.COLOR_BGR2RGB)
    for t in L[fn]:
        if t['conf']<0.6 or t['num'] in AMBIG: continue
        for r in range(3):
            a=math.atan2(t['uy'],t['ux'])+random.gauss(0,0.1); w=t['w']*math.exp(random.gauss(0,0.1))
            X.append(crop_tile(img,t['cx']+random.gauss(0,.06*t['w']),t['cy']+random.gauss(0,.06*t['w']),math.cos(a),math.sin(a),w))
            N.append(t['num']); C.append(t['color'] if t['color']<3 else 99); FL.append(-1)   # 99 = naranja-o-amarillo
nreal=len(X); print('recortes reales',nreal,flush=True)
for k in random.sample(syn,700):
    img=cv2.cvtColor(cv2.imread('/home/claude/rummi/ml/data/jpg/'+k),cv2.COLOR_BGR2RGB)
    for cx,cy,ux,uy,w,num,col in SL[k]:
        a=math.atan2(uy,ux)+random.gauss(0,0.12); fl=random.random()<0.2
        if fl: a+=math.pi
        ww=w*math.exp(random.gauss(0,0.12))
        X.append(crop_tile(img,cx+random.gauss(0,.08*w),cy+random.gauss(0,.08*w),math.cos(a),math.sin(a),ww)); N.append(int(num)); C.append(int(col)); FL.append(int(fl))
X=np.stack(X); N=np.array(N); C=np.array(C); FL=np.array(FL); print('recortes totales',len(X),flush=True)
cls=Cls(); cls.load_state_dict(torch.load('/home/claude/rummi/ml/cls.pt')); opt=torch.optim.AdamW(cls.parameters(),5e-4,weight_decay=1e-4)
def col_loss(lc,tc):
    lp=F.log_softmax(lc,1); m1=(tc>=0)&(tc<6); m2=tc==99; l=0
    if m1.any(): l=l+F.nll_loss(lp[m1],tc[m1],reduction='sum')
    if m2.any(): l=l-torch.logsumexp(lp[m2][:,[3,4]],1).sum()
    return l/max(1,int((m1|m2).sum()))
for ep in range(4):
    perm=np.random.permutation(len(X)); cls.train(); t0=time.time()
    for b in range(0,len(X)-127,128):
        idx=perm[b:b+128]; x=to_tensor(X[idx]); x=x*torch.empty(len(idx),1,1,1).uniform_(0.75,1.25)+torch.empty(len(idx),3,1,1).uniform_(-0.08,0.08)
        n,c,f=cls(x); tn=torch.from_numpy(N[idx]); tc=torch.from_numpy(C[idx]); tf=torch.from_numpy(FL[idx])
        l=F.cross_entropy(n,tn)+col_loss(c,tc)+0.5*F.cross_entropy(f,tf,ignore_index=-1)
        opt.zero_grad(); l.backward(); opt.step()
    print('cls ep',ep,f'{time.time()-t0:.0f}s',flush=True)
torch.save(cls.state_dict(),'ft/cls_ft.pt')
del X
# ---------- detector ----------
R=[]
for fn in train:
    img=cv2.cvtColor(cv2.imread('images/'+fn),cv2.COLOR_BGR2RGB); s=960/max(img.shape[:2]); img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    lab=np.array([[t['cx']*s,t['cy']*s,t['ux'],t['uy'],t['w']*s,t['num'],t['color']] for t in L[fn]],np.float32)
    R.append((img,lab))
def real_crop():
    img,lab=random.choice(R); H,W=img.shape[:2]; S=512
    x0=random.randint(0,max(0,W-S)); y0=random.randint(0,max(0,H-S))
    c=np.full((S,S,3),128,np.uint8); c[:min(S,H-y0),:min(S,W-x0)]=img[y0:y0+S,x0:x0+S]
    l=lab.copy(); l[:,0]-=x0; l[:,1]-=y0; l=l[(l[:,0]>=0)&(l[:,0]<S)&(l[:,1]>=0)&(l[:,1]<S)]
    return c,l
det=Det(); det.load_state_dict(torch.load('/home/claude/rummi/ml/det.pt')); opt=torch.optim.AdamW(det.parameters(),5e-4,weight_decay=1e-4)
STEPS=500; t0=time.time()
for st in range(STEPS):
    xs=[];hs=[];gs=[];ms=[];um=[]
    for i in range(8):
        if i<4: im,lb=real_crop(); real=1
        else:
            k=random.choice(syn); im=cv2.cvtColor(cv2.imread('/home/claude/rummi/ml/data/jpg/'+k),cv2.COLOR_BGR2RGB); lb=SL[k]; real=0
        hm,reg,msk=targets(lb,512); xs.append(im); hs.append(hm); gs.append(reg); ms.append(msk); um.append(0.0 if real else 1.0)
    X_=to_tensor(np.stack(xs)); H_=torch.from_numpy(np.stack(hs)); G_=torch.from_numpy(np.stack(gs)); M_=torch.from_numpy(np.stack(ms)); U_=torch.tensor(um)[:,None,None,None]
    out=det(X_); lh=focal(out[:,0],H_)
    up=out[:,1:3]; up=up/(up.norm(dim=1,keepdim=True)+1e-6); m=M_.unsqueeze(1)
    lu=((up-G_[:,:2]).abs()*m*U_).sum()/((m*U_).sum()*2+1)    # orientación: solo sintético
    lw=((out[:,3:4]-G_[:,2:3]).abs()*m).sum()/(m.sum()+1)
    l=lh+2*lu+lw; opt.zero_grad(); l.backward(); opt.step()
    if st%100==0: print('det step',st,round(lh.item(),3),round(lu.item(),3),round(lw.item(),3),f'{time.time()-t0:.0f}s',flush=True)
torch.save(det.state_dict(),'ft/det_ft.pt'); print('fin',flush=True)

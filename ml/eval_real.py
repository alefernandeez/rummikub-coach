import sys; sys.path.insert(0,'.')
import cv2, copy
from infer import *
N=['ne','ro','az','na','am','ve']
GT=[['7az','8az','9az'],['3az','4az','5az'],['6ve','6na','6ro','6am'],['4na','4ne','4ve'],['12ne','12ve','12az','12na','12am'],
 ['2am','2ne','2az','2ro','2ve'],['10na','10ro','10am'],['10ro','J','12ro','13ro'],['1am','1na','1az','1ne'],['10az','11az','12az'],
 ['6am','6na','6ne'],['12am','12ve','12ne'],['7ve','7am','7na'],['8ne','J','10ne','11ne'],['1ro','2ro','3ro']]
gtsig=[sorted(s) for s in GT]
lab=lambda t:'J' if t['num']==0 else f"{t['num']}{N[t['color']]}"
P=Pipeline(*(sys.argv[1:3] if len(sys.argv)>2 else []))
def evaluate(img, size):
    t=P.classify(img,P.detect(img,size)); rk=rack_filter(t); tb=[x for x in t if x not in rk]
    sets=[g for g in group_sets(tb) if not (len(g)==1 and g[0]['pn']<0.7)]
    res={}
    for name in ['sin corrección','con corrección']:
        S=copy.deepcopy(sets)
        ch=apply_corrections(S) if name=='con corrección' else 0
        sig=[sorted(lab(x) for x in g) for g in S]
        exact=sum(1 for s in gtsig if s in sig)
        # fichas correctas: por set GT, mejor solapamiento
        ok=0
        for s in gtsig:
            best=0
            for q in sig:
                a=list(q); m=0
                for x in s:
                    if x in a: a.remove(x); m+=1
                best=max(best,m)
            ok+=best
        res[name]=(exact,ok,ch)
    return res
img=cv2.cvtColor(cv2.imread('../real/foto1.png'),cv2.COLOR_BGR2RGB)
print('foto completa:',evaluate(img,960))
crop=cv2.resize(img[850:1360,230:1000],None,fx=1.6,fy=1.6,interpolation=cv2.INTER_CUBIC)
print('(sets exactos de 15, fichas correctas de 53, fichas cambiadas)')

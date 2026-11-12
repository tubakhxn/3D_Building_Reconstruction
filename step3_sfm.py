# Developer/Creator: tubakhxn
import cv2, numpy as np, pickle, time, json
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
files,K,_,M=pickle.load(open('reconstruction/matches.pkl','rb'))
n=len(files); W,H=1600,900; cx,cy=W/2,H/2
import os
f0=float(os.environ.get("F0",1100))
def Km(f): return np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
def mget(i,j):
    if (i,j) in M: return M[(i,j)]
    if (j,i) in M: return M[(j,i)][:,::-1]
    return None
R={};T={};pts=[];ptobs=[];kp2pt=[{} for _ in range(n)];f=f0
def angle(c1,c2,X):
    a=X-c1;b=X-c2
    return np.degrees(np.arccos(np.clip((a*b).sum(1)/np.linalg.norm(a,axis=1)/np.linalg.norm(b,axis=1),-1,1)))
def center(i): return (-R[i].T@T[i]).ravel()
def triangulate(i,j,idx1,idx2):
    Pi=Km(f)@np.hstack([R[i],T[i]]);Pj=Km(f)@np.hstack([R[j],T[j]])
    x1=K[i][idx1];x2=K[j][idx2]
    X=cv2.triangulatePoints(Pi,Pj,x1.T.astype(float),x2.T.astype(float));X=(X[:3]/X[3]).T
    ok=np.ones(len(X),bool)
    for c,x in ((i,x1),(j,x2)):
        Xc=(R[c]@X.T+T[c]).T; ok&=Xc[:,2]>0
        pr=(Km(f)@Xc.T).T; pr=pr[:,:2]/pr[:,2:3]; ok&=np.linalg.norm(pr-x,axis=1)<2.0
    ok&=angle(center(i),center(j),X)>1.0
    return X,ok
def add_points(i,j):
    m=mget(i,j)
    if m is None: return 0
    sel=[(a,b) for a,b in m if a not in kp2pt[i] and b not in kp2pt[j]]
    if len(sel)<8: return 0
    sel=np.array(sel);X,ok=triangulate(i,j,sel[:,0],sel[:,1])
    for (a,b),x,o in zip(sel,X,ok):
        if not o: continue
        pid=len(pts);pts.append(x);ptobs.append({i:a,j:b});kp2pt[i][a]=pid;kp2pt[j][b]=pid
    return int(ok.sum())
def extend_obs(i,j):
    m=mget(i,j)
    if m is None: return
    for a,b in m:
        pa=kp2pt[i].get(a);pb=kp2pt[j].get(b)
        if pa is not None and pb is None: kp2pt[j][b]=pa;ptobs[pa][j]=b
        elif pb is not None and pa is None: kp2pt[i][a]=pb;ptobs[pb][i]=a
def ba(maxnfev=15):
    global f,pts
    cams=sorted(R);ci={c:k for k,c in enumerate(cams)}
    live=[p for p in range(len(pts)) if len(ptobs[p])>=2]
    pi={p:k for k,p in enumerate(live)}
    ob=[(ci[c],pi[p],K[c][k]) for p in live for c,k in ptobs[p].items() if c in ci]
    ca=np.array([c for c,_,_ in ob]);pa=np.array([p for _,p,_ in ob]);xy=np.array([x for *_,x in ob])
    x0=np.concatenate([np.hstack([cv2.Rodrigues(R[c])[0].ravel(),T[c].ravel()]) for c in cams]+[[f]]+[pts[p] for p in live])
    nc=len(cams);nf=6*nc
    def fun(x):
        cp=x[:nf].reshape(nc,6);ff=x[nf];P=x[nf+1:].reshape(-1,3)
        out=np.empty((len(ca),2))
        rv=cp[ca,:3];th=np.linalg.norm(rv,axis=1,keepdims=True)+1e-12;k=rv/th
        Pp=P[pa];c=np.cos(th);s=np.sin(th)
        Xr=Pp*c+np.cross(k,Pp)*s+k*(k*Pp).sum(1,keepdims=True)*(1-c)+cp[ca,3:]
        u=ff*Xr[:,0]/Xr[:,2]+cx;v=ff*Xr[:,1]/Xr[:,2]+cy
        return np.concatenate([(u-xy[:,0]),(v-xy[:,1])])
    m=len(ca);A=lil_matrix((2*m,len(x0)),dtype=int);r=np.arange(m)
    for d in range(6):
        A[r,6*ca+d]=1;A[r+m,6*ca+d]=1
    A[r,nf]=1;A[r+m,nf]=1
    for d in range(3):
        A[r,nf+1+3*pa+d]=1;A[r+m,nf+1+3*pa+d]=1
    lo=-np.inf*np.ones(len(x0));hi=np.inf*np.ones(len(x0))
    c0=x0[:6];lo[:6]=c0-1e-9;hi[:6]=c0+1e-9;lo[nf]=300;hi[nf]=4000
    res=least_squares(fun,x0,jac_sparsity=A,bounds=(lo,hi),loss='huber',f_scale=1.5,x_scale='jac',max_nfev=maxnfev,method='trf')
    x=res.x;cp=x[:nf].reshape(nc,6);f=float(x[nf])
    for c in cams: R[c]=cv2.Rodrigues(cp[ci[c],:3])[0];T[c]=cp[ci[c],3:].reshape(3,1)
    for p in live: pts[p]=x[nf+1+3*pi[p]:nf+4+3*pi[p]]
    e=np.abs(res.fun).reshape(2,-1);err=np.sqrt((e**2).sum(0))
    bad=set()
    for (c,p,_),er in zip(ob,err):
        if er>4.0:
            cam=cams[c];pid=live[p]
            if cam in ptobs[pid]:
                k=ptobs[pid].pop(cam);kp2pt[cam].pop(k,None)
    return float(np.sqrt(np.mean(err**2))),f
t0=time.time()
best=None
for (i,j),m in M.items():
    if j-i<3 or len(m)<400: continue
    p1=K[i][m[:,0]];p2=K[j][m[:,1]]
    E,mk=cv2.findEssentialMat(p1,p2,Km(f),cv2.RANSAC,0.999,1.0)
    if E is None: continue
    _,Rr,tt,mk2=cv2.recoverPose(E,p1,p2,Km(f),mask=mk)
    sc=int(mk2.sum())*min(1,np.median(np.linalg.norm(p1-p2,axis=1))/25)
    if best is None or sc>best[0]: best=(sc,i,j,Rr,tt)
_,i0,j0,Rr,tt=best
print('init pair',i0,j0,best[0],flush=True)
R[i0]=np.eye(3);T[i0]=np.zeros((3,1));R[j0]=Rr;T[j0]=tt
print('init pts',add_points(i0,j0));print('BA',ba(),flush=True)
order=[k for k in range(n) if k not in R]
order.sort(key=lambda k:min(abs(k-i0),abs(k-j0)))
failed=[]
for k in order:
    obj=[];img=[]
    for j in sorted(R,key=lambda c:abs(c-k))[:6]:
        m=mget(k,j)
        if m is None: continue
        for a,b in m:
            p=kp2pt[j].get(b)
            if p is not None and a not in kp2pt[k]: obj.append(pts[p]);img.append(K[k][a])
    if len(obj)<30: failed.append(k);continue
    obj=np.array(obj);img=np.array(img)
    ok,rv,tv,inl=cv2.solvePnPRansac(obj,img,Km(f),None,iterationsCount=3000,reprojectionError=3.0,confidence=0.999)
    if not ok or inl is None or len(inl)<25: failed.append(k);print('fail',k,0 if inl is None else len(inl));continue
    R[k]=cv2.Rodrigues(rv)[0];T[k]=tv.reshape(3,1)
    for j in list(R):
        if j!=k and mget(k,j) is not None: extend_obs(k,j)
    na=0
    for j in sorted(R,key=lambda c:abs(c-k)):
        if j!=k: na+=add_points(j,k)
    print('reg',k,'pnp inl',len(inl),'new',na,'pts',len(pts),flush=True)
    if len(R)%5==0: print(' BA',ba(8),time.time()-t0,flush=True)
rms,f=ba(40);print('final BA rms px',rms,'f',f)
rms,f=ba(25);print('final BA rms px',rms,'f',f)
live=[p for p in range(len(pts)) if len(ptobs[p])>=2]
imgs={c:cv2.imread(files[c]) for c in R}
P=[];C=[];nobs=[]
for p in live:
    P.append(pts[p]);cols=[]
    for c,k in ptobs[p].items():
        x,y=K[c][k].astype(int);cols.append(imgs[c][min(y,H-1),min(x,W-1)][::-1])
    C.append(np.median(cols,axis=0));nobs.append(len(ptobs[p]))
P=np.array(P);C=np.array(C)/255.;nobs=np.array(nobs)
pickle.dump(dict(R=R,T=T,f=f,P=P,C=C,nobs=nobs,files=files,failed=failed,ptobs=[ptobs[p] for p in live]),open('reconstruction/sfm_'+os.environ.get('F0','1100')+'.pkl','wb'))
print('registered',len(R),'/',n,'failed',failed,'points',len(P),'mean track',nobs.mean())

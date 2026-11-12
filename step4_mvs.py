# Developer/Creator: tubakhxn
import cv2, numpy as np, pickle, time
S=pickle.load(open('reconstruction/sfm_1050.pkl','rb'))
R,T,f,P=S['R'],S['T'],S['f'],S['P']; cams=sorted(R)
sc=0.5; W,H=int(1600*sc),int(900*sc)
Ks=np.array([[f*sc,0,W/2],[0,f*sc,H/2],[0,0,1.]])
imgs={c:cv2.resize(cv2.imread(S['files'][c]),(W,H),interpolation=cv2.INTER_AREA) for c in cams}
gray={c:cv2.cvtColor(imgs[c],cv2.COLOR_BGR2GRAY).astype(np.float32)/255 for c in cams}
cen={c:(-R[c].T@T[c]).ravel() for c in cams}
def zncc(a,b,w=7):
    k=(w,w);ma=cv2.blur(a,k);mb=cv2.blur(b,k)
    va=cv2.blur(a*a,k)-ma*ma;vb=cv2.blur(b*b,k)-mb*mb;cv=cv2.blur(a*b,k)-ma*mb
    return cv/np.sqrt(va*vb+1e-4)
def depthmap(r,srcs,nd=96):
    Xc=(R[r]@P.T+T[r]).T;z=Xc[:,2];z=z[z>0]
    zmin,zmax=np.percentile(z,2)*0.8,np.percentile(z,98)*1.25
    inv=np.linspace(1/zmax,1/zmin,nd);Kinv=np.linalg.inv(Ks)
    best=np.full((H,W),-2,np.float32);bd=np.zeros((H,W),np.float32);sec=best.copy()
    cost=np.zeros((nd,H,W),np.float32)
    for di,iv in enumerate(inv):
        d=1/iv;cs=[]
        for s in srcs:
            Rrel=R[s]@R[r].T;trel=T[s]-Rrel@T[r]
            Hm=Ks@(Rrel+trel@np.array([[0,0,1/d]]))@Kinv
            w=cv2.warpPerspective(gray[s],Hm,(W,H),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR,borderValue=-5)
            c=zncc(gray[r],w);c[w<-1]=-1;cs.append(c)
        cs=np.sort(np.stack(cs),0)[::-1]
        cost[di]=cs[:2].mean(0)
    idx=cost.argmax(0);bc=cost.max(0)
    ii=np.clip(idx,1,nd-2);yy,xx=np.mgrid[:H,:W]
    c0,c1,c2=cost[ii-1,yy,xx],cost[ii,yy,xx],cost[ii+1,yy,xx]
    off=np.clip(0.5*(c0-c2)/(c0-2*c1+c2-1e-6),-0.5,0.5)
    ivd=np.interp(ii+off,np.arange(nd),inv)
    depth=1/ivd
    return depth.astype(np.float32),bc,cost
def backproj(r,depth):
    yy,xx=np.mgrid[:H,:W];pix=np.stack([xx,yy,np.ones_like(xx)],-1).reshape(-1,3).T
    Xc=(np.linalg.inv(Ks)@pix)*depth.ravel();return (R[r].T@(Xc-T[r])).T
if __name__=='__main__':
    t0=time.time();step=3;refs=cams[::step];DM={}
    for r in refs:
        near=sorted([c for c in cams if c!=r and 0.02<np.linalg.norm(cen[c]-cen[r])],key=lambda c:abs(c-r))
        srcs=[c for c in near if 2<=abs(c-r)<=8][:4] or near[:4]
        d,bc,_=depthmap(r,srcs);DM[r]=(d,bc,srcs);print('depth',r,srcs,round(float(np.median(bc)),2),round(time.time()-t0),flush=True)
    pickle.dump(DM,open('reconstruction/depthmaps.pkl','wb'))
    pts=[];cols=[];norms=[]
    for r in refs:
        d,bc,srcs=DM[r];mask=(bc>0.75)
        X=backproj(r,d);votes=np.zeros(H*W,int)
        for o in refs:
            if o==r: continue
            Xo=(R[o]@X.T+T[o]).T;z=Xo[:,2];pr=(Ks@Xo.T).T;u=pr[:,0]/pr[:,2];v=pr[:,1]/pr[:,2]
            ok=(z>0)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1)
            do=cv2.remap(DM[o][0],u.astype(np.float32).reshape(H,W),v.astype(np.float32).reshape(H,W),cv2.INTER_NEAREST).ravel()
            votes+=(ok&(np.abs(do-z)/z<0.02)).astype(int)
        keep=mask.ravel()&(votes>=2)
        pts.append(X[keep]);cols.append(imgs[r].reshape(-1,3)[keep][:,::-1]/255.)
        print('fuse',r,int(keep.sum()),flush=True)
    pts=np.concatenate(pts);cols=np.concatenate(cols)
    pickle.dump((pts,cols),open('reconstruction/dense_raw.pkl','wb'));print('raw dense',len(pts))

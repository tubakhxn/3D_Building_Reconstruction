# Developer/Creator: tubakhxn
import pickle,numpy as np,cv2,time
from scipy.spatial import cKDTree
import step4_mvs as m
R,T,Ks,W,H,imgs=m.R,m.T,m.Ks,m.W,m.H,m.imgs
DM=pickle.load(open('reconstruction/depthmaps.pkl','rb'));refs=sorted(DM)
bcs=np.concatenate([DM[r][1].ravel() for r in refs]);print('bc pct',np.percentile(bcs,[10,50,75,90]))
pts=[];cols=[];MK={}
for r in refs:
    d,bc,_=DM[r];d=cv2.medianBlur(d.astype(np.float32),5)
    gx=cv2.Sobel(np.log(d),cv2.CV_32F,1,0);gy=cv2.Sobel(np.log(d),cv2.CV_32F,0,1)
    smooth=(np.abs(gx)+np.abs(gy))<0.35
    mask=(bc>0.55)&smooth
    mask[:int(0.22*H)]=False
    X=m.backproj(r,d);votes=np.zeros(H*W,int)
    for o in refs:
        if o==r: continue
        Xo=(R[o]@X.T+T[o]).T;z=Xo[:,2];pr=(Ks@Xo.T).T;u=pr[:,0]/pr[:,2];v=pr[:,1]/pr[:,2]
        ok=(z>0)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1)
        do=cv2.remap(DM[o][0],u.astype(np.float32).reshape(H,W),v.astype(np.float32).reshape(H,W),cv2.INTER_NEAREST).ravel()
        votes+=(ok&(np.abs(do-z)/z<0.012)).astype(int)
    keep=mask.ravel()&(votes>=4)
    MK[r]=(d,keep.reshape(H,W))
    pts.append(X[keep]);cols.append(imgs[r].reshape(-1,3)[keep][:,::-1]/255.)
pts=np.concatenate(pts);cols=np.concatenate(cols);print('after consistency',len(pts))
vs=0.05;key=np.floor(pts/vs).astype(np.int64);_,ui=np.unique(key,axis=0,return_index=True);pts,cols=pts[ui],cols[ui];print('voxel',len(pts))
tr=cKDTree(pts);dd,_=tr.query(pts,k=12);md=dd[:,1:].mean(1);ok=md<md.mean()+1.5*md.std()
pts,cols=pts[ok],cols[ok];print('after SOR',len(pts))
pickle.dump((pts,cols),open('reconstruction/dense_clean.pkl','wb'))
pickle.dump(MK,open('reconstruction/masked_depth.pkl','wb'))

# Developer/Creator: tubakhxn
import pickle,numpy as np,cv2
from skimage import measure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import step4_mvs as m
R,T,Ks,W,H,imgs=m.R,m.T,m.Ks,m.W,m.H,m.imgs
DM=pickle.load(open('reconstruction/depthmaps.pkl','rb'));refs=sorted(DM)
cen=pickle.load(open('reconstruction/slab_centre.pkl','rb'));d,c=pickle.load(open('reconstruction/dense_clean.pkl','rb'))
rad=np.array([9.,4.5,7.5]);lo=cen-rad;hi=cen+rad;vs=0.13;tr=0.7
MK={}
for r in refs:
    dep,bc,_=DM[r];dep=cv2.medianBlur(dep.astype(np.float32),5)
    gx=cv2.Sobel(np.log(dep),cv2.CV_32F,1,0);gy=cv2.Sobel(np.log(dep),cv2.CV_32F,0,1)
    mk=(bc>0.4)&((np.abs(gx)+np.abs(gy))<0.5);mk[:int(0.22*H)]=False
    X=m.backproj(r,dep);votes=np.zeros(H*W,int)
    for o in refs:
        if o==r:continue
        Xo=(R[o]@X.T+T[o]).T;z=Xo[:,2];pr=(Ks@Xo.T).T;u=pr[:,0]/pr[:,2];v=pr[:,1]/pr[:,2]
        ok=(z>0)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1)
        do=cv2.remap(DM[o][0],u.astype(np.float32).reshape(H,W),v.astype(np.float32).reshape(H,W),cv2.INTER_NEAREST).ravel()
        votes+=(ok&(np.abs(do-z)/z<0.02)).astype(int)
    MK[r]=(dep,mk&(votes.reshape(H,W)>=3))
dims=np.ceil((hi-lo)/vs).astype(int);print('grid',dims,dims.prod())
g=np.stack(np.meshgrid(*[np.arange(n) for n in dims],indexing='ij'),-1).reshape(-1,3)*vs+lo+vs/2
tsdf=np.zeros(len(g),np.float32);wt=np.zeros(len(g),np.float32);col=np.zeros((len(g),3),np.float32)
for r,(dep,mk) in MK.items():
    Xc=(R[r]@g.T+T[r]).T;z=Xc[:,2];pr=(Ks@Xc.T).T;u=pr[:,0]/pr[:,2];v=pr[:,1]/pr[:,2]
    ok=(z>0.1)&(u>=0)&(u<W-1)&(v>=0)&(v<H-1);ui=np.clip(u.astype(int),0,W-1);vi=np.clip(v.astype(int),0,H-1)
    sdf=dep[vi,ui]-z;valid=ok&mk[vi,ui]&(sdf>-tr);t=np.clip(sdf/tr,-1,1)
    tsdf[valid]=(tsdf[valid]*wt[valid]+t[valid])/(wt[valid]+1)
    near=valid&(np.abs(sdf)<tr*0.5)
    col[near]=(col[near]*wt[near,None]+imgs[r][vi[near],ui[near]][:,::-1]/255.)/(wt[near,None]+1);wt[valid]+=1
vol=tsdf.reshape(dims);W3=wt.reshape(dims);C3=col.reshape(*dims,3);vol[W3<2]=1.0
vol=cv2.GaussianBlur(vol.reshape(dims[0],-1).astype(np.float32),(0,0),0.8).reshape(dims)
V,F,_,_=measure.marching_cubes(vol,0.0,spacing=(vs,)*3,allow_degenerate=False);V=V+lo
idx=np.clip(np.round((V-lo)/vs-0.5).astype(int),0,dims-1)
Vc=C3[idx[:,0],idx[:,1],idx[:,2]];vw=W3[idx[:,0],idx[:,1],idx[:,2]]
A=coo_matrix((np.ones(len(F)*3),(np.r_[F[:,0],F[:,1],F[:,2]],np.r_[F[:,1],F[:,2],F[:,0]])),shape=(len(V),)*2)
nc,lab=connected_components(A,directed=False);cnt=np.bincount(lab);kv=cnt[lab]>=120
Fk=F[kv[F].all(1)&(vw[F].min(1)>=2)]
used=np.unique(Fk);rm=-np.ones(len(V),int);rm[used]=np.arange(len(used));V2,C2,F2=V[used],Vc[used],rm[Fk]
tt=cKDTree(d);_,ni=tt.query(V2);bad=C2.sum(1)<0.02;C2[bad]=c[ni[bad]]
print('mesh verts',len(V2),'faces',len(F2))
pickle.dump((V2,F2,C2),open('reconstruction/mesh.pkl','wb'))
def ply(path,P,C,F=None):
    with open(path,'wb') as fh:
        h=f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
        if F is not None:h+=f"element face {len(F)}\nproperty list uchar int vertex_indices\n"
        fh.write((h+"end_header\n").encode())
        a=np.empty(len(P),np.dtype([('p','<f4',3),('c','u1',3)]));a['p']=P;a['c']=(np.clip(C,0,1)*255).astype(np.uint8);fh.write(a.tobytes())
        if F is not None:
            b=np.empty(len(F),np.dtype([('n','u1'),('i','<i4',3)]));b['n']=3;b['i']=F;fh.write(b.tobytes())
ply('reconstruction/mesh/building_mesh.ply',V2,C2,F2)
ply('reconstruction/point_cloud/dense_point_cloud.ply',d,c)
S=pickle.load(open('reconstruction/sfm_1050.pkl','rb'));ply('reconstruction/point_cloud/sparse_point_cloud.ply',S['P'],S['C'])

# Developer/Creator: tubakhxn
import numpy as np, cv2
def look_at(eye,tgt,up):
    z=tgt-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    R=np.stack([x,y,z]);return R,-R@eye
def splat(img,P,C,R,t,f,size,zb=None,rad=1,dark=1.0):
    W,H=size;Xc=(R@P.T).T+t;z=Xc[:,2];ok=z>1e-3
    Xc=Xc[ok];z=z[ok];c=C[ok]
    u=(f*Xc[:,0]/z+W/2).astype(int);v=(f*Xc[:,1]/z+H/2).astype(int)
    m=(u>=rad)&(u<W-rad)&(v>=rad)&(v<H-rad);u,v,z,c=u[m],v[m],z[m],c[m]
    o=np.argsort(-z);u,v,z,c=u[o],v[o],z[o],c[o]
    col=(np.clip(c[:,::-1]*dark,0,1)*255).astype(np.uint8)
    for dy in range(-rad+1,rad):
        for dx in range(-rad+1,rad):
            img[v+dy,u+dx]=col
    return img
def draw_axes(img,R,t,f,origin,L,size):
    W,H=size
    def pr(p):
        X=R@p+t;return (int(f*X[0]/X[2]+W/2),int(f*X[1]/X[2]+H/2))
    for d,c in ((np.array([L,0,0]),(80,80,255)),(np.array([0,L,0]),(80,255,80)),(np.array([0,0,L]),(255,160,60))):
        cv2.line(img,pr(origin),pr(origin+d),c,2,cv2.LINE_AA)
def mesh_render(img,V,F,C,R,t,f,size,light=np.array([0.3,-0.8,-0.5])):
    W,H=size;Xc=(R@V.T).T+t;z=Xc[:,2]
    tri=Xc[F];zc=tri[...,2]
    ok=(zc>0.2).all(1);F=F[ok];tri=tri[ok]
    nrm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);nl=np.linalg.norm(nrm,axis=1)+1e-9;nrm/=nl[:,None]
    ctr=tri.mean(1);facing=(nrm*ctr).sum(1);nrm[facing>0]*=-1
    lg=light/np.linalg.norm(light);lg=R@lg
    sh=0.45+0.55*np.clip(-(nrm*lg).sum(1)*-1,0,1)
    sh=0.4+0.6*np.abs((nrm*(-ctr/np.linalg.norm(ctr,axis=1)[:,None])).sum(1))
    uv=np.stack([f*tri[...,0]/tri[...,2]+W/2,f*tri[...,1]/tri[...,2]+H/2],-1)
    col=C[F].mean(1)*sh[:,None];col=(np.clip(col[:,::-1],0,1)*255).astype(np.uint8)
    o=np.argsort(-ctr[:,2]);uv=uv.astype(np.int32)
    for k in o:
        cv2.fillConvexPoly(img,uv[k],tuple(int(x) for x in col[k]),cv2.LINE_8)
    return img

# Developer/Creator: tubakhxn
import pickle,json,numpy as np,cv2,sys,time,subprocess
sys.path.insert(0,'.')
from render_lib import look_at,splat,mesh_render
W,H,FPS=1280,720,30
S=pickle.load(open('reconstruction/sfm_1050.pkl','rb'));Rm,Tm,F0=S['R'],S['T'],S['f']
cams=sorted(Rm);Pc=np.array([(-Rm[c].T@Tm[c]).ravel() for c in cams])
P,PC=S['P'],S['C'];dn,dc=pickle.load(open('reconstruction/dense_clean.pkl','rb'))
V,Fc,VC=pickle.load(open('reconstruction/mesh.pkl','rb'))
n,_=pickle.load(open('reconstruction/ground.pkl','rb'));cen=pickle.load(open('reconstruction/slab_centre.pkl','rb'))
files,Kp,_,M=pickle.load(open('reconstruction/matches.pkl','rb'))
meta=json.load(open('reconstruction/frames/meta.json'))
a=np.cross(n,[1,0,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
v0=Pc.mean(0)-cen;az0=np.arctan2(v0@b,v0@a)
NCAM,NSP,NDN,NV,NF=len(cams),len(P),len(dn),len(V),len(Fc)
WHITE=(255,255,255)
def eye_at(az,el,r,c=cen):return c+r*(np.cos(el)*(np.cos(az)*a+np.sin(az)*b)+np.sin(el)*n)
def txt(img,s,pos,sc=0.6,th=1,col=WHITE,al=1.0,font=cv2.FONT_HERSHEY_SIMPLEX):
    if al>=1:cv2.putText(img,s,pos,font,sc,col,th,cv2.LINE_AA);return
    o=img.copy();cv2.putText(o,s,pos,font,sc,col,th,cv2.LINE_AA);cv2.addWeighted(o,al,img,1-al,0,img)
def bg():
    y=np.linspace(0,1,H)[:,None,None];return np.broadcast_to((20+10*y)*np.ones((1,W,3)),(H,W,3)).astype(np.uint8).copy()
def hud(img,k,label,info):
    ov=img.copy();cv2.rectangle(ov,(0,0),(W,64),(10,10,12),-1);cv2.addWeighted(ov,0.55,img,0.45,0,img)
    txt(img,f"SCENE {k} / 7",(30,26),0.45,1,(150,150,150));txt(img,label,(30,54),0.9,2,WHITE,1,cv2.FONT_HERSHEY_DUPLEX)
    txt(img,info,(30,H-24),0.5,1,(190,190,190),0.9)
def wm(img):txt(img,"tubakhxn",(W-135,H-24),0.6,1,WHITE,0.6)
def fade(img,i,nf,k=10):
    al=min(1,(i+1)/k,(nf-i)/k)
    return img if al>=1 else (img*al).astype(np.uint8)
def seg(img,p,q,Rv,tv,f,col,th=1):
    A=Rv@p+tv;B=Rv@q+tv
    if A[2]<0.2 and B[2]<0.2:return
    if A[2]<0.2:A=A+(B-A)*((0.2-A[2])/(B[2]-A[2]))
    if B[2]<0.2:B=B+(A-B)*((0.2-B[2])/(A[2]-B[2]))
    f1=lambda X:(int(f*X[0]/X[2]+W/2),int(f*X[1]/X[2]+H/2))
    cv2.line(img,f1(A),f1(B),col,th,cv2.LINE_AA)
def grid_axes(img,Rv,tv,f,c0,ext=12,step=2):
    gc=c0-((c0-cen)@n)*n-n*0.0
    base=cen-(( cen@n)*n)
    h=np.percentile(dn@n,3)
    org=cen+(h-cen@n)*n
    ov=img.copy()
    for t in np.arange(-ext,ext+.1,step):
        seg(ov,org+t*a-ext*b,org+t*a+ext*b,Rv,tv,f,(70,70,75));seg(ov,org+t*b-ext*a,org+t*b+ext*a,Rv,tv,f,(70,70,75))
    cv2.addWeighted(ov,0.6,img,0.4,0,img)
    o=org-ext*0.85*a-ext*0.85*b;L=3.0
    for d,col,nm in ((a,(90,90,255),'X'),(b,(90,230,90),'Y'),(n,(255,170,60),'Z')):
        seg(img,o,o+L*d,Rv,tv,f,col,3)
        X=Rv@(o+1.15*L*d)+tv
        if X[2]>0.2:txt(img,nm,(int(f*X[0]/X[2]+W/2)-6,int(f*X[1]/X[2]+H/2)+6),0.6,2,col)
def view(az,el,r,c=cen):
    e=eye_at(az,el,r,c);return look_at(e,c,n)
def frustum(img,c,Rv,tv,f,col,s=0.5,th=1):
    w,h=1600,900;cs=[np.array([x*w/2,y*h/2,F0])*(s/F0) for x,y in((-1,-1),(1,-1),(1,1),(-1,1))]
    C=-Rm[c].T@Tm[c];C=C.ravel();wc=[Rm[c].T@(p-Tm[c].ravel()) for p in cs]
    for p in wc:seg(img,C,p,Rv,tv,f,col,th)
    for i in range(4):seg(img,wc[i],wc[(i+1)%4],Rv,tv,f,col,th)
def sc_title(nf,title,sub,bgcloud=True):
    for i in range(nf):
        img=bg()
        if bgcloud:
            Rv,tv=view(az0+np.radians(40)*np.sin(i/nf*np.pi),np.radians(30),30,cen)
            splat(img,P,PC*0.45,Rv,tv,800,(W,H),rad=2)
        ov=img.copy();cv2.rectangle(ov,(0,H//2-120),(W,H//2+110),(8,8,10),-1);cv2.addWeighted(ov,0.72,img,0.28,0,img)
        (tw,_),_=cv2.getTextSize(title,cv2.FONT_HERSHEY_DUPLEX,1.35,2);txt(img,title,((W-tw)//2,H//2-20),1.35,2,WHITE,1,cv2.FONT_HERSHEY_DUPLEX)
        (sw,_),_=cv2.getTextSize(sub,cv2.FONT_HERSHEY_SIMPLEX,0.62,1);txt(img,sub,((W-sw)//2,H//2+30),0.62,1,(190,190,190))
        cv2.line(img,(W//2-90,H//2+60),(W//2+90,H//2+60),(120,120,120),1,cv2.LINE_AA)
        wm(img);yield fade(img,i,nf,15)
def sc_end(nf):
    L=["Developed by tubakhxn.","Research & Educational Purposes Only."]
    for i in range(nf):
        img=bg();Rv,tv=view(az0+np.radians(40)*np.cos(i/nf*np.pi),np.radians(30),30);splat(img,dn,dc*0.4,Rv,tv,800,(W,H),rad=2)
        ov=img.copy();cv2.rectangle(ov,(0,H//2-80),(W,H//2+80),(8,8,10),-1);cv2.addWeighted(ov,0.75,img,0.25,0,img)
        for k,s in enumerate(L):
            (tw,_),_=cv2.getTextSize(s,cv2.FONT_HERSHEY_DUPLEX,0.95 if k==0 else 0.6,1);txt(img,s,((W-tw)//2,H//2-10+k*46),0.95 if k==0 else 0.6,1,WHITE if k==0 else (180,180,180),1,cv2.FONT_HERSHEY_DUPLEX)
        wm(img);yield fade(img,i,nf,15)
def load_video():
    cap=cv2.VideoCapture('input/original_drone.mp4');fr={};i=0
    while True:
        ok,f=cap.read()
        if not ok:break
        if i%2==0:fr[i]=cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA)
        i+=1
    return fr
def sc1(fr):
    ks=sorted(fr);nf=len(ks)
    for i,k in enumerate(ks):
        img=fr[k].copy();hud(img,1,"ORIGINAL DRONE FOOTAGE",f"3840x2160 · 29.97 fps · {i*2+1}/405 frames · shown at 2x speed");wm(img);yield fade(img,i,nf,10)
def sc2(nf_each=42):
    pairs=[(8,16),(24,32),(36,44)];tw,th=620,349;sx=tw/1600
    for pi_,(i,j) in enumerate(pairs):
        m=M[(i,j)];rng=np.random.default_rng(pi_);sel=rng.choice(len(m),min(140,len(m)),replace=False)
        I1=cv2.resize(cv2.imread(files[i]),(tw,th));I2=cv2.resize(cv2.imread(files[j]),(tw,th))
        for t in range(nf_each):
            img=bg();x1,x2,y=15,W-tw-15,190
            img[y:y+th,x1:x1+tw]=I1;img[y:y+th,x2:x2+tw]=I2
            cnt=int(len(sel)*min(1,(t+1)/(nf_each*0.7)));ov=img.copy()
            for q,s in enumerate(sel[:cnt]):
                p=Kp[i][m[s,0]]*sx;r_=Kp[j][m[s,1]]*sx;col=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(q*180/len(sel)),220,255]]]),cv2.COLOR_HSV2BGR)[0,0])
                a1=(int(p[0]+x1),int(p[1]+y));a2=(int(r_[0]+x2),int(r_[1]+y))
                cv2.circle(img,a1,3,col,1,cv2.LINE_AA);cv2.circle(img,a2,3,col,1,cv2.LINE_AA);cv2.line(ov,a1,a2,col,1,cv2.LINE_AA)
            cv2.addWeighted(ov,0.55,img,0.45,0,img)
            txt(img,f"frame {meta[i][1]}",(x1,y-10),0.5,1,(190,190,190));txt(img,f"frame {meta[j][1]}",(x2,y-10),0.5,1,(190,190,190))
            txt(img,f"{len(m):,} geometrically verified matches (showing {cnt})",(15,y+th+40),0.6,1,WHITE)
            hud(img,2,"FEATURE DETECTION & MATCHING","SIFT keypoints · Lowe ratio test 0.8 · MAGSAC fundamental-matrix verification");wm(img);yield fade(img,t,nf_each,8)
def sc3(nf):
    c3=(Pc.mean(0)+cen)/2
    for i in range(nf):
        img=bg();Rv,tv=view(az0+np.radians(55)*np.sin(i/nf*2*np.pi)-np.radians(20),np.radians(30),30,c3)
        f=800;splat(img,P,np.clip(PC*1.0,0,1),Rv,tv,f,(W,H),rad=2);grid_axes(img,Rv,tv,f,cen,ext=14,step=2)
        k=max(2,int(NCAM*min(1,(i+1)/(nf*0.75))))
        for q in range(k-1):seg(img,Pc[q],Pc[q+1],Rv,tv,f,(60,200,255),3)
        for q in range(k):frustum(img,cams[q],Rv,tv,f,(235,235,235) if q<k-1 else (70,70,255),0.9 if q==k-1 else 0.7,2 if q==k-1 else 1)
        hud(img,3,"CAMERA POSE ESTIMATION",f"{NCAM}/{NCAM} cameras registered · PnP + bundle adjustment · RMS 0.86 px · focal assumed (~1050 px)");wm(img);yield fade(img,i,nf)
def orbit_scene(k,label,info,nf,draw,r,el=28,amp=60):
    for i in range(nf):
        img=bg();u=i/max(nf-1,1);az=az0+np.radians(amp)*np.sin(u*2*np.pi);Rv,tv=view(az,np.radians(el),r,cen);f=900
        draw(img,Rv,tv,f);grid_axes(img,Rv,tv,f,cen);hud(img,k,label,info);wm(img);yield fade(img,i,nf)
def sc4(nf):return orbit_scene(4,"STRUCTURE-FROM-MOTION",f"Sparse point cloud · {NSP:,} triangulated 3D points · mean track {S['nobs'].mean():.1f} views",nf,lambda im,R_,t_,f:splat(im,P,np.clip(PC*1.2,0,1),R_,t_,f,(W,H),rad=2),22)
def sc5(nf):return orbit_scene(5,"MULTI-VIEW STEREO",f"Dense point cloud · {NDN:,} points · plane-sweep ZNCC depth maps · geometric-consistency filtered · RGB from source frames",nf,lambda im,R_,t_,f:splat(im,dn,dc,R_,t_,f,(W,H),rad=2),15)
def sc6(nf):return orbit_scene(6,"3D RECONSTRUCTED BUILDING",f"TSDF fusion + marching cubes · {NV:,} vertices · {NF:,} triangles · vertex colours · unobserved areas left empty",nf,lambda im,R_,t_,f:(splat(im,dn,dc*0.35,R_,t_,f,(W,H),rad=1),mesh_render(im,V,Fc,VC,R_,t_,f,(W,H))),13)
def sc7(fr,nf):
    tw,th=620,349;ks=sorted(fr)
    for i in range(nf):
        u=i/(nf-1);j=int(round(u*(NCAM-1)));c=cams[j];fi=meta[c][1];fk=min(ks,key=lambda x:abs(x-fi))
        img=bg();left=cv2.resize(fr[fk],(tw,th),interpolation=cv2.INTER_AREA)
        r_=np.full((H,W,3),22,np.uint8);Rv=Rm[c];tv=Tm[c].ravel()
        splat(r_,dn,dc*0.6,Rv,tv,F0*W/1600,(W,H),rad=1);mesh_render(r_,V,Fc,VC,Rv,tv,F0*W/1600,(W,H))
        right=cv2.resize(r_,(tw,th),interpolation=cv2.INTER_AREA)
        y=190;img[y:y+th,15:15+tw]=left;img[y:y+th,W-tw-15:W-15]=right
        txt(img,"REAL DRONE FOOTAGE",(15,y-12),0.55,1,WHITE);txt(img,"RECONSTRUCTION (rendered from estimated camera pose)",(W-tw-15,y-12),0.55,1,WHITE)
        cv2.arrowedLine(img,(W//2-18,y+th//2),(W//2+18,y+th//2),(200,200,200),2,cv2.LINE_AA,tipLength=0.5)
        chips=["VIDEO","FEATURES","SfM","MVS","MESH"];x=15
        for q,ch in enumerate(chips):
            on=u>=q/len(chips)*0.999;cv2.rectangle(img,(x,y+th+30),(x+110,y+th+58),(230,230,230) if on else (60,60,64),-1 if on else 1)
            txt(img,ch,(x+10,y+th+50),0.5,1,(20,20,20) if on else (140,140,140));x+=122
        cv2.rectangle(img,(15,y+th+80),(W-15,y+th+84),(60,60,64),-1);cv2.rectangle(img,(15,y+th+80),(15+int((W-30)*u),y+th+84),(60,200,255),-1)
        hud(img,7,"FINAL COMPARISON","Real-world footage → 3D reconstruction · not metric-scaled (focal length assumed, no scale reference)");wm(img);yield fade(img,i,nf)
class Out:
    def __init__(s,p):s.p=p;s.w=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*'mp4v'),FPS,(W,H));s.n=0
    def add(s,g):
        for f in g:s.w.write(f);s.n+=1
    def close(s):
        s.w.release();t=s.p.replace('.mp4','_raw.mp4')
        import os;os.rename(s.p,t)
        r=subprocess.run(['ffmpeg','-y','-v','error','-i',t,'-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-preset','medium','-movflags','+faststart',s.p])
        if r.returncode!=0:os.rename(t,s.p)
        else:os.remove(t)
if __name__=='__main__':
    which=sys.argv[1];t0=time.time();fr=load_video() if which in('main','all') else None;print('video loaded',time.time()-t0,flush=True)
    T="AI-POWERED 3D BUILDING RECONSTRUCTION";SUB="DRONE PHOTOGRAMMETRY  |  STRUCTURE-FROM-MOTION  |  MULTI-VIEW STEREO"
    if which in('sparse','all'):
        o=Out('output/sparse_point_cloud.mp4');o.add(sc_title(60,"STRUCTURE-FROM-MOTION","SPARSE POINT CLOUD  |  "+T));o.add(sc4(240));o.add(sc_end(60));o.close();print('sparse done',time.time()-t0,flush=True)
    if which in('dense','all'):
        o=Out('output/dense_reconstruction.mp4');o.add(sc_title(60,"MULTI-VIEW STEREO","DENSE RECONSTRUCTION  |  "+T));o.add(sc5(210));o.add(sc6(210));o.add(sc_end(60));o.close();print('dense done',time.time()-t0,flush=True)
    if which in('main','all'):
        o=Out('output/3d_building_reconstruction.mp4');o.add(sc_title(90,T,SUB));o.add(sc1(fr));o.add(sc2());o.add(sc3(150));o.add(sc4(150));o.add(sc5(150));o.add(sc6(150));o.add(sc7(fr,180));o.add(sc_end(90));o.close();print('main done',time.time()-t0,o.n,flush=True)

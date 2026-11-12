# Developer/Creator: tubakhxn
import cv2, numpy as np, pickle, glob, time
files=sorted(glob.glob('reconstruction/frames/f*.jpg'))
sift=cv2.SIFT_create(nfeatures=9000,contrastThreshold=0.02)
K=[];D=[]
t=time.time()
for f in files:
    g=cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2GRAY)
    kp,d=sift.detectAndCompute(g,None)
    K.append(np.float32([k.pt for k in kp])); D.append(d)
print('sift',time.time()-t,np.mean([len(k) for k in K]),flush=True)
n=len(files); pairs=[(i,j) for i in range(n) for j in (i+1,i+2,i+3,i+5,i+8) if j<n]
bf=cv2.BFMatcher(cv2.NORM_L2)
M={}
for c,(i,j) in enumerate(pairs):
    m=bf.knnMatch(D[i],D[j],k=2)
    good=[a for a,b in m if a.distance<0.8*b.distance]
    if len(good)<30: continue
    p1=K[i][[g.queryIdx for g in good]];p2=K[j][[g.trainIdx for g in good]]
    F,mask=cv2.findFundamentalMat(p1,p2,cv2.USAC_MAGSAC,1.0,0.999,10000)
    if F is None: continue
    mask=mask.ravel().astype(bool)
    M[(i,j)]=np.int32([[g.queryIdx,g.trainIdx] for g,k in zip(good,mask) if k])
    if c%20==0: print(i,j,len(good),mask.sum(),time.time()-t,flush=True)
pickle.dump((files,K,D if False else None,M),open('reconstruction/matches.pkl','wb'))
print('pairs',len(M),'mean inliers',np.mean([len(v) for v in M.values()]))

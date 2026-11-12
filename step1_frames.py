# Developer/Creator: tubakhxn
import cv2, numpy as np, json
cap=cv2.VideoCapture('input/original_drone.mp4')
N=int(cap.get(7)); W=1600
cands=[]
for i in range(N):
    ok,f=cap.read()
    if not ok: break
    if i%4: continue
    s=cv2.resize(f,(W,900),interpolation=cv2.INTER_AREA)
    g=cv2.cvtColor(s,cv2.COLOR_BGR2GRAY)
    cands.append((i,cv2.Laplacian(g,cv2.CV_64F).var(),s))
keep=[]
for k in range(0,len(cands),2):
    keep.append(max(cands[k:k+2],key=lambda c:c[1]))
meta=[]
for j,(i,b,s) in enumerate(keep):
    cv2.imwrite(f'reconstruction/frames/f{j:03d}.jpg',s,[cv2.IMWRITE_JPEG_QUALITY,95]); meta.append((j,i,b))
json.dump(meta,open('reconstruction/frames/meta.json','w'))
print(len(cands),'->',len(keep)); print('blur min/med',min(m[2] for m in meta),np.median([m[2] for m in meta]))

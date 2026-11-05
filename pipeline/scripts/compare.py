#!/usr/bin/env python3
"""Frame-wise compare of a test window against the reference v6 cut. Writes PSNR/SSIM stats + contact sheet."""
import argparse, cv2, numpy as np, json
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--ref'); ap.add_argument('--test'); ap.add_argument('--start',type=float); ap.add_argument('--end',type=float); ap.add_argument('--out')
a=ap.parse_args(); FPS=24
r0=round(a.start*FPS); n=round((a.end-a.start)*FPS)
R=cv2.VideoCapture(a.ref); T=cv2.VideoCapture(a.test)
R.set(cv2.CAP_PROP_POS_FRAMES,r0)
# test clip starts at 0 relative to window
psnrs=[]; ssims=[]; thumbs=[]
for i in range(n):
    okR,fr=R.read(); okT,ft=T.read()
    if not (okR and okT): break
    # resize if needed
    if ft.shape!=fr.shape: ft=cv2.resize(ft,(fr.shape[1],fr.shape[0]))
    mse=np.mean((fr.astype(float)-ft.astype(float))**2)
    psnr=99.0 if mse<1e-6 else 10*np.log10(255**2/mse)
    # SSIM on luma, downscaled
    a_=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    b_=cv2.cvtColor(ft,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    a_=cv2.resize(a_,(160,90)); b_=cv2.resize(b_,(160,90))
    mu1,mu2=a_.mean(),b_.mean(); s1,s2=a_.var(),b_.var(); s12=((a_-mu1)*(b_-mu2)).mean()
    ssim=((2*mu1*mu2+1e-3)*(2*s12+1e-3))/((mu1**2+mu2**2+1e-3)*(s1+s2+1e-3))
    psnrs.append(psnr); ssims.append(ssim)
    if i%48==0:
        pair=np.hstack([cv2.resize(fr,(320,180)),cv2.resize(ft,(320,180))])
        cv2.putText(pair,f'{a.start+i/FPS:.1f}s  PSNR {psnr:.1f}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        thumbs.append(pair)
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
stats=dict(frames=len(psnrs),psnr_mean=float(np.mean(psnrs)),psnr_min=float(np.min(psnrs)),ssim_mean=float(np.mean(ssims)),ssim_min=float(np.min(ssims)))
json.dump(stats,open(f'{a.out}_stats.json','w'),indent=1)
if thumbs:
    sheet=np.vstack(thumbs); cv2.imwrite(f'{a.out}_sheet.jpg',sheet)
print(json.dumps(stats,indent=1))

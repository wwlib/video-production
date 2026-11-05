#!/usr/bin/env python3
"""Code de Vie VFX look pass (post only). Reads the clean (subtitle-free) cut, applies two-world grades, world-switch
glitches, tech code overlays, hospital UI callouts and chorus light leaks, keeps faces (YuNet) and the subtitle band clear,
then burns the subtitles ON TOP and muxes the song.
usage: CDV_DATA=~/video-data/code-de-vie python apply_fx.py config.json OUT.mp4 [--start S --end S]"""
import cv2, numpy as np, json, sys, subprocess, argparse, math
ap=argparse.ArgumentParser(); ap.add_argument('cfg'); ap.add_argument('out'); ap.add_argument('--start',type=float,default=0); ap.add_argument('--end',type=float,default=None)
import os
from pathlib import Path
ROOT=Path(os.environ.get('CDV_ROOT', Path(__file__).resolve().parent))
DATA=Path(os.environ.get('CDV_DATA', Path.home()/'video-data'/'code-de-vie'))
def _resolve(p):
    p=Path(p)
    if p.is_absolute() and p.exists(): return str(p)
    for base in (DATA, ROOT, ROOT/'config', Path.cwd()):
        q=base/p
        if q.exists(): return str(q)
    return str(DATA/p) if not Path(p).is_absolute() else str(p)
a=ap.parse_args(); C=json.load(open(a.cfg))
for k in ('src','song','beats','subs'):
    if k in C: C[k]=_resolve(C[k])
if 'future' in C and 'pose_model' in C['future']: C['future']['pose_model']=_resolve(C['future']['pose_model'])
if 'distress' in C:
    for e in C['distress'].get('tracks',[]): e['track']=_resolve(e['track'])
for k in ('code_font',):
    if C.get(k): C[k]=_resolve(C[k])
if C.get('title'):
    for k in ('font','font_small'):
        if C['title'].get(k): C['title'][k]=_resolve(C['title'][k])
FPS=C['fps']; W,H=C['size']
SH=C['shots']; N=SH[-1]['start']+SH[-1]['frames']
f0=round(a.start*FPS); f1=N if a.end is None else min(N,round(a.end*FPS))
beats=np.array(json.load(open(C['beats'])))
def shot_at(n): return next(s for s in SH if s['start']<=n<s['start']+s['frames'])
def wl(s): return ['tech','hospital'] if s['world']=='split' else [s['world']]*2   # [left,right]
# ---- world-switch cuts (left/right halves can differ: split screen)
switch=[s['start'] for i,s in enumerate(SH) if i and wl(s)!=wl(SH[i-1])]
cuts=sorted({min(switch,key=lambda c:abs(c-t*FPS)) for t in C['glitch'].get('cuts_s',[])}) if 'cuts_s' in C['glitch'] else switch
from rembg import new_session, remove
from PIL import Image, ImageFont, ImageDraw
_RS=None
def personmask(fr):   # soft person matte (1 = person), grown, for keeping the glitch off her
    global _RS
    if _RS is None: _RS=new_session('u2net_human_seg')
    m=np.asarray(remove(Image.fromarray(fr[...,::-1]),session=_RS,only_mask=True),np.float32)/255
    m=cv2.dilate((m>0.3).astype(np.float32),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    return cv2.GaussianBlur(m,(0,0),6)
# ---- chorus leaks: chorus downbeats every k beats from section start
leaks=[]
for c0,c1 in C['chorus']:
    bb=beats[(beats>=c0-0.05)&(beats<c1)][::C['leak_every_beats']]; leaks+= [round(b*FPS) for b in bb]
det=cv2.FaceDetectorYN.create(_resolve('models/face_detection_yunet_2023mar.onnx'),'',(W,H),0.5,0.3,50)
rng=np.random.default_rng(3)
yy,xx=np.mgrid[0:H,0:W].astype(np.float32); rr=np.sqrt(((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2)
SUBY=C['subtitle_zone_y']
SUBK=np.clip((SUBY-yy)/60+1,0,1)[...,None]   # v1 tweak: no bloom/halation behind the subtitle band (readability)
def facemask(fr):
    _,r=det.detect(fr); m=np.zeros((H,W),np.float32); boxes=[]
    for f in ([] if r is None else r):
        x,y,w,h=f[:4]; b=(int(x-.35*w),int(y-.45*h),int(x+1.35*w),int(y+1.3*h)); boxes.append(b)
        cv2.rectangle(m,b[:2],b[2:],1,-1)
    return cv2.GaussianBlur(m,(0,0),12),boxes
def grade(img,p):
    x0=img; x=img*np.array(p['gain'][::-1],np.float32)+np.array(p['lift'][::-1],np.float32)*(1-img)
    g=x.mean(2,keepdims=True); x=np.clip(g+(x-g)*p['sat'],0,1)
    if p.get('protect'):   # keep true whites/blacks: grade weighted by a midtone mask of the source luma
        L=x0.mean(2,keepdims=True); w=np.clip(4*L*(1-L),0,1)**0.7; x=x0*(1-w)+x*w
    return x
def tech(img,p,n):
    if p['denoise']: img=img*(1-p['denoise'])+cv2.bilateralFilter(img,5,0.06,3)*p['denoise']
    img=grade(img,p)
    # chromatic fringe grows towards the edges: scale R up, B down about the centre, weighted by radius
    k=p['fringe']; M=lambda s: cv2.getRotationMatrix2D((W/2,H/2),0,s)
    R=cv2.warpAffine(img[...,2],M(1+k),(W,H),borderMode=cv2.BORDER_REFLECT); B=cv2.warpAffine(img[...,0],M(1-k),(W,H),borderMode=cv2.BORDER_REFLECT)
    wgt=np.clip((rr-0.55)/0.6,0,1); img[...,2]=img[...,2]*(1-wgt)+R*wgt; img[...,0]=img[...,0]*(1-wgt)+B*wgt
    img*= (1-p['scan']*((np.arange(H)%3==0)[:,None,None]))              # faint scanlines
    hi=np.clip(img-0.55,0,1); img+=cv2.GaussianBlur(hi,(0,0),14)*p['glow']*np.array([1.0,0.85,0.5],np.float32)  # monitor glow (BGR: blue-ish)
    return img
# ---- v_future: nurse world = Look B (futuristic hyper-real) + angelic glow + hologram mesh moments (no grain)
FUT=C['future']
def _lum(x): return (0.114*x[...,0]+0.587*x[...,1]+0.299*x[...,2])[...,None]
def lookB(img,p):
    x=img*np.array(p['gain_bgr'],np.float32); l=_lum(x)
    x=x+np.clip(1-l*2,0,1)*np.array(p['teal_shadow_bgr'],np.float32)
    x=x+np.clip(l-0.7,0,1)/0.3*(l-x)*p['white_neutral']
    L=_lum(x); bl=cv2.GaussianBlur(L[...,0],(0,0),18)[...,None]
    x=x+p['clarity']*(L-bl)*np.clip(1-np.abs(L-0.5)*1.6,0.2,1)
    x=x+p['sharpen']*(x-cv2.GaussianBlur(x,(0,0),p['sharpen_sigma']))
    x=0.5+(x-0.5)*p['contrast']; g=_lum(x); x=g+(x-g)*p['sat']
    return x
_MESH={}
def mesh_layer(n,pm):
    for i,m in enumerate(FUT['mesh']):
        n0=round(m['t']*FPS); d=n-n0; D=round(m['dur']*FPS)
        if not 0<=d<D: continue
        if i not in _MESH:   # build once at the first frame of the moment (static mesh)
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); x0,y0,x1,y1=m['box']
            reg=np.zeros((H,W),np.uint8); reg[y0:y1,x0:x1]=255
            if m.get('avoid_jen'): reg[pm>0.3]=0
            reg[fm>0.05]=0; reg[SUBY-10:]=0
            pts=cv2.goodFeaturesToTrack(g,m['npts'],0.01,m['mindist'],mask=reg)
            P=[] if pts is None else [tuple(p) for p in pts.reshape(-1,2)]
            for yy_ in range(y0,min(y1,SUBY-10),m['grid']):
                for xx_ in range(x0,x1,m['grid']): P.append((xx_+rng.uniform(-8,8),yy_+rng.uniform(-8,8)))
            P=[q for q in P if 0<=q[0]<W and 0<=q[1]<H and reg[int(q[1]),int(q[0])]]
            sd=cv2.Subdiv2D((0,0,W,H)); [sd.insert((float(q[0]),float(q[1]))) for q in P]
            lay=np.zeros((H,W),np.float32)
            for t in sd.getTriangleList():
                q=t.reshape(3,2)
                if np.max(np.linalg.norm(q-np.roll(q,1,0),axis=1))>m['maxedge']: continue
                cv2.polylines(lay,[np.int32(q)],True,1.0,m.get('thick',1),cv2.LINE_AA)
            for q in P: cv2.circle(lay,(int(q[0]),int(q[1])),2,1.0,-1,cv2.LINE_AA)
            lay*=cv2.GaussianBlur(reg.astype(np.float32)/255,(0,0),10)
            _MESH[i]=lay
        lay=_MESH[i]; e=min(1,(d+1)/6,(D-d)/6)               # fade in/out (6 frames)
        x0,y0,x1,y1=m['box']; sy=int(y0+(y1-y0)*d/max(D-1,1))  # scan line sweep
        scan=np.exp(-((yy-sy)/18)**2)*((xx>=x0)&(xx<x1))*(yy<SUBY-10)
        a=(lay*(1+1.5*scan)*m['opacity']*e*(1-fm))[...,None]; a=a*(1-cv2.GaussianBlur(pm,(0,0),4)[...,None]*m.get('avoid_jen',0))
        col=np.array(m['color_bgr'],np.float32)
        if m.get('dark'):   # darkening pass under the lines so they read on white objects
            x_dark=cv2.GaussianBlur(a[...,0],(0,0),2.5)[...,None]*m['dark']
            return -x_dark*FRAME_NOW, a*col
        glow=cv2.GaussianBlur(a[...,0],(0,0),m.get('glow_sigma',3))[...,None]
        return a*col, glow*col*m.get('glow',0.8)
    return None
import mediapipe as _mp
from mediapipe.tasks.python import vision as _vis, BaseOptions as _BO
_FL=None  # face scan removed in v3
_TES=[(c.start,c.end) for c in _vis.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
def face_scan(x,n):   # holographic face-scan wireframe + glowing irises, tracked per frame
    p=FUT['face_scan']; n0=round(p['t']*FPS); D=round(p['dur']*FPS); d=n-n0
    if not 0<=d<D: return x
    r=_FL.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB,data=np.ascontiguousarray(fr[...,::-1])))
    if not r.face_landmarks: return x
    P=np.array([(l.x*W,l.y*H) for l in r.face_landmarks[0]],np.float32)
    e=min(1,(d+1)/p['fade'],(D-d)/p['fade'])
    lay=np.zeros((H,W),np.float32)
    for a_,b_ in _TES: cv2.line(lay,tuple(np.int32(P[a_])),tuple(np.int32(P[b_])),1.0,1,cv2.LINE_AA)
    for q in P[::3]: cv2.circle(lay,tuple(np.int32(q)),1,1.0,-1,cv2.LINE_AA)
    y0,y1=P[:,1].min(),P[:,1].max(); sy=y0-20+(y1-y0+40)*((d/(D*0.6))%1)   # sweep (repeats)
    lay=lay*(1+2.0*np.exp(-((yy-sy)/14)**2))
    hull=np.zeros((H,W),np.float32); cv2.fillConvexPoly(hull,np.int32(cv2.convexHull(P)),1.0)
    band=np.exp(-((yy-sy)/6)**2)*hull*0.5                                   # bright scan line inside the face
    a=(lay*p['opacity']+band)*e; a[SUBY-6:]=0
    col=np.array(p['color_bgr'],np.float32)
    x=x+a[...,None]*col+cv2.GaussianBlur(a,(0,0),3)[...,None]*col*p['glow']
    eye=np.zeros((H,W),np.float32)
    for c,ring in ((468,(469,470,471,472)),(473,(474,475,476,477))):
        rad=max(2,float(np.mean(np.linalg.norm(P[list(ring)]-P[c],axis=1))))
        cv2.circle(eye,tuple(np.int32(P[c])),int(rad+0.5),1.0,-1,cv2.LINE_AA)
    ee=e*p['eye']; x=x*(1-0.5*eye[...,None]*ee)+eye[...,None]*ee*np.array([1.0,1.0,0.75],np.float32)*0.9
    x=x+cv2.GaussianBlur(eye,(0,0),6)[...,None]*ee*col*2.2+cv2.GaussianBlur(eye,(0,0),18)[...,None]*ee*col*2.5
    return x
_PL=_vis.PoseLandmarker.create_from_options(_vis.PoseLandmarkerOptions(base_options=_BO(model_asset_path=FUT['pose_model']),num_poses=1))
def body_wire(x,n):
    for p in FUT['bodies']:
        n0=round(p['t']*FPS); D=round(p['dur']*FPS); d=n-n0
        if 0<=d<D: return _body(x,n,p,d,D)
    return x
def _body(x,n,p,d,D):   # sparse low-density body wireframe on Jen: mask outline + pose joints, few large triangles
    pm=personmask(fr); m8=(cv2.erode(pm,np.ones((15,15),np.uint8))>0.5).astype(np.uint8)
    cs,_=cv2.findContours(m8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cs: return x
    cs=[c for c in cs if cv2.contourArea(c)>3000] or cs
    pk=p.get('pick','largest')
    if pk=='largest': c=max(cs,key=cv2.contourArea)
    else: c=(min if pk=='left' else max)(cs,key=lambda c:c[:,0,0].mean())
    m8=np.zeros_like(m8); cv2.drawContours(m8,[c],-1,1,-1)   # Jen only
    c=c.reshape(-1,2).astype(np.float32)
    seg=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.r_[c,c[:1]],axis=0),axis=1))]
    k0=int(np.argmin(c[:,1])); c=np.roll(c,-k0,0); seg=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.r_[c,c[:1]],axis=0),axis=1))]
    tt=np.linspace(0,seg[-1],p['contour_pts'],endpoint=False)
    O=np.stack([np.interp(tt,seg,np.r_[c[:,0],c[0,0]]),np.interp(tt,seg,np.r_[c[:,1],c[0,1]])],1)
    r=_PL.detect(_mp.Image(image_format=_mp.ImageFormat.SRGB,data=np.ascontiguousarray(fr[...,::-1])))
    J=[]
    if r.pose_landmarks:
        for i in (11,12,13,14,15,16,23,24,25,26,27,28):
            l=r.pose_landmarks[0][i]
            if (l.visibility or 0)>0.4 and 0<=l.x<1 and 0<=l.y<1 and m8[int(l.y*H),int(l.x*W)]: J.append((l.x*W,l.y*H))
    P=np.r_[O,np.array(J,np.float32).reshape(-1,2)]
    sd=cv2.Subdiv2D((0,0,W,H))
    for q in P: sd.insert((float(np.clip(q[0],0,W-1)),float(np.clip(q[1],0,H-1))))
    lay=np.zeros((H,W),np.float32)
    for t in sd.getTriangleList():
        q=t.reshape(3,2); cx,cy=q.mean(0)
        if 0<=cx<W and 0<=cy<H and m8[int(cy),int(cx)]: cv2.polylines(lay,[np.int32(q)],True,0.7,1,cv2.LINE_AA)
    cv2.polylines(lay,[np.int32(O)],True,1.0,p.get('contour_thick',1),cv2.LINE_AA)
    for q in P: cv2.circle(lay,tuple(np.int32(q)),2,1.0,-1,cv2.LINE_AA)
    e=min(1,(d+1)/p['fade'],(D-d)/p['fade'])
    y0,y1=O[:,1].min(),O[:,1].max(); sy=y0+(y1-y0)*d/max(D-1,1)
    lay=lay*(1+1.5*np.exp(-((yy-sy)/16)**2))
    a=lay*p['opacity']*e*(1-p['face_clear']*fm); a[SUBY-6:]=0
    col=np.array(p['color_bgr'],np.float32)
    return x+a[...,None]*col+cv2.GaussianBlur(a,(0,0),3.5)[...,None]*col*p['glow']
def hosp(img,p,n):
    x=lookB(img,FUT['lookB'])
    gp=FUT['glow']; pm=personmask(fr) if gp['halo'] else np.zeros((H,W),np.float32)
    if gp['halo']:   # halo around her silhouette, outside her (face detail untouched)
        ring=cv2.GaussianBlur(pm,(0,0),gp['halo_sigma'])*(1-pm)*(1-fm)
        x=x+(ring*gp['halo'])[...,None]*np.array(gp['halo_bgr'],np.float32)
    hi=np.clip(_lum(x)-gp['bloom_thr'],0,1)/(1-gp['bloom_thr'])*x
    x=x+cv2.GaussianBlur(hi,(0,0),gp['bloom_sigma'])*gp['bloom']*SUBK*(1-0.6*fm[...,None])   # dreamy window/highlight bloom
    global FRAME_NOW; FRAME_NOW=x
    ml=mesh_layer(n,pm)
    if ml is not None: x=x+ml[0]+ml[1]
    x=body_wire(x,n)
    return x
# ---- code text layer (tech), scrolled at the edges
CODE=["const dose = calc(weight, rx);","if (!allergy.ok) return alert();","await chart.save(note);","patient.vitals.push(hr, spo2);",
      "// zero papier, zero attente","for (const t of tasks) t.done();","export function codeDeVie() {","  return care + time;","}","fetch('/api/mar')","assert(dose < max);","git commit -m 'fewer errors'"]
_im=Image.new('L',(230,2400),0); _d=ImageDraw.Draw(_im); _F=ImageFont.truetype(str(C.get('code_font') or ROOT/'fonts'/'DejaVuSansMono.ttf'),12)
for i in range(120): _d.text((6,8+i*20),CODE[(i*7)%len(CODE)][:28],fill=255,font=_F)
CL=np.asarray(_im,np.uint8)
CL=CL.astype(np.float32)/255
def code_overlay(img,n,alpha,fm,xrange):
    off=(n*1.5)%1200; strip=CL[int(off):int(off)+H]
    for x0 in (8,W-238):
        if not (xrange[0]<=x0+115<xrange[1]): continue
        m=strip*alpha*(1-fm[:,x0:x0+230]); m[SUBY-20:]*=0
        img[:,x0:x0+230]=img[:,x0:x0+230]*(1-m[...,None])+m[...,None]*np.array([0.95,0.85,0.45],np.float32)
    return img
# ---- callouts
CO={}
for c in C['callouts']:
    s=next((x for x in SH if x['shot']==c['shot']),None)
    if s: CO[c['shot']]=dict(c,n0=s['start']+int(s['frames']*c['at']),n1=s['start']+s['frames']-3)
placed={}
def callout(img,n,s,boxes):
    c=CO.get(s['shot'])
    if not c or not (c['n0']<=n<=c['n1']): return img
    tw=cv2.getTextSize(c['text'],cv2.FONT_HERSHEY_DUPLEX,0.6,1)[0][0]; w,h=tw+(52 if c['kind']=='check' else 30),40
    if s['shot'] not in placed:   # first corner (tr, tl, mid-right) clear of faces over the shot
        cand=[(W-w-30,30),(30,30),(W-w-30,300),(30,300)]; placed[s['shot']]=cand[0]
        for x,y in cand:
            if not any(x<b[2] and x+w>b[0] and y<b[3] and y+h>b[1] for b in boxes): placed[s['shot']]=(x,y);break
    x,y=placed[s['shot']]; t=n-c['n0']; al=min(1,t/5)*min(1,(c['n1']-n)/5)*0.85; sc=1+0.25*max(0,1-t/4)
    ov=(np.clip(img,0,1)*255).astype(np.uint8); cx,cy=x+w//2,y+h//2; ww,hh=int(w*sc/2),int(h*sc/2)
    cv2.rectangle(ov,(cx-ww,cy-hh),(cx+ww,cy+hh),(46,31,20),-1,cv2.LINE_AA)
    cv2.rectangle(ov,(cx-ww,cy-hh),(cx-ww+4,cy+hh),(217,191,140) if c['kind']=='vitals' else (115,217,89),-1)
    tx=cx-ww+16
    if c['kind']=='check':
        pr=min(1,t/6); p1,p2,p3=(tx,cy),(tx+7,cy+7),(tx+20,cy-8)
        cv2.line(ov,p1,p2,(128,242,102),3,cv2.LINE_AA)
        if pr>0.4: cv2.line(ov,p2,(int(p2[0]+(p3[0]-p2[0])*min(1,(pr-0.4)/0.6)),int(p2[1]+(p3[1]-p2[1])*min(1,(pr-0.4)/0.6))),(128,242,102),3,cv2.LINE_AA)
        tx+=30
    cv2.putText(ov,c['text'],(tx,cy+7),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    return img*(1-al)+ov.astype(np.float32)/255*al
def glitch(img,prev_img,d,st):
    k=(1-abs(d)/ (C['glitch']['frames']/2+0.5))*st          # 0..1 peak at the cut
    out=img.copy()
    for _ in range(int(3+8*k)):                               # slice displacement
        y=rng.integers(0,H-8); hgt=int(rng.integers(4,30)); dx=int(rng.normal(0,35*k))
        out[y:y+hgt]=np.roll(out[y:y+hgt],dx,1)
        if rng.random()<0.5: out[y:y+hgt,:,2]=np.roll(out[y:y+hgt,:,2],int(8*k),1)
    bs=int(8+24*k)                                             # pixel-dissolve blocks mixing in the other side of the cut
    sm=cv2.resize(cv2.resize(prev_img,(W//bs,H//bs),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
    msk=cv2.resize((rng.random((H//bs+1,W//bs+1))<0.18*k).astype(np.float32),(W,H),interpolation=cv2.INTER_NEAREST)[:H,:W]
    return out*(1-msk[...,None])+sm*msk[...,None]
def leak(img,n,fm):
    for L in leaks:
        t=n-L
        if 0<=t<C['leak']['frames']:
            e=math.sin(math.pi*t/C['leak']['frames'])*C['leak']['strength']; side=(L//7)%2
            gx=np.clip(1-(xx/W if side==0 else 1-xx/W)*1.6,0,1)**2*np.exp(-((yy-H*0.3)/(H*0.6))**2)
            m=(gx*e*(1-0.85*fm))[...,None]*SUBK; col=np.array([0.35,0.65,1.0],np.float32)
            img=1-(1-img)*(1-m*col)                              # screen blend
    return img
# ---- v3 optional layers: EKG pulse line + 'Code de Vie' title with DNA-helix code stream (enabled via config 'ekg' / 'title')
EK=C.get('ekg',{}); TI=C.get('title',{})
SECT=[(0,22.767),(22.767,51.624),(51.624,66.415),(66.415,74.722),(74.722,103.277),(103.277,117.969),(117.969,132.632),(132.632,999)]
def qrs(u):   # u = seconds relative to the beat; P-QRS-T shape, peak (1.0) at u=0
    return (0.12*np.exp(-((u+0.11)/0.025)**2) - 0.15*np.exp(-((u+0.018)/0.008)**2) + 1.0*np.exp(-(u/0.009)**2)
            - 0.3*np.exp(-((u-0.02)/0.01)**2) + 0.22*np.exp(-((u-0.16)/0.04)**2))
def chorus_down(b):
    for c0,c1 in C['chorus']:
        if c0-0.05<=b<c1:
            k=round((b-c0)/C['beat_s']) if 'beat_s' in C else int(np.argmin(np.abs(beats-b))-np.argmin(np.abs(beats-c0)))
            return k%4==0
    return False
if EK.get('bar_downbeats'):   # v4.5: beat 1 of every bar, whole song; bar phase = section starts (verified: chord changes land on this phase)
    _ref=int(np.argmin(np.abs(beats-EK.get('bar_ref_s',22.767)))); BD=((np.arange(len(beats))-_ref)%4==0)
else: BD=np.array([chorus_down(b) for b in beats])
def ekg(img,n,fm,worlds,sx):
    if not EK.get('enabled'): return img
    t=n/FPS; y0=EK.get('y',575); sp=EK.get('speed',420); amp=EK.get('amp',26); big=EK.get('big',1.8)
    xs=np.arange(W,dtype=np.float32); tt=t-(xs-W/2)/sp*-1      # trace scrolls left; centre x shows "now"; right = future
    tt=t+(xs-W/2)/sp*(-1)
    near=beats[(beats>t-W/sp)&(beats<t+W/sp)]; y=np.zeros(W,np.float32); g=np.zeros(W,np.float32)
    for b,bd in zip(near,BD[(beats>t-W/sp)&(beats<t+W/sp)]):
        u=tt-b; a=amp*(big if bd else 1); y+=qrs(u/EK.get('big_sharp',0.65) if bd else u)*a; g+=bd*np.exp(-(u/0.07)**2)
    # fade the trace out towards the frame edges and ahead of 'now' (the sweep head is at centre)
    env=np.clip(1-np.abs(xs-W/2)/(W/2),0,1)**0.6*np.where(xs>W/2,np.clip(1-(xs-W/2)/40,0,1),1)
    lay=np.zeros((H,W),np.uint8); pts=np.stack([xs,y0-y],1).astype(np.int32)
    cv2.polylines(lay,[pts],False,255,EK.get('width',2),cv2.LINE_AA)
    L=lay.astype(np.float32)/255*env[None]
    hd=np.exp(-((xs-W/2)/6)**2)[None]*np.exp(-((np.arange(H)[:,None]-(y0-y[W//2]))/6)**2); L=np.maximum(L,hd)  # sweep head dot
    glow_amt=EK.get('glow',0.5)*(0.35+EK.get('flash',1.3)*g.max()); G=cv2.GaussianBlur(L,(0,0),5)*glow_amt
    if EK.get('white'):   # v4: bright white line + dark soft shadow + optional faint coloured outer glow
        fk=(1-fm)[...,None]; Sd=cv2.GaussianBlur(np.roll(L,2,0),(0,0),3.5)*EK.get('shadow',0.6)
        img=img*(1-np.clip(Sd,0,1)[...,None]*fk)
        Go=cv2.GaussianBlur(L,(0,0),9)*glow_amt; cg=np.array(EK.get('glow_bgr',[255,235,200]),np.float32)/255
        img=img+(Go[...,None]*cg)*fk*0.6
        Aw=np.clip(L*EK.get('alpha',0.95)+G*0.6,0,1)[...,None]*fk
        return img*(1-Aw)+Aw
    A=np.clip(L*EK.get('alpha',0.55)+G,0,1)*(1-fm)
    colT=np.array(EK.get('tech_bgr',[200,230,40]),np.float32)/255; colH=np.array(EK.get('hosp_bgr',[110,125,255]),np.float32)/255
    col=np.where((xs<sx)[:,None],colT if worlds[0]=='tech' else colH,colT if worlds[1]=='tech' else colH)[None]
    return img*(1-A[...,None])+col*A[...,None]
# title
_TF=ImageFont.truetype(str(TI.get('font') or ROOT/'fonts'/'DejaVuSansMono-Bold.ttf'),TI.get('size',54)) if TI.get('enabled') else None
HCH=['const','=>','{}','()',';','===','let','async','await','return','fn','[]','&&','++']+['let life = heal()','console.log(life)','async () => {','return care;','await heal(pt)','function life()','const dose = rx;']
_tpos=None
def title(img,n,fr,boxes):
    global _tpos
    if not TI.get('enabled'): return img
    t=n/FPS; mom=TI.get('moments') or [TI]
    M_=next((m for m in mom if m['type_start']<=t<m['end']),None)
    if M_ is None: _tpos=None; return img
    t0,t1,tt,te=M_['type_start'],M_['type_end'],M_['tear_at'],M_['end']
    txt=TI.get('text','Code de Vie')
    if _tpos is None:   # first candidate rect clear of faces (upper area, above subs)
        tw,th=_TF.getbbox(txt)[2]+40,_TF.getbbox(txt)[3]+10
        cand=TI.get('pos_candidates',[[60,60],[W-tw-60,60],[60,250],[W-tw-60,250]]); _tpos=cand[0]
        for x,y in cand:
            if not any(x<b[2] and x+tw>b[0] and y<b[3] and y+th>b[1] for b in boxes): _tpos=[x,y];break
    x,y=_tpos
    # ---- helix code stream (behind the title), masked by the person matte and faces
    h0,h1=M_['helix_start'],M_['helix_end']
    if h0<=t<h1:
        e=min(1,(t-h0)/0.25,(h1-t)/0.3); lay=Image.new('L',(W,H),0); d=ImageDraw.Draw(lay); F=ImageFont.truetype(str(TI.get('font_small') or ROOT/'fonts'/'DejaVuSansMono.ttf'),15)
        cx=x+TI.get('helix_dx',_TF.getbbox(txt)[2]//2); r=TI.get('helix_r',70); rng2=np.random.default_rng(11)
        for k in range(46):
            yy_=int((k*16+ (t-h0)*260)%(H-150))   # falling
            ph=yy_/38.0+ (t-h0)*2.5
            xa,xb=cx+r*np.sin(ph),cx+r*np.sin(ph+np.pi); za,zb=np.cos(ph),np.cos(ph+np.pi)
            ch=HCH[(k*5+int((t-h0)*12))%14]
            for xx_,zz in ((xa,za),(xb,zb)): d.text((int(xx_)-4*len(ch),yy_),ch,fill=int(130+110*(zz+1)/2),font=F)
            if k%3==0:   # rung
                for j in (1,3,5): d.text((int(xa+(xb-xa)*j/6)-4,yy_),'=;.(),'[(k+j)%6],fill=90,font=F)
            if k%7==0: d.text((int(cx+r+18),yy_),HCH[14+(k//7)%7],fill=150,font=F)
        A=np.asarray(lay,np.float32)/255*e*TI.get('helix_alpha',0.6)
        trail=cv2.GaussianBlur(A,(0,0),3)*0.5; A=np.clip(A+trail,0,1)
        pm=np.maximum(personmask(fr),FM_NOW); A*=(1-pm); A[C['subtitle_zone_y']-30:]=0
        bgL=cv2.GaussianBlur(img.mean(2),(0,0),9)[...,None]; dk=np.clip((bgL-0.45)/0.25,0,1)   # bright bg -> deep teal ink
        col=np.array([0.55,1.0,0.45],np.float32)*(1-dk)+np.array([0.42,0.38,0.05],np.float32)*dk
        img=img*(1-A[...,None])+col*A[...,None]
    # ---- typed title + cursor; glitch-tear on the title only at tear_at, then resolve
    nch=int(np.clip((t-t0)/(t1-t0),0,1)*len(txt)); s=txt[:nch]
    cur='_' if (int(t*4)%2==0 or t<t1) else ' '
    lay=Image.new('L',(W,H),0); d=ImageDraw.Draw(lay); d.text((x,y),s+cur,fill=255,font=_TF)
    T=np.asarray(lay,np.float32)/255
    if tt<=t<tt+TI.get('tear_dur',0.25):
        k=1-(t-tt)/TI.get('tear_dur',0.25); rr2=np.random.default_rng(n); bb=T.copy()
        for _ in range(7):
            y1=int(rr2.integers(y,y+70)); h=int(rr2.integers(3,14)); T[y1:y1+h]=np.roll(bb[y1:y1+h],int(rr2.normal(0,40*k)),1)
        Rch=np.roll(T,int(6*k),1); Bch=np.roll(T,-int(6*k),1)
    else: Rch=Bch=T
    fade=min(1,(te-t)/0.3); a=TI.get('alpha',0.92)*fade
    base=np.array([1.0,1.0,0.75],np.float32)
    glow=cv2.GaussianBlur(T,(0,0),6)*0.5
    Tm=np.stack([Bch,T,Rch],2)*a; Gm=glow[...,None]*a*np.array([0.9,0.9,0.4],np.float32)
    Mx=np.maximum(np.maximum(Bch,T),Rch)*a
    face_keep=(1-FM_NOW)[...,None]
    bgL=cv2.GaussianBlur(img.mean(2),(0,0),12)[...,None]; dk=np.clip((bgL-0.45)/0.25,0,1)
    sh=cv2.GaussianBlur(np.roll(np.roll(T,3,0),3,1),(0,0),4)[...,None]*a*0.55*face_keep   # soft drop shadow for bright frames
    img=img*(1-sh)
    ink=np.array([1.0,1.0,0.75],np.float32)*(1-dk)+np.array([0.45,0.40,0.06],np.float32)*dk
    Tc=Tm*ink; Gm=Gm*(1-dk)
    return img*(1-Mx[...,None]*face_keep)+(Tc+Gm)*face_keep


# ---- v5: coder-shot tech distress (approved coder_distress_test_v2), applied before EKG/titles; subs burned later
DZ=C['distress']; _dr=np.random.default_rng(5); _DF=ImageFont.truetype(str(ROOT/'fonts'/'DejaVuSansMono-Bold.ttf'),30)
def _to720(q): return [[x*1280/832, y*1280/832-(1280*480/832-720)/2] for x,y in q]
DTR={}
for e_ in DZ['tracks']:
    t_=json.load(open(e_['track']))['tracks']; DTR[e_['shot']]={k:[_to720(c) for c in v['corners']] for k,v in t_.items()}
DHITS=[(round(h['t']*FPS),h['k'],h['n'],h.get('full',0)) for h in DZ['hits']]
def _poly(q,erode=True):
    m=np.zeros((H,W),np.float32); cv2.fillPoly(m,[np.int32(q)],1.0)
    if erode: m=cv2.erode(m,np.ones((7,7),np.uint8))
    m[SUBY:]=0; return cv2.GaussianBlur(m,(0,0),1.5)
def distress(out,n,s,fr):
    hit=[h for h in DHITS if 0<=n-h[0]<h[2]]
    mons=DTR.get(s['shot'])
    if not mons and not hit: return out
    pm=personmask(fr)
    if mons:
        i=n-s['start']; sm=np.zeros((H,W),np.float32); per={}
        for k,v in mons.items():
            per[k]=_poly(v[min(i,len(v)-1)]); sm=np.maximum(sm,per[k])
        sm=sm*(1-pm)
        sc=out*(1-DZ['scan']*((np.arange(H)%3==0)[:,None,None]))
        j=np.roll(sc,int(_dr.integers(-DZ['jitter_px'],DZ['jitter_px']+1)),1)
        for _b in range(int(_dr.poisson(DZ['band_rate']))):
            y=int(_dr.integers(0,SUBY-30)); h=int(_dr.integers(3,24)); bs_=DZ.get('band_shift',40); j[y:y+h]=np.roll(j[y:y+h],int(_dr.integers(-bs_,bs_+1)),1)
        out=out*(1-sm[...,None])+j*sm[...,None]
        for rf in DZ['red_flashes']:
            if s['shot']==rf['shot'] and rf['t0']*FPS<=n<rf['t0']*FPS+rf['frames'] and rf['monitor'] in per:
                k=n-round(rf['t0']*FPS); a=[1,0.6,1,0.75,0.5,0.25][min(k,5)]*DZ.get('flash_op',1.0)
                mm=per[rf['monitor']]*(1-pm)*a              # v5 fix: only the intended screen
                red=out*np.array([0.25,0.25,1.0],np.float32)+np.array([0,0,0.12],np.float32)
                out=out*(1-mm[...,None])+red*mm[...,None]
                q=np.int32(mons[rf['monitor']][min(i,len(mons[rf['monitor']])-1)]); x0,y0=max(8,q[:,0].min()+30),max(8,q[:,1].min()+30)
                lay=Image.new('L',(W,H),0); ImageDraw.Draw(lay).text((int(x0),int(y0)),rf['text'],font=_DF,fill=255)
                L=np.asarray(lay,np.float32)/255*per[rf['monitor']]*a*(1-pm); out=out*(1-L[...,None])+np.array([0.35,0.35,1.0],np.float32)*L[...,None]
    for h,K,HL,full in hit:
        d=n-h; k=K*DZ.get('amp',1.0)*[1.0,0.8,0.55,0.3,0.2][min(d,4)]; g=out.copy()
        for _ in range(int(10+16*k)):
            y=int(_dr.integers(0,SUBY-10)); hh=int(_dr.integers(4,40)); g[y:y+hh]=np.roll(g[y:y+hh],int(_dr.normal(0,90*k)),1)
        sh=int(14*k); g[...,2]=np.roll(g[...,2],sh,1); g[...,0]=np.roll(g[...,0],-sh,1)
        bs=16; s2=cv2.resize(cv2.resize(g,(W//bs,H//bs),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
        mb=cv2.resize((_dr.random((H//bs+1,W//bs+1))<0.22*k).astype(np.float32),(W,H),interpolation=cv2.INTER_NEAREST)[:H,:W]
        g=g*(1-mb[...,None])+s2*mb[...,None]
        keep=pm*(0 if (full and d==0) else 1); keep=keep.copy(); keep[SUBY:]=1
        out=g*(1-keep[...,None])+out*keep[...,None]
    return out
# ---- io
cap=cv2.VideoCapture(C['src']); cap.set(cv2.CAP_PROP_POS_FRAMES,f0)
tmp=a.out+'.fx_nosubs.mp4'
enc=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-c:v','libx264','-crf','14','-preset','medium','-pix_fmt','yuv420p',tmp],stdin=subprocess.PIPE)
buf={}
def raw(n):
    if n not in buf:
        c2=cv2.VideoCapture(C['src']); c2.set(cv2.CAP_PROP_POS_FRAMES,n); ok,f=c2.read(); buf[n]=f.astype(np.float32)/255
    return buf[n]
G=C['glitch']['frames']
for n in range(f0,f1):
    ok,fr=cap.read()
    if not ok: break
    s=shot_at(n); fm,boxes=facemask(fr); img=fr.astype(np.float32)/255
    L,R=wl(s); sx=C['split_x'] if s['world']=='split' else W
    out=np.empty_like(img)
    for w,(x0,x1) in ((L,(0,sx)),(R,(sx,W))):
        if x1<=x0: continue
        part=tech(img.copy(),C['tech'],n) if w=='tech' else hosp(img.copy(),C['hospital'],n)
        if w=='tech': part=code_overlay(part,n,C['tech']['code_alpha'],fm,(x0,x1))
        out[:,x0:x1]=part[:,x0:x1]
    out=out*(1-fm[...,None])+np.clip(out,0,1)*fm[...,None]
    out=callout(out,n,s,boxes)
    out=leak(out,n,fm)
    FM_NOW=fm
    for c in cuts:
        d=n-c
        if -G//2<=d<G-G//2:
            other=raw(c-1) if d>=0 else raw(min(c,N-1))
            gl=glitch(out,other,d+0.5,C['glitch']['strength'])
            pm=np.maximum(personmask(fr),fm)[...,None]   # v2: person matte -> no blocks on her
            out=gl*(1-pm)+out*pm
    if s['world']=='tech': out=distress(out,n,s,fr)
    out=ekg(out,n,fm,(L,R),sx)
    out=title(out,n,fr,boxes)
    if len(buf)>8: buf.clear()
    enc.stdin.write((np.clip(out,0,1)*255+0.5).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()
t0=f0/FPS; dur=(f1-f0)/FPS
subs=C['subs']
if C.get('subs_style'):   # restyled copy (shadow/outline) of the subtitle master
    import re; L=open(subs).read().splitlines(); ss=C['subs_style']
    for i,l in enumerate(L):
        if l.startswith('Format:') and 'Fontname' in l: keys=[k.strip() for k in l[7:].split(',')]
        if l.startswith('Style:'):
            v=[x.strip() for x in l[6:].split(',')]
            for k,val in ss.items(): v[keys.index(k)]=str(val)
            L[i]='Style: '+','.join(v)
    subs=a.out+'.subs.ass'; open(subs,'w').write('\n'.join(L)+'\n')
subprocess.run(['ffmpeg','-y','-v','error','-i',tmp,'-i',C['song'],'-filter_complex',
  f"[0:v]setpts=PTS+{t0}/TB,ass='{subs}',setpts=PTS-STARTPTS[v];[1:a]atrim={t0}:{t0+dur},asetpts=PTS-STARTPTS,afade=t=in:d=0.05,afade=t=out:st={dur-0.5}:d=0.5[a]",
  '-map','[v]','-map','[a]','-r',str(FPS),'-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','256k','-t',f'{dur:.4f}','-movflags','+faststart',a.out],check=True)
import os; os.remove(tmp); os.remove(subs) if subs!=C['subs'] else None
print('frames',f0,f1,'cuts',[c for c in cuts if f0<=c<f1],'leaks',[l for l in leaks if f0<=l<f1])

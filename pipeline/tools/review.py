# review overlay (same look as cut_A1/B1): SHOT NN SECTION + MM:SS:FF, tl -> tr -> bl if a YuNet face overlaps
import sys,json,cv2,subprocess
src,out,plans=sys.argv[1],sys.argv[2],sys.argv[3:]
sec={s['shot']:s['section'].upper() for s in json.load(open('/workspace/poc/storyboard/shots.json'))['shots']}
P=[];off=0
for pf in plans:
    for p in json.load(open(pf)): P.append((p['shot'],off+p['start_frame'],p['frames']))
    off=P[-1][1]+P[-1][2]
det=cv2.FaceDetectorYN.create('/workspace/poc/full/lib/models/face_detection_yunet_2023mar.onnx','',(1280,720),0.55,0.3,5000)
cap=cv2.VideoCapture(src)
def faces(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,f=cap.read()
    if not ok: return []
    _,r=det.detect(f);return [] if r is None else [(x-.15*w,y-.2*h,x+1.15*w,y+1.15*h) for x,y,w,h in r[:,:4]]
POS={'tl':(14,14),'tr':(1280-246,14),'bl':(14,720-70)}
F="/usr/share/fonts/truetype/dejavu/DejaVuSansMono"
L=[];moved=[]
for s,a,n in P:
    fb=sum((faces(a+k*n//4) for k in range(4)),[])
    pos='tl'
    for c in ('tl','tr','bl'):
        x,y=POS[c]
        if not any(x<f[2] and x+232>f[0] and y<f[3] and y+56>f[1] for f in fb): pos=c;break
    if pos!='tl': moved.append(f'{s}->{pos}')
    x,y=POS[pos];e=f"enable='between(n,{a},{a+n-1})'"
    L+= [f"drawbox=x={x}:y={y}:w=232:h=56:color=black@0.50:t=fill:{e}",f"drawbox=x={x}:y={y}:w=4:h=56:color=0xD97757@0.9:t=fill:{e}",
         f"drawtext=fontfile={F}-Bold.ttf:text='SHOT {s:02d}  {sec[s]}':x={x+14}:y={y+8}:fontsize=17:fontcolor=white@0.95:{e}",
         f"drawtext=fontfile={F}.ttf:text='%{{eif\\:floor(n/1440)\\:d\\:2}}\\:%{{eif\\:mod(floor(n/24)\\,60)\\:d\\:2}}\\:%{{eif\\:mod(n\\,24)\\:d\\:2}}':x={x+14}:y={y+32}:fontsize=16:fontcolor=white@0.80:{e}"]
ft=out+'.filter.txt';open(ft,'w').write(',\n'.join(L))
subprocess.run(['ffmpeg','-y','-v','error','-i',src,'-filter_script:v',ft,'-c:v','libx264','-crf','20','-preset','medium','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',out],check=True)
print('moved',moved)

#!/usr/bin/env bash
# Retime/fit each generated clip into its beat slot at 24 fps / 1280x720.
set -euo pipefail
PIPE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CDV_DATA:-$HOME/video-data/code-de-vie}"
ONLY="${1:-}"
python3 - <<PY
import json,os,subprocess,sys
from pathlib import Path
try:
  import yaml; doc=yaml.safe_load(open("$PIPE/shots.yaml"))
except Exception:
  doc=json.load(open("$PIPE/shots.json"))
DATA=Path("$DATA"); only="$ONLY".split(",") if "$ONLY" else None
for s in doc["shots"]:
  if only and s["id"] not in only and str(s["shot"]) not in only: continue
  src=DATA/"gen"/s["id"]/f'{s["id"]}.mp4'
  if not src.exists():
    print("missing",src); continue
  outdir=DATA/"conform"/s["id"]; outdir.mkdir(parents=True,exist_ok=True)
  out=outdir/f'{s["id"]}_24fps_720p.mp4'
  n=s["frames"]
  if out.exists() and out.stat().st_mtime>=src.stat().st_mtime:
    print("skip",out); continue
  # scale+center crop+pad/trim to exact frame count
  vf=f"setpts=PTS-STARTPTS,fps=24,scale=1280:-2,crop=1280:720,setsar=1,format=yuv420p"
  # tpad if short
  have=int(subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(src)],capture_output=True,text=True).stdout or 0)
  if have and have<n: vf+=f",tpad=stop_mode=clone:stop={n-have}"
  subprocess.run(["ffmpeg","-y","-v","error","-i",str(src),"-vf",vf,"-an","-r","24","-frames:v",str(n),"-c:v","libx264","-crf","16","-preset","medium",str(out)],check=True)
  print("wrote",out,"frames",n,"from",have)
PY

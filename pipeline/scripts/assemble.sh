#!/usr/bin/env bash
# Concat conformed (or vscode) clips for the given shot ids into a silent video, mux song audio for that span.
set -euo pipefail
PIPE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CDV_DATA:-$HOME/video-data/code-de-vie}"
IDS="${1:?usage: assemble.sh shot_02,shot_04}"
OUT="${2:-$DATA/assembled/test_cut.mp4}"
python3 - <<PY
import json,subprocess,os
from pathlib import Path
try:
  import yaml; doc=yaml.safe_load(open("$PIPE/shots.yaml"))
except Exception:
  doc=json.load(open("$PIPE/shots.json"))
DATA=Path("$DATA"); ids="$IDS".split(",")
by={s["id"]:s for s in doc["shots"]}
clips=[]; t0=None; t1=None
for i in ids:
  s=by[i]; t0=s["start"] if t0 is None else t0; t1=s["end"]
  # Prefer conformed 720p (includes vscode content when scripts/vscode_comp + conform were run).
  conf=DATA/"conform"/s["id"]/f'{s["id"]}_24fps_720p.mp4'
  p=conf
  if not p.exists(): raise SystemExit(f"missing clip for {i}: {p}")
  clips.append(p)
tmpdir=DATA/"assembled"/("_tmp")
tmpdir.mkdir(parents=True,exist_ok=True)
lst=tmpdir/"list.txt"
lst.write_text("".join(f"file '{c}'\n" for c in clips))
silent=tmpdir/"video.mp4"
subprocess.run(["ffmpeg","-y","-v","error","-f","concat","-safe","0","-i",str(lst),"-c","copy",str(silent)],check=True)
audio=DATA/"audio"/"code-de-vie_v4_5_reverbtail.wav"
out=Path("$OUT"); out.parent.mkdir(parents=True,exist_ok=True)
dur=t1-t0
subprocess.run(["ffmpeg","-y","-v","error","-i",str(silent),"-ss",str(t0),"-i",str(audio),
  "-filter_complex",f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,afade=t=out:st={max(0,dur-0.3)}:d=0.3[a]",
  "-map","0:v","-map","[a]","-c:v","copy","-c:a","aac","-b:a","192k","-shortest","-movflags","+faststart",str(out)],check=True)
print("wrote",out,"span",t0,t1,"dur",dur)
PY

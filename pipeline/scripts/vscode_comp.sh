#!/usr/bin/env bash
# Run vscomp on shots that have screen tracks. Expects conformed clips.
set -euo pipefail
PIPE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CDV_DATA:-$HOME/video-data/code-de-vie}"
export VSCOMP_ROOT="$DATA/shots"
export VSCOMP_CFGDIR="$PIPE/tools/vscomp"
export SCREENS="$DATA/plates"
# Point each shot folder at the conformed clip name vscomp expects
python3 - <<PY
import json,os,subprocess,shutil
from pathlib import Path
try:
  import yaml; doc=yaml.safe_load(open("$PIPE/shots.yaml"))
except Exception:
  doc=json.load(open("$PIPE/shots.json"))
DATA=Path("$DATA"); PIPE=Path("$PIPE")
only=os.environ.get("ONLY","").split(",") if os.environ.get("ONLY") else None
py=str(PIPE/"../pipeline/.venv/bin/python") if (PIPE/"../pipeline/.venv/bin/python").exists() else str(PIPE/".venv/bin/python")
for s in doc["shots"]:
  if not s.get("screen_track"): continue
  if only and s["id"] not in only and str(s["shot"]) not in only: continue
  gen=DATA/"gen"/s["id"]/f'{s["id"]}.mp4'
  if not gen.exists():
    print("skip vscode: no gen",s["id"]); continue
  shot_dir=DATA/"shots"/f'shot_{s["shot"]:02d}'
  shot_dir.mkdir(parents=True,exist_ok=True)
  # vscomp expects native-resolution *_clip_24fps_grain.mp4 (track coords are in that space)
  dst=shot_dir/f'shot_{s["shot"]:02d}_clip_24fps_grain.mp4'
  shutil.copy2(gen,dst)
  env={**os.environ,"VSCOMP_ROOT":str(DATA/"shots"),"VSCOMP_CFGDIR":str(PIPE/"tools/vscomp"),"SCREENS":str(DATA/"plates")}
  # Prefer existing track if present
  cmd=[py,str(PIPE/"tools/vscomp/vscomp.py"),"comp",str(s["shot"]),"--force"]
  print("run",cmd)
  r=subprocess.run(cmd,env=env)
  if r.returncode!=0:
    print("comp failed; trying track+comp")
    subprocess.run([py,str(PIPE/"tools/vscomp/vscomp.py"),"run",str(s["shot"])],env=env,check=False)
PY

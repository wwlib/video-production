#!/usr/bin/env bash
# Entry point: ./run.sh test | section | v6
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="${CDV_DATA:-$HOME/video-data/code-de-vie}"
export CDV_ROOT="$ROOT" CDV_DATA="$DATA"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
CFG="$ROOT/config/v6.json"
OUTDIR="$DATA/out"
mkdir -p "$OUTDIR"
cmd="${1:-test}"

need() { [[ -f "$1" ]] || { echo "missing $1 — see README data layout" >&2; exit 1; }; }
need "$DATA/video_v4_3.mp4"
need "$DATA/audio/code-de-vie_v4_5_reverbtail.wav"
need "$DATA/subs/code_de_vie_v4_2_subs_en.ass"
need "$DATA/code_de_vie_FINAL.mp4"
need "$DATA/models/pose_landmarker_full.task"
need "$DATA/models/face_detection_yunet_2023mar.onnx"
# make kit-relative assets visible via DATA too
ln -sfn "$ROOT/tracks" "$DATA/tracks"
ln -sfn "$ROOT/fonts" "$DATA/fonts"
ln -sfn "$ROOT/config/beats_v3.json" "$DATA/beats_v3.json"
rm -rf "$ROOT/models"
ln -sfn "$DATA/models" "$ROOT/models"

finish_tail() {
  # Append FINAL's last 6.4 s (Team XRay credit + flatline + reverb) from frame 3732
  local body="$1" out="$2"
  ffmpeg -y -v error -i "$body" -i "$DATA/code_de_vie_FINAL.mp4" \
    -filter_complex "[1:v]trim=start_frame=3732,setpts=PTS-STARTPTS[t];[0:v][t]concat=n=2:v=1:a=0,fps=24[v]" \
    -map "[v]" -map 1:a -c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p -c:a copy \
    -movflags +faststart "$out"
}

case "$cmd" in
  test)
    # ~20 s spanning coder (shots 2–3) and nurse (shot 4) worlds
    echo "Render test window 3.0–23.0 s ..."
    python "$ROOT/apply_fx.py" "$CFG" "$OUTDIR/test_body.mp4" --start 3 --end 23
    # no credit tail for the short test; mux is already in apply_fx
    mv "$OUTDIR/test_body.mp4" "$OUTDIR/test_section.mp4"
    echo "Wrote $OUTDIR/test_section.mp4"
    python "$ROOT/scripts/compare.py" --ref "$DATA/code_de_vie_v6_ref.mp4" --test "$OUTDIR/test_section.mp4" --start 3 --end 23 --out "$OUTDIR/test_compare"
    ;;
  section)
    python "$ROOT/apply_fx.py" "$CFG" "$OUTDIR/sec_body.mp4" --start "${2:-53.46}" --end "${3:-72.42}"
    echo "Wrote $OUTDIR/sec_body.mp4"
    ;;
  v6)
    echo "Full FX body 0–155.5 s ..."
    python "$ROOT/apply_fx.py" "$CFG" "$OUTDIR/v6_body.mp4" --start 0 --end 155.5
    finish_tail "$OUTDIR/v6_body.mp4" "$OUTDIR/code_de_vie_v6.mp4"
    ffprobe -v error -show_entries format=duration:stream=nb_frames,codec_type -of compact "$OUTDIR/code_de_vie_v6.mp4"
    echo "Wrote $OUTDIR/code_de_vie_v6.mp4"
    ;;
  *)
    echo "usage: $0 test | section [start end] | v6" >&2; exit 2
    ;;
esac

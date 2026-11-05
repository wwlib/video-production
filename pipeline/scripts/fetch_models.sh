#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${CDV_DATA:-$HOME/video-data/code-de-vie}"
mkdir -p "$DATA/models" "$ROOT/models"
cd "$DATA/models"
dl() {
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then echo "have $out"; return; fi
  echo "fetch $out"
  curl -fsSL -o "$out.part" "$url" && mv "$out.part" "$out"
}
# MediaPipe Tasks models (Google)
dl https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task pose_landmarker_full.task
dl https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task face_landmarker.task
# YuNet face detector (OpenCV Zoo)
if [[ ! -f face_detection_yunet_2023mar.onnx ]]; then
  if [[ -f "$ROOT/models/face_detection_yunet_2023mar.onnx" ]]; then
    cp "$ROOT/models/face_detection_yunet_2023mar.onnx" .
  else
    dl https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx face_detection_yunet_2023mar.onnx
  fi
fi
# rembg human-seg weight is downloaded on first rembg call into ~/.u2net or ~/.rembg
echo "Models ready in $DATA/models"
ls -lh

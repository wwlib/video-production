# Code de Vie — local post pipeline

Team XRay. Portable rebuild of the approved **v6** look pass for *Code de Vie*.

## What runs where

| Stage | Where |
|---|---|
| Shot generation (Wan / InfiniteTalk / Qwen-Image) | NVIDIA PC via **ComfyUI Studio** (`~/github/comfyui-studio`) |
| Conform, VS Code screen composites, lip-sync assembly → clean cut | Historically on the render box; the clean result is `video_v4_3.mp4` |
| **This kit** — per-world grade, coder distress, nurse meshes/halo, EKG, titles, subtitles, credit ending | **Mac mini (Apple Silicon)** or any machine with the venv |

This folder rebuilds `code_de_vie_v6.mp4` from the clean picture + audio + overlays. It does **not** regenerate shots.

## Layout

```
pipeline/                 # this folder (in the video-production repo)
  apply_fx.py             # v6 look pass (portable paths)
  config/v6.json          # approved parameters
  config/beats_v3.json    # ~129.9 bpm grid
  tracks/                 # monitor corner tracks for distress
  fonts/                  # DejaVu Sans Mono
  setup.sh / run.sh
  scripts/fetch_models.sh
  scripts/compare.py
  requirements.txt
  package.json            # optional npm wrappers

~/video-data/code-de-vie/ # LOCAL, not in git
  video_v4_3.mp4          # clean picture (no subtitles)
  code_de_vie_FINAL.mp4   # for the credit/flatline tail + audio master
  code_de_vie_v6_ref.mp4  # optional reference for compare
  audio/code-de-vie_v4_5_reverbtail.wav
  subs/code_de_vie_v4_2_subs_en.ass
  models/                 # YuNet + MediaPipe pose (fetched)
  out/                    # renders land here
```

Override the data root with `export CDV_DATA=/path/to/code-de-vie`.

## Setup (macOS Apple Silicon)

```bash
cd ~/github/video-production/pipeline
./setup.sh          # or: npm run setup
# expects Homebrew ffmpeg; creates .venv; pip installs; fetches models
```

System: **ffmpeg** (Homebrew), **Python 3.11+**. No CUDA required (CPU / onnxruntime).

## Commands

```bash
./run.sh test                 # ~20 s (3–23 s): coder + nurse; compare to v6_ref
./run.sh section 53.46 72.42  # arbitrary window
./run.sh v6                   # full film + Team XRay credit ending
# npm equivalents: npm test / npm run section / npm run final
```

## What the look pass does

- **Coder shots:** scanlines/jitter inside tracked screens; beat-locked tears (bar-1 big, beats 3–4 smaller); red test-failed flashes on one screen; person matte keeps Jen clear except one-frame full-frame hits.
- **Nurse shots:** futuristic hyper-real grade, angelic halo, window bloom; sparse 1 px hologram meshes on distinct shapes; occasional Jen body wireframe (pose + silhouette).
- **Shared:** EKG on the beat grid, “Code de Vie” titles + JS helix, chorus light leaks, styled subtitles, FINAL audio with reverb tail, credit ending from FINAL frames 3732+.

## Future projects

Copy `pipeline/` → `pipeline-<project>/`, point `CDV_DATA` at a new `~/video-data/<project>/`, replace `config/v6.json`, tracks and beats. Keep generation on ComfyUI Studio; keep post here.

## Credits

Team XRay · Produced with Grok Bot

## Verified on Mac mini (M4, 2026-11-05)

Setup: Homebrew ffmpeg, Python 3.11 venv, `./setup.sh` (~1–2 min + rembg weight download on first run).

**Short test** (`./run.sh test`, window 3–23 s, coder + nurse):
- Runtime: ~4.9 min wall
- vs box `code_de_vie_v6_ref.mp4`: PSNR mean **33.4 dB** (min 13.9 on a distress hit), SSIM mean **0.983** (min 0.62 on the same hit). Contact sheet: `$CDV_DATA/out/test_compare_sheet.jpg`.

**Full v6** (`./run.sh v6`):
- Runtime: ~39 min wall for the FX body on M4, plus ~30 s to append the credit tail
- Output: 161.917 s, **3886** frames, audio from FINAL (matches box ref / FINAL exactly on duration and frame count)
- Spot check vs box ref (every ~10 s + credit frames): PSNR mean **35.1 dB** (min 24.5), SSIM mean **0.989** (min 0.956)
- Artifact: `$CDV_DATA/out/code_de_vie_v6.mp4`

## Gaps / not in this kit

- Shot generation and ComfyUI graphs stay on the NVIDIA PC (`~/github/comfyui-studio`).
- Upstream conform / VS Code compositing / lip-sync assembly that produced `video_v4_3.mp4` are not re-run here; feed that clean cut as input.
- Making-of HTML lives in `docs/` of this repo (already published); this kit does not rebuild the site.
- RNG-based distress tears will not bit-match the box render frame-for-frame; use PSNR/SSIM, not pixel equality.

#!/usr/bin/env bash
# Build the full demo: capture app footage → Remotion intro/outro render →
# high-quality MP4 + GIF for the README.
#
# Prereqs: docker compose stack healthy on :7007, Python venv with
# playwright + chromium, Node + `npm install` already run in video/.
#
#   scripts/build_demo.sh
#   # → docs/demo.mp4  (1080p, branded intro + footage + outro)
#   # → docs/demo.gif  (README hero)
#   # → docs/screenshot.png
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
GIF_WIDTH="${GIF_WIDTH:-1000}"
GIF_FPS="${GIF_FPS:-15}"

echo "==> 1/4  Capturing 1080p app footage with Playwright"
"$PY" scripts/capture_demo.py

echo "==> 2/4  Installing Remotion deps (if needed)"
[ -d video/node_modules ] || npm --prefix video install

echo "==> 3/4  Rendering branded video with Remotion → docs/demo.mp4"
npm --prefix video run render

echo "==> 4/4  Encoding high-quality GIF → docs/demo.gif"
PAL="$(mktemp --suffix=.png)"
VF="fps=${GIF_FPS},scale=${GIF_WIDTH}:-1:flags=lanczos"
ffmpeg -y -i docs/demo.mp4 -vf "${VF},palettegen=max_colors=200" "$PAL" \
  >/dev/null 2>&1
ffmpeg -y -i docs/demo.mp4 -i "$PAL" \
  -lavfi "${VF}[x];[x][1:v]paletteuse=dither=sierra2_4a" \
  -loop 0 docs/demo.gif >/dev/null 2>&1
rm -f "$PAL"
if command -v gifsicle >/dev/null 2>&1; then
  gifsicle -O3 --lossy=40 -o docs/demo.gif docs/demo.gif 2>/dev/null || true
fi

echo "==> done:"
ls -lh docs/demo.mp4 docs/demo.gif docs/screenshot.png

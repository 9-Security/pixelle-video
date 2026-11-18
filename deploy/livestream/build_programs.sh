#!/usr/bin/env bash
# Build stream-ready programs from rendered loop clips with ffmpeg only (no Python environment needed).
# Same output as `python -m pixelle_video.loops program`: the clip's video is repeated with -c copy and the
# exactly-periodic WAV is re-encoded continuously, so repeats inside a program have no audio gap.
#
#   bash build_programs.sh CLIPS_DIR OUT_DIR [MINUTES]
#   CLIPS_DIR contains one folder per scene with clip.mp4, clip.wav and meta.json
set -euo pipefail

CLIPS="${1:?usage: build_programs.sh CLIPS_DIR OUT_DIR [MINUTES]}"
OUT="${2:?usage: build_programs.sh CLIPS_DIR OUT_DIR [MINUTES]}"
MINUTES="${3:-30}"
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (sudo apt install ffmpeg)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 1; }
mkdir -p "$OUT"

built=0
for meta in "$CLIPS"/*/meta.json; do
  [[ -f "$meta" ]] || continue
  dir="$(dirname "$meta")"
  read -r scene period clip audio abr < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m["scene"], int(m["period"]), m["clip"], m["audio"], m.get("settings", {}).get("audio_bitrate", "128k"))
' "$meta")
  repeats=$(( (MINUTES * 60 + period - 1) / period ))
  (( repeats < 1 )) && repeats=1
  total=$(( repeats * period ))
  tmp="$OUT/$scene.tmp.mp4"
  echo "Building $scene: ${repeats} x ${period}s = $(( total / 60 )) min"
  ffmpeg -hide_banner -loglevel error -y \
    -stream_loop $(( repeats - 1 )) -i "$dir/$clip" \
    -stream_loop $(( repeats - 1 )) -i "$dir/$audio" \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a "$abr" -ar 48000 -ac 2 \
    -t "$total" -movflags +faststart "$tmp"
  mv "$tmp" "$OUT/$scene.mp4"
  chmod 644 "$OUT/$scene.mp4"
  built=$(( built + 1 ))
done

if (( built == 0 )); then
  echo "No clips found under $CLIPS (expected */meta.json)" >&2
  exit 1
fi
echo "Done: $built program(s) in $OUT"

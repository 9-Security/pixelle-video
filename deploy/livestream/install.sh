#!/usr/bin/env bash
# Install the loop streamer as a systemd service (idempotent). Does not start it unless --start is given.
#   sudo bash deploy/livestream/install.sh [--library DIR] [--start]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBRARY="/srv/loop-streamer/programs"
START=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --library) LIBRARY="$2"; shift 2 ;;
    --start) START=1; shift ;;
    -h|--help) sed -n '2,3p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then echo "Run as root (sudo)." >&2; exit 1; fi
for tool in python3 ffmpeg ffprobe systemctl; do
  command -v "$tool" >/dev/null || { echo "Missing dependency: $tool" >&2; exit 1; }
done

install -d -m 755 /opt/loop-streamer "$LIBRARY"
install -m 755 "$HERE/streamer.py" /opt/loop-streamer/streamer.py
install -d -m 700 /etc/loop-streamer
if [[ ! -f /etc/loop-streamer/stream.env ]]; then
  install -m 600 "$HERE/stream.env.example" /etc/loop-streamer/stream.env
  sed -i "s#^STREAM_LIBRARY=.*#STREAM_LIBRARY=$LIBRARY#" /etc/loop-streamer/stream.env
  echo "Created /etc/loop-streamer/stream.env - add your stream key there (see README)."
else
  chmod 600 /etc/loop-streamer/stream.env
  echo "Keeping existing /etc/loop-streamer/stream.env"
fi
install -m 644 "$HERE/loop-streamer.service" /etc/systemd/system/loop-streamer.service
systemctl daemon-reload
systemctl enable loop-streamer.service >/dev/null

if [[ $START -eq 1 ]]; then
  systemctl restart loop-streamer.service
  echo "Started. Follow logs with: journalctl -u loop-streamer -f"
else
  echo "Installed. When the key and programs are in place: sudo systemctl start loop-streamer"
fi

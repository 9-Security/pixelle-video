#!/usr/bin/env python3
# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
24/7 loop streamer for YouTube Live (or any RTMP/RTMPS ingest).

Stdlib only + ffmpeg/ffprobe, so it runs on any small Linux host. It streams
pre-encoded "program" MP4 files (see `python -m pixelle_video.loops program`)
with `-c copy`: no re-encoding, so CPU use stays tiny.

- The stream key is read from the YOUTUBE_STREAM_KEY environment variable and
  is never written to logs, the status file or the command line shown to users.
- Programs whose codec settings differ from the first one are skipped
  (copy-concatenation needs identical parameters).
- ffmpeg is restarted with exponential backoff when it exits; the library is
  rescanned and reshuffled on each start.

Examples:
    YOUTUBE_STREAM_KEY=... ./streamer.py --library /srv/loops/programs
    ./streamer.py --library programs --output /tmp/test.flv --duration 60   # local test, no key
    ./streamer.py --library programs --check                                # validate only
"""

import argparse
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_INGEST = "rtmps://a.rtmps.youtube.com:443/live2"
KEY_ENV = "YOUTUBE_STREAM_KEY"
HEALTHY_AFTER = 300  # seconds of streaming before the backoff resets
BACKOFF = [5, 10, 20, 40, 60]
COMPAT_FIELDS = ("v_codec", "width", "height", "fps", "pix_fmt", "a_codec", "sample_rate", "channels")

log = logging.getLogger("loop-streamer")


class Redactor:
    """Replaces secrets in any text before it reaches a log or file."""

    def __init__(self, secrets: List[str]):
        self.secrets = [s for s in secrets if s and len(s) >= 4]

    def __call__(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "***")
        return text


def probe(path: Path, ffprobe: str) -> Optional[Dict]:
    """Codec parameters of a program file, or None if it cannot be read."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels"
             ":format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None
    info: Dict = {"duration": float(data.get("format", {}).get("duration", 0) or 0)}
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and "v_codec" not in info:
            info.update(v_codec=st.get("codec_name"), width=st.get("width"), height=st.get("height"),
                        fps=st.get("r_frame_rate"), pix_fmt=st.get("pix_fmt"))
        elif st.get("codec_type") == "audio" and "a_codec" not in info:
            info.update(a_codec=st.get("codec_name"), sample_rate=st.get("sample_rate"), channels=st.get("channels"))
    if "v_codec" not in info or "a_codec" not in info:
        return None
    return info


def scan_library(library: Path, ffprobe: str) -> List[Path]:
    """Compatible, non-temporary MP4 programs (all matching the first valid file's codec settings)."""
    files = sorted(p for p in library.glob("*.mp4") if ".tmp" not in p.name and p.is_file())
    accepted, reference = [], None
    for path in files:
        info = probe(path, ffprobe)
        if info is None or info["duration"] < 1:
            log.warning("Skipping unreadable program: %s", path.name)
            continue
        key = {f: info.get(f) for f in COMPAT_FIELDS}
        if reference is None:
            reference = key
            log.info("Stream format: %sx%s %s fps %s / %s %s Hz",
                     key["width"], key["height"], key["fps"], key["v_codec"], key["a_codec"], key["sample_rate"])
        elif key != reference:
            log.warning("Skipping %s: codec settings differ from the first program", path.name)
            continue
        accepted.append(path)
    return accepted


def write_playlist(programs: List[Path], path: Path, shuffle: bool, rng: random.Random) -> None:
    order = list(programs)
    if shuffle:
        rng.shuffle(order)
    lines = ["ffconcat version 1.0"]
    for p in order:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_command(ffmpeg: str, playlist: Path, target: str, is_network: bool, duration: Optional[float]) -> List[str]:
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-nostats",
           "-re", "-stream_loop", "-1", "-f", "concat", "-safe", "0", "-i", str(playlist),
           "-map", "0:v:0", "-map", "0:a:0", "-c", "copy"]
    if duration:
        cmd += ["-t", str(duration)]
    if is_network:
        cmd += ["-f", "flv", "-flvflags", "no_duration_filesize", target]
    else:
        cmd += ["-y", target]
    return cmd


class Streamer:
    def __init__(self, args: argparse.Namespace, key: str):
        self.args = args
        self.key = key
        self.redact = Redactor([key])
        self.stop_event = threading.Event()
        self.proc: Optional[subprocess.Popen] = None
        self.rng = random.Random()
        self.state_dir = Path(args.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.playlist = self.state_dir / "playlist.ffconcat"
        self.status_path = self.state_dir / "status.json"
        self.status = {"state": "starting", "restarts": 0, "started_at": None, "last_exit": None,
                       "last_error": None, "programs": 0}
        if args.output:
            self.target, self.is_network = args.output, "://" in args.output
        else:
            self.target, self.is_network = f"{args.ingest.rstrip('/')}/{key}", True

    def _write_status(self, **changes) -> None:
        self.status.update(changes)
        self.status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        safe = json.loads(self.redact(json.dumps(self.status)))
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(safe, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def stop(self, *_):
        log.info("Stop requested")
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        last = None
        for raw in proc.stderr:
            line = self.redact(raw.decode(errors="replace").rstrip())
            if line:
                last = line
                log.info("ffmpeg: %s", line)
        if last:
            self.status["last_error"] = last

    def run_once(self) -> int:
        programs = scan_library(Path(self.args.library), self.args.ffprobe)
        if not programs:
            log.error("No compatible programs in %s", self.args.library)
            self._write_status(state="no_programs", programs=0)
            return -1
        write_playlist(programs, self.playlist, not self.args.no_shuffle, self.rng)
        cmd = build_command(self.args.ffmpeg, self.playlist, self.target, self.is_network, self.args.duration)
        log.info("Streaming %d program(s) to %s", len(programs), self.redact(self.target))
        self._write_status(state="streaming", programs=len(programs),
                           started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        pump = threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True)
        pump.start()
        deadline = time.monotonic() + self.args.rotate_hours * 3600 if self.args.rotate_hours else None
        while self.proc.poll() is None:
            if self.stop_event.wait(1.0):
                break
            if deadline and time.monotonic() > deadline:
                log.info("Rotating: restarting to pick up new programs")
                self.proc.terminate()
                break
        try:
            code = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        pump.join(timeout=5)
        return code

    def run(self) -> int:
        attempt = 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            code = self.run_once()
            ran = time.monotonic() - started
            if self.stop_event.is_set():
                break
            if self.args.duration and code == 0:
                log.info("Finished test run")
                self._write_status(state="finished", last_exit=code)
                return 0
            attempt = 0 if ran >= HEALTHY_AFTER else attempt + 1
            delay = BACKOFF[min(attempt, len(BACKOFF) - 1)] if attempt else 3
            self.status["restarts"] += 1
            self._write_status(state="restarting", last_exit=code)
            log.warning("ffmpeg exited with %s after %.0fs; restarting in %ds", code, ran, delay)
            if self.stop_event.wait(delay):
                break
        self._write_status(state="stopped")
        return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream looping programs to YouTube Live 24/7")
    p.add_argument("--library", default=os.environ.get("STREAM_LIBRARY"),
                   help="Directory of program .mp4 files (env STREAM_LIBRARY)")
    p.add_argument("--ingest", default=os.environ.get("STREAM_INGEST_URL", DEFAULT_INGEST),
                   help=f"RTMP(S) ingest base URL (default {DEFAULT_INGEST})")
    p.add_argument("--output", help="Test target instead of YouTube: a file path or an rtmp:// URL (no key needed)")
    p.add_argument("--duration", type=float, help="Stop after this many seconds (tests)")
    p.add_argument("--rotate-hours", type=float, default=0.0,
                   help="Restart every N hours to pick up new programs (brief reconnect); 0 = never")
    p.add_argument("--no-shuffle", action="store_true", help="Keep programs in name order")
    p.add_argument("--state-dir", default=os.environ.get("STREAM_STATE_DIR", "/tmp/loop-streamer"),
                   help="Where the playlist and status.json are written")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    p.add_argument("--ffprobe", default=shutil.which("ffprobe") or "ffprobe")
    p.add_argument("--check", action="store_true", help="Validate the library and configuration, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.library:
        print("error: --library (or STREAM_LIBRARY) is required", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    for tool in (args.ffmpeg, args.ffprobe):
        if not shutil.which(tool) and not Path(tool).is_file():
            log.error("%s not found", tool)
            return 2
    if not Path(args.library).is_dir():
        log.error("Library directory not found: %s", args.library)
        return 2
    key = os.environ.get(KEY_ENV, "").strip()
    if not args.output and not key:
        log.error("%s is not set (or use --output for a local test)", KEY_ENV)
        return 2
    if args.check:
        programs = scan_library(Path(args.library), args.ffprobe)
        total = sum((probe(p, args.ffprobe) or {}).get("duration", 0) for p in programs)
        log.info("%d compatible program(s), %.1f min in total; target %s", len(programs), total / 60,
                 Redactor([key])(args.output or f"{args.ingest.rstrip('/')}/{key}"))
        return 0 if programs else 1
    streamer = Streamer(args, key)
    signal.signal(signal.SIGTERM, streamer.stop)
    signal.signal(signal.SIGINT, streamer.stop)
    return streamer.run()


if __name__ == "__main__":
    sys.exit(main())

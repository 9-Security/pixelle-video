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
Render loop scenes to stream-ready files.

clip:    one period, H.264 + AAC, keyframe every 2 s (so the period boundary is a keyframe)
         plus the exactly-periodic WAV and meta.json with a seam report
program: the clip repeated for N minutes with the WAV re-encoded continuously
         (no AAC priming gap at each repeat); every program uses the same codec
         settings, so a streamer can concatenate them with `-c copy`
"""

import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from pixelle_video.loops.audio import SAMPLE_RATE, render_audio, write_wav
from pixelle_video.loops.base import LoopScene, get_scene

GOP_SECONDS = 2
SEAM_WARN_RATIO = 3.0


@dataclass
class EncodeSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    crf: int = 23
    maxrate: str = "3500k"  # cap keeps the stream within YouTube's 1080p30 range and egress predictable
    preset: str = "medium"
    audio_bitrate: str = "128k"

    def validate(self) -> None:
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even (yuv420p)")
        if not 10 <= self.fps <= 60:
            raise ValueError("fps must be between 10 and 60")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")


@dataclass
class SeamReport:
    seam_diff: float  # mean abs pixel change across the loop point
    typical_diff: float  # median change between consecutive frames inside the loop
    # ratio compares seam_diff with the largest of typical_diff and the steps next to the loop point
    wrap_diff: float  # frame(0) vs frame(period): 0 for a truly periodic scene
    ratio: float
    ok: bool


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH")
    return path


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def check_seam(scene: LoopScene, fps: int, samples: int = 6) -> SeamReport:
    """Compare the change across the loop point with ordinary frame-to-frame change."""
    dt = 1.0 / fps
    T = scene.period
    first = scene.frame(0.0)
    seam = _diff(scene.frame(T - dt), first)
    wrap = _diff(scene.frame(float(T)), first)
    inner = [_diff(scene.frame(t), scene.frame(t + dt)) for t in np.linspace(T * 0.1, T * 0.9, samples)]
    typical = float(np.median(inner))
    # Steps right before and after the loop point: a fade or blink legitimately changes a lot there
    before = _diff(scene.frame(T - 2 * dt), scene.frame(T - dt))
    after = _diff(first, scene.frame(dt))
    ratio = seam / max(typical, before, after, 0.05)
    return SeamReport(seam, typical, wrap, ratio, ok=wrap < 0.5 and ratio <= SEAM_WARN_RATIO)


def render_clip(scene_name: str, out_dir: str, settings: Optional[EncodeSettings] = None, seed: int = 7,
                dither: Optional[float] = None) -> dict:
    """Render one period of a scene; returns the meta dict (also written to meta.json)."""
    settings = settings or EncodeSettings()
    settings.validate()
    ffmpeg = _require_ffmpeg()
    scene_cls = get_scene(scene_name)
    scene = scene_cls(settings.width, settings.height, seed=seed)
    if dither is not None:  # grain against banding; 0 disables it (lower bitrate, more visible banding)
        if dither < 0:
            raise ValueError("dither must be >= 0")
        scene.dither_strength = float(dither)
        scene._dither = scene._make_dither()
    T = scene.period
    if T % GOP_SECONDS:
        raise ValueError(f"Scene period {T}s must be a multiple of {GOP_SECONDS}s (keyframe at the loop point)")
    out = Path(out_dir) / scene_name
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / "clip.wav"
    clip_path = out / "clip.mp4"
    tmp_path = out / "clip.tmp.mp4"

    started = time.monotonic()
    seam = check_seam(scene, settings.fps)
    if not seam.ok:
        logger.warning(f"{scene_name}: loop seam looks visible (ratio {seam.ratio:.1f}, wrap {seam.wrap_diff:.2f})")

    audio = render_audio(scene.sound_design(), T, SAMPLE_RATE, seed=seed)
    write_wav(str(wav_path), audio)

    n_frames = T * settings.fps
    gop = settings.fps * GOP_SECONDS
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{settings.width}x{settings.height}",
        "-r", str(settings.fps), "-i", "-",
        "-i", str(wav_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-maxrate", settings.maxrate, "-bufsize", _double_rate(settings.maxrate),
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-c:a", "aac", "-b:a", settings.audio_bitrate, "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-frames:v", str(n_frames), "-movflags", "+faststart", str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(n_frames):
            frame = scene.frame(i / settings.fps)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            if i and i % (settings.fps * 10) == 0:
                logger.info(f"{scene_name}: {i}/{n_frames} frames")
        proc.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {scene_name} (exit {code}): {stderr[-600:]}")
    tmp_path.replace(clip_path)

    meta = {
        "scene": scene_name,
        "title_zh": scene.info.title_zh,
        "title_en": scene.info.title_en,
        "kind": scene.info.kind,
        "period": T,
        "seed": seed,
        "dither": scene.dither_strength,
        "settings": asdict(settings),
        "seam": asdict(seam),
        "clip": clip_path.name,
        "audio": wav_path.name,
        "render_seconds": round(time.monotonic() - started, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"{scene_name}: clip ready ({T}s, seam ratio {seam.ratio:.2f}) in {meta['render_seconds']}s")
    return meta


def build_program(clip_dir: str, out_path: str, minutes: float = 30.0) -> str:
    """Repeat a rendered clip for about `minutes` (whole periods) into one stream-ready MP4."""
    ffmpeg = _require_ffmpeg()
    clip_dir_p = Path(clip_dir)
    meta = json.loads((clip_dir_p / "meta.json").read_text(encoding="utf-8"))
    period = int(meta["period"])
    repeats = max(1, math.ceil(minutes * 60 / period))
    total = repeats * period
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    s = meta["settings"]
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-stream_loop", str(repeats - 1), "-i", str(clip_dir_p / meta["clip"]),
        "-stream_loop", str(repeats - 1), "-i", str(clip_dir_p / meta["audio"]),
        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", s.get("audio_bitrate", "128k"), "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-t", str(total), "-movflags", "+faststart", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg program build failed (exit {proc.returncode}): {proc.stderr[-600:]}")
    tmp.replace(out)
    logger.success(f"Program {out.name}: {meta['scene']} x{repeats} = {total / 60:.1f} min")
    return str(out)


def _double_rate(rate: str) -> str:
    rate = rate.strip().lower()
    for suffix, mult in (("k", 1_000), ("m", 1_000_000)):
        if rate.endswith(suffix):
            return f"{int(float(rate[:-1]) * mult * 2)}"
    return str(int(float(rate) * 2))

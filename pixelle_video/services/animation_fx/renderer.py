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
2.5D parallax renderer.

Turns one still image into a short animated clip:
  - background and character layers move at different speeds (parallax)
  - the character "breathes" (subtle squash/stretch + bob around its feet)
  - ambient particles (fireflies, snow, rain, ...) drift over the scene
  - vignette, gentle light pulse and fade in/out

Frames are composed with NumPy/OpenCV on CPU and streamed as raw RGB to an
ffmpeg subprocess, so memory stays flat regardless of clip length.
"""

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageOps

from pixelle_video.models.animation import CameraMotion
from pixelle_video.services.animation_fx.layers import SceneLayers, build_layers
from pixelle_video.services.animation_fx.particles import ParticleField
from pixelle_video.services.animation_fx.segmentation import (
    ForegroundSegmenter,
    SegmentationUnavailable,
)
from pixelle_video.utils.motion_util import clamp_intensity, even, frame_count

# veryfast is ~3-4x faster than medium at a similar CRF (slightly larger files)
X264_PRESET = os.environ.get("PIXELLE_X264_PRESET", "veryfast")


@dataclass
class ParallaxOptions:
    motion: str = "zoom_in"
    intensity: float = 0.12        # camera zoom/pan amount (0-0.5)
    parallax: float = 0.6          # extra foreground movement relative to background
    ambient: str = "none"          # particle effect name
    breathing: bool = True         # character idle motion
    fade: float = 0.3              # fade-in/out seconds (0 disables)
    vignette: float = 0.22         # edge darkening 0..1
    fps: int = 30
    seed: int = 0


def ease_in_out(p: float) -> float:
    """Smoothstep easing so moves start and stop gently."""
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def _affine(zoom: float, cx: float, cy: float, dx: float, dy: float) -> np.ndarray:
    """3x3 matrix: scale by `zoom` around (cx, cy), then translate by (dx, dy)."""
    return np.array([
        [zoom, 0.0, (1.0 - zoom) * cx + dx],
        [0.0, zoom, (1.0 - zoom) * cy + dy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _breath(anchor_x: float, anchor_y: float, t: float, amount: float, height: float) -> np.ndarray:
    """Squash/stretch around the feet plus a small bob."""
    phase = math.sin(2.0 * math.pi * t / 3.2)
    sy = 1.0 + amount * phase
    sx = 1.0 - amount * 0.35 * phase
    bob = -0.0035 * height * (0.5 + 0.5 * math.sin(2.0 * math.pi * t / 3.2 + 0.6))
    return np.array([
        [sx, 0.0, (1.0 - sx) * anchor_x],
        [0.0, sy, (1.0 - sy) * anchor_y + bob],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


class CameraPath:
    """Zoom/translation of the background plane over normalized time."""

    def __init__(self, motion: str, intensity: float, width: int, height: int):
        self.motion = CameraMotion.parse(motion)
        self.k = clamp_intensity(intensity)
        self.w, self.h = width, height

    def at(self, p: float) -> tuple[float, float, float]:
        """Return (zoom, dx, dy) for progress p in [0, 1]."""
        e = ease_in_out(p)
        k = self.k
        if self.motion == CameraMotion.ZOOM_IN:
            return 1.0 + k * e, 0.0, 0.0
        if self.motion == CameraMotion.ZOOM_OUT:
            return 1.0 + k * (1.0 - e), 0.0, 0.0
        z = 1.0 + max(k, 0.04)
        span_x = (z - 1.0) * self.w / 2.0
        span_y = (z - 1.0) * self.h / 2.0
        if self.motion == CameraMotion.PAN_LEFT:      # view moves left -> content moves right
            return z, span_x * (2 * e - 1.0), 0.0
        if self.motion == CameraMotion.PAN_RIGHT:
            return z, -span_x * (2 * e - 1.0), 0.0
        if self.motion == CameraMotion.PAN_UP:
            return z, 0.0, span_y * (2 * e - 1.0)
        if self.motion == CameraMotion.PAN_DOWN:
            return z, 0.0, -span_y * (2 * e - 1.0)
        # static: a barely perceptible drift keeps parallax alive
        return 1.04, 0.012 * self.w * math.sin(math.pi * e), 0.0


class ParallaxRenderer:
    """Render a still image as a 2.5D animated clip."""

    def __init__(self, segmenter: Optional[ForegroundSegmenter] = None, use_segmentation: bool = True):
        self.segmenter = segmenter if segmenter is not None else ForegroundSegmenter()
        self.use_segmentation = use_segmentation

    # ------------------------------------------------------------ layers

    @staticmethod
    def _load(image_path: str, width: int, height: int) -> Image.Image:
        return ImageOps.fit(Image.open(image_path).convert("RGB"), (width, height), Image.LANCZOS)

    def segment(self, image_path: str, width: int, height: int) -> Optional[np.ndarray]:
        """Foreground mask at width x height, or None when segmentation is unavailable/disabled."""
        if not self.use_segmentation:
            return None
        try:
            return self.segmenter.predict(self._load(image_path, width, height))
        except SegmentationUnavailable as e:
            logger.warning(f"Segmentation unavailable, using single-layer motion: {e}")
            return None

    def prepare(self, image_path: str, width: int, height: int,
                mask: Optional[np.ndarray] = None, segment: bool = True) -> SceneLayers:
        """
        Build layers. A precomputed `mask` is used as-is; otherwise the image is
        segmented when `segment` is True (False = single layer, e.g. camera-only motion).
        """
        image = self._load(image_path, width, height)
        if mask is None and segment and self.use_segmentation:
            try:
                mask = self.segmenter.predict(image)
            except SegmentationUnavailable as e:
                logger.warning(f"Segmentation unavailable, using single-layer motion: {e}")
        layers = build_layers(image, mask)
        logger.debug(f"Layers ready: foreground={layers.has_foreground} area={layers.fg_area:.1%}")
        return layers

    # ------------------------------------------------------------ frames

    def _vignette(self, width: int, height: int, strength: float) -> Optional[np.ndarray]:
        if strength <= 0:
            return None
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xs - width / 2.0) / (width / 2.0)
        ny = (ys - height / 2.0) / (height / 2.0)
        r = np.sqrt(nx * nx * 0.9 + ny * ny * 0.6)
        return (1.0 - strength * np.clip(r - 0.35, 0.0, 1.0) ** 1.6)[..., None].astype(np.float32)

    def compose_frame(self, layers: SceneLayers, camera: CameraPath, particles: ParticleField,
                      vignette: Optional[np.ndarray], opts: ParallaxOptions,
                      t: float, duration: float) -> np.ndarray:
        """Compose one float32 RGB frame in [0, 1]."""
        w, h = layers.size
        p = t / duration if duration > 0 else 0.0
        zoom, dx, dy = camera.at(p)
        cx, cy = w / 2.0, h / 2.0

        bg_m = _affine(zoom, cx, cy, dx, dy)
        frame = cv2.warpAffine(layers.background, bg_m[:2], (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

        if layers.has_foreground:
            gain = 1.0 + opts.parallax
            fg_zoom = 1.0 + (zoom - 1.0) * gain
            fg_m = _affine(fg_zoom, cx, cy, dx * gain, dy * gain)
            if opts.breathing:
                fg_m = fg_m @ _breath(layers.anchor[0], layers.anchor[1], t, 0.012, h)
            fg = cv2.warpAffine(layers.foreground, fg_m[:2], (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            alpha = cv2.warpAffine(layers.alpha, fg_m[:2], (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            frame = frame * (1.0 - alpha[..., None]) + fg

        frame = particles.apply(frame, t)

        light = 1.0
        if particles.effect in ("fireflies", "embers"):
            light = 1.0 + 0.025 * math.sin(2.0 * math.pi * t / 2.3)
        if vignette is not None:
            frame = frame * vignette
        if opts.fade > 0:
            light *= min(1.0, t / opts.fade, max(0.0, duration - t) / opts.fade)
        if light != 1.0:
            frame = frame * light
        return np.clip(frame, 0.0, 1.0)

    # ------------------------------------------------------------ output

    def render(self, image_path: str, output: str, duration: float, width: int, height: int,
               options: Optional[ParallaxOptions] = None, audio: Optional[str] = None,
               overlay: Optional[str] = None, mask: Optional[np.ndarray] = None,
               segment: bool = True) -> str:
        """
        Render the clip to `output` (.mp4).

        Frames are composed at width x height. With `overlay` (a transparent PNG such
        as a rendered subtitle template), the frames are fit into the overlay's canvas
        (contain + pad), the overlay is drawn on top and `audio` is muxed, all in ONE
        ffmpeg encode, which avoids separate overlay / audio re-encodes per scene.
        """
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found")
        if duration <= 0:
            raise ValueError("duration must be positive")
        opts = options or ParallaxOptions()
        w, h = even(width), even(height)
        layers = self.prepare(image_path, w, h, mask=mask, segment=segment)
        camera = CameraPath(opts.motion, opts.intensity, w, h)
        particles = ParticleField(opts.ambient, w, h, seed=opts.seed)
        vignette = self._vignette(w, h, opts.vignette)
        n_frames = frame_count(duration, opts.fps)

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(opts.fps), "-i", "-"]
        graph, video_map = [], "0:v"
        if overlay:
            with Image.open(overlay) as ov:
                ow, oh = even(ov.size[0]), even(ov.size[1])
            inputs += ["-i", overlay]
            graph = ["-filter_complex",
                     f"[0:v]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=bicubic,"
                     f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:color=black[bg];"
                     f"[bg][1:v]overlay=0:0:format=auto,format=yuv420p[v]"]
            video_map = "[v]"
        maps = ["-map", video_map]
        if audio:
            maps += ["-map", f"{inputs.count('-i')}:a", "-c:a", "aac", "-b:a", "192k"]
            inputs += ["-i", audio]
        cmd = (["ffmpeg", "-y", "-loglevel", "error"] + inputs + graph + maps
               + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", X264_PRESET, "-crf", "20",
                  "-r", str(opts.fps), "-t", f"{duration:.3f}", output])

        logger.info(
            f"Parallax clip: {w}x{h}@{opts.fps} {duration:.2f}s motion={opts.motion} "
            f"ambient={particles.effect} layers={'2' if layers.has_foreground else '1'}"
        )
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(n_frames):
                t = i / float(opts.fps)
                frame = self.compose_frame(layers, camera, particles, vignette, opts, t, duration)
                process.stdin.write((frame * 255.0 + 0.5).astype(np.uint8).tobytes())
            process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace")
            code = process.wait()
        except BrokenPipeError:
            stderr = process.stderr.read().decode(errors="replace")
            code = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if code != 0:
            raise RuntimeError(f"ffmpeg failed to encode parallax clip (exit {code}): {stderr[-500:]}")
        logger.success(f"Parallax clip created: {output}")
        return output


def camera_only_options(motion: str, intensity: float, fps: int, seed: int = 0) -> ParallaxOptions:
    """Plain zoom/pan (camera mode): no parallax, particles, breathing or vignette."""
    return ParallaxOptions(motion=motion, intensity=intensity, parallax=0.0, ambient="none",
                           breathing=False, vignette=0.0, fps=fps, seed=seed)

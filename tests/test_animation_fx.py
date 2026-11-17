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

"""Tests for the 2.5D animation effects engine (no network, no real model)."""

import hashlib
import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixelle_video.services.animation_fx import segmentation as seg_mod
from pixelle_video.services.animation_fx.layers import build_layers
from pixelle_video.services.animation_fx.particles import (
    AMBIENT_EFFECTS,
    ParticleField,
    normalize_effect,
)
from pixelle_video.services.animation_fx.renderer import (
    CameraPath,
    ParallaxOptions,
    ParallaxRenderer,
    ease_in_out,
)
from pixelle_video.services.animation_fx.segmentation import (
    ForegroundSegmenter,
    SegmentationModel,
    SegmentationUnavailable,
    ensure_model,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
W, H = 96, 160


def _scene() -> Image.Image:
    """Blue gradient background with a red 'character' disc near the bottom."""
    grad = np.linspace(0.2, 0.8, H, dtype=np.float32)[:, None, None] * np.array([0.2, 0.4, 1.0])
    img = Image.fromarray((np.broadcast_to(grad, (H, W, 3)) * 255).astype(np.uint8))
    ImageDraw.Draw(img).ellipse((28, 70, 68, 150), fill=(230, 30, 30))
    return img


def _disc_mask() -> np.ndarray:
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).ellipse((28, 70, 68, 150), fill=255)
    return np.asarray(m, dtype=np.float32) / 255.0


class FakeSegmenter:
    def __init__(self, fail=False):
        self.fail = fail

    def predict(self, image):
        if self.fail:
            raise SegmentationUnavailable("offline")
        assert image.size == (W, H)
        return _disc_mask()


# ------------------------------------------------------------------ particles

def test_normalize_effect():
    assert normalize_effect("Firefly") == "fireflies"
    assert normalize_effect("sakura") == "petals"
    assert normalize_effect("lava") == "none"
    assert normalize_effect(None) == "none"


@pytest.mark.parametrize("effect", [e for e in AMBIENT_EFFECTS if e != "none"])
def test_particles_are_deterministic_and_visible(effect):
    base = np.full((H * 4, W * 4, 3), 0.3, dtype=np.float32)
    a = ParticleField(effect, W * 4, H * 4, seed=3)
    b = ParticleField(effect, W * 4, H * 4, seed=3)
    out_a, out_b = a.apply(base, 1.0), b.apply(base, 1.0)
    assert np.array_equal(out_a, out_b)
    assert not np.array_equal(out_a, base)
    assert out_a.min() >= 0.0 and out_a.max() <= 1.0


def test_particles_none_is_noop():
    base = np.zeros((H, W, 3), dtype=np.float32)
    field = ParticleField("none", W, H)
    assert not field.enabled
    assert field.apply(base, 0.5) is base


# ------------------------------------------------------------------ layers

def test_build_layers_splits_and_inpaints():
    layers = build_layers(_scene(), _disc_mask(), dof_sigma=0)
    assert layers.has_foreground
    assert 0.1 < layers.fg_area < 0.3
    assert abs(layers.anchor[0] - 48) <= 2 and layers.anchor[1] >= 145
    # The red disc is removed from the background
    center = layers.background[110, 48]
    assert center[0] < 0.6 and center[2] > center[0]
    # Foreground is premultiplied: zero outside the disc
    assert layers.foreground[5, 5].sum() == 0


def test_build_layers_single_layer_cases():
    assert not build_layers(_scene(), None).has_foreground
    assert not build_layers(_scene(), np.ones((H, W), np.float32)).has_foreground  # too large
    assert not build_layers(_scene(), np.zeros((H, W), np.float32)).has_foreground  # empty
    with pytest.raises(ValueError):
        build_layers(_scene(), np.zeros((10, 10), np.float32))


# ------------------------------------------------------------------ camera

def test_camera_paths():
    assert ease_in_out(0) == 0 and ease_in_out(1) == 1 and ease_in_out(0.5) == 0.5
    zoom_in = CameraPath("zoom_in", 0.2, W, H)
    assert zoom_in.at(0)[0] == pytest.approx(1.0) and zoom_in.at(1)[0] == pytest.approx(1.2)
    pan = CameraPath("pan_left", 0.2, W, H)
    z, dx0, _ = pan.at(0)
    _, dx1, _ = pan.at(1)
    assert dx0 == pytest.approx(-dx1) and abs(dx0) <= (z - 1) * W / 2 + 1e-6


# ------------------------------------------------------------------ segmentation

def test_preprocess_letterbox():
    segmenter = ForegroundSegmenter(SegmentationModel("m.onnx", "http://x", "0", input_size=64))
    tensor, (ox, oy, nw, nh) = segmenter.preprocess(_scene())
    assert tensor.shape == (1, 3, 64, 64)
    assert nh == 64 and nw == round(W * 64 / H) and oy == 0 and ox == (64 - nw) // 2
    assert tensor.min() >= 0 and tensor.max() <= 1


def test_predict_crops_letterbox(monkeypatch):
    model = SegmentationModel("m.onnx", "http://x", "0", input_size=64)
    segmenter = ForegroundSegmenter(model)
    _, (ox, oy, nw, nh) = segmenter.preprocess(_scene())

    class Session:
        def get_inputs(self):
            return [type("I", (), {"name": "img"})()]

        def run(self, _, feeds):
            out = np.zeros((1, 1, 64, 64), np.float32)
            out[..., oy:oy + nh, ox:ox + nw] = 1.0  # content area only; padding stays 0
            return [out]

    monkeypatch.setattr(segmenter, "_session", lambda: Session())
    mask = segmenter.predict(_scene())
    assert mask.shape == (H, W)
    assert mask.mean() > 0.97  # padding must be cropped away, not stretched in


def test_ensure_model_verifies_checksum(monkeypatch, tmp_path):
    payload = b"fake-model-bytes"

    @contextmanager
    def fake_stream(method, url, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass

            def iter_bytes(self, _):
                yield payload
        yield Resp()

    monkeypatch.setattr(seg_mod.httpx, "stream", fake_stream)
    good = SegmentationModel("ok.onnx", "http://x", hashlib.sha256(payload).hexdigest())
    path = ensure_model(good, tmp_path)
    assert path.read_bytes() == payload
    assert ensure_model(good, tmp_path) == path  # cached, verified

    bad = SegmentationModel("bad.onnx", "http://x", "0" * 64)
    with pytest.raises(SegmentationUnavailable, match="Checksum"):
        ensure_model(bad, tmp_path)
    assert not (tmp_path / "bad.onnx").exists()
    assert not list(tmp_path.glob("*.part"))


# ------------------------------------------------------------------ renderer

def _grab(video: str, index: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"select=eq(n\\,{index})",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float32)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("fail", [False, True])
def test_render_clip(tmp_path, fail):
    src = tmp_path / "scene.png"
    _scene().save(src)
    out = tmp_path / "clip.mp4"
    renderer = ParallaxRenderer(FakeSegmenter(fail=fail))
    renderer.render(str(src), str(out), 1.0, W, H,
                    ParallaxOptions(motion="zoom_in", intensity=0.3, ambient="snow", fps=20))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries", "stream=nb_read_frames,width,height",
         "-of", "csv=p=0", str(out)], check=True, capture_output=True, text=True).stdout.strip()
    width, height, frames = map(int, probe.split(","))
    assert (width, height, frames) == (W, H, 20)
    assert np.abs(_grab(str(out), 6) - _grab(str(out), 13)).mean() > 1.0  # it moves
    assert _grab(str(out), 0).mean() < _grab(str(out), 10).mean()        # fade-in


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_single_pass_overlay_and_audio(tmp_path):
    """One encode: frames fit into the overlay canvas, overlay on top, audio muxed."""
    src = tmp_path / "scene.png"
    _scene().save(src)
    overlay = tmp_path / "overlay.png"
    canvas = Image.new("RGBA", (W * 2, H * 3), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rectangle((0, 0, W * 2, 20), fill=(255, 255, 0, 255))  # "subtitle bar"
    canvas.save(overlay)
    audio = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=d=1", str(audio)], check=True)

    out = tmp_path / "segment.mp4"
    ParallaxRenderer(FakeSegmenter()).render(str(src), str(out), 1.0, W, H,
                                             ParallaxOptions(fps=10), audio=str(audio), overlay=str(overlay))
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                            "-of", "csv=p=0", str(out)], check=True, capture_output=True, text=True).stdout
    assert f"video,{W * 2},{H * 3}" in probe and "audio" in probe
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    frame = np.frombuffer(raw, np.uint8).reshape(H * 3, W * 2, 3).astype(int)
    assert frame[5, 10, 0] > 200 and frame[5, 10, 1] > 200 and frame[5, 10, 2] < 80   # overlay drawn

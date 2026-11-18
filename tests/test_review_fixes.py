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

"""Regression tests for the code-review fixes."""

import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixelle_video.config.loader import restore_env_refs
from pixelle_video.pipelines.animation import SceneMediaError
from pixelle_video.services.animation_fx.renderer import ParallaxRenderer
from pixelle_video.services.animation_fx.segmentation import (
    ForegroundSegmenter,
    SegmentationUnavailable,
)
from pixelle_video.utils.content_policy import is_content_policy_rejection

HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------- 1. prefetch concurrency

async def _pipeline():
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    return core.pipelines["animation"]


@pytest.mark.parametrize("media,tts_mode,tts_workflow,expected", [
    ("api/cloudflare/flux-1-schnell", "local", None, 3),
    ("selfhost/image_flux.json", "local", None, 1),
    (None, "local", None, 1),                                   # default ComfyUI workflow
    ("api/cloudflare/flux-1-schnell", "comfyui", "selfhost/tts_edge.json", 1),
    ("runninghub/image_flux.json", "local", None, "limit"),
    ("api/cloudflare/flux-1-schnell", "comfyui", "runninghub/tts_edge.json", "limit"),
])
async def test_prefetch_limit_respects_backend(monkeypatch, media, tts_mode, tts_workflow, expected):
    from pixelle_video.config import config_manager

    monkeypatch.setattr(config_manager.config.comfyui, "runninghub_concurrent_limit", 2)
    pipeline = await _pipeline()
    ctx = SimpleNamespace(
        config=SimpleNamespace(media_workflow=media, tts_inference_mode=tts_mode, tts_workflow=tts_workflow),
        params={},
    )
    assert pipeline._prefetch_limit(ctx) == (2 if expected == "limit" else expected)


# ---------------------------------------------------------------- 3 & 5. safety classification, public message

@pytest.mark.parametrize("message,blocked", [
    ("AiError: Input prompt contains NSFW content.", True),                     # Cloudflare
    ("DataInspectionFailed: Input data may contain inappropriate content.", True),  # DashScope
    ("Error code: 400 - content_policy_violation", True),                      # OpenAI
    ("SensitiveContentDetected", True),                                        # Seedream
    ("HTTP 503 Service Unavailable", False),
])
def test_content_policy_classification(message, blocked):
    assert is_content_policy_rejection(RuntimeError(message)) is blocked


def test_content_policy_follows_cause_chain():
    try:
        try:
            raise ValueError("Input prompt contains NSFW content")
        except ValueError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        assert is_content_policy_rejection(outer)


def test_scene_error_public_message_has_no_paths():
    err = SceneMediaError(2, "a fox", RuntimeError("DataInspectionFailed"), "/data/secret/dir/animation_script.json")
    assert err.safety_blocked
    assert "/data/secret" not in err.public_message and "Scene 3" in err.public_message
    assert "/data/secret" in str(err)          # full detail stays in logs / local UI


# ---------------------------------------------------------------- 4. env reference restore

def test_cleared_env_backed_field_is_saved_cleared(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "s3cr3t")
    refs = {("llm", "api_key"): "${SECRET_KEY}"}
    assert restore_env_refs({"llm": {"api_key": "s3cr3t"}}, refs)["llm"]["api_key"] == "${SECRET_KEY}"
    assert restore_env_refs({"llm": {"api_key": ""}}, refs)["llm"]["api_key"] == ""
    monkeypatch.delenv("SECRET_KEY")
    assert restore_env_refs({"llm": {"api_key": ""}}, refs)["llm"]["api_key"] == "${SECRET_KEY}"  # unchanged


# ---------------------------------------------------------------- 6. segmentation lock / warm-up

def test_warmup_reports_unavailable(monkeypatch):
    seg = ForegroundSegmenter()

    def fail():
        raise SegmentationUnavailable("offline")

    monkeypatch.setattr(seg, "_session", fail)
    assert seg.warmup() is False


def test_segmentation_locks_are_per_model():
    a = ForegroundSegmenter()._key_lock(("a.onnx", "x"))
    b = ForegroundSegmenter()._key_lock(("b.onnx", "x"))
    assert a is not b and a is ForegroundSegmenter()._key_lock(("a.onnx", "x"))


async def test_pipeline_warmup_thread_is_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(ForegroundSegmenter, "warmup", lambda self: calls.append(1) or True)
    monkeypatch.delenv("PIXELLE_SEGMENTATION_WARMUP", raising=False)
    await _pipeline()
    assert calls == []
    monkeypatch.setenv("PIXELLE_SEGMENTATION_WARMUP", "1")
    await _pipeline()                      # core.initialize() builds one AnimationPipeline
    for t in threading.enumerate():
        if t.name == "segmentation-warmup":
            t.join(timeout=5)
    assert calls == [1]


# ---------------------------------------------------------------- 7. precomputed masks

def test_prepare_uses_precomputed_mask_without_segmenting(tmp_path):
    class Exploding:
        def predict(self, image):
            raise AssertionError("must not segment")

    img = Image.new("RGB", (64, 96), (40, 80, 200))
    ImageDraw.Draw(img).ellipse((16, 40, 48, 92), fill=(220, 30, 30))
    path = tmp_path / "s.png"
    img.save(path)
    mask = np.zeros((96, 64), np.float32)
    mask[40:92, 16:48] = 1.0
    renderer = ParallaxRenderer(Exploding())
    assert renderer.prepare(str(path), 64, 96, mask=mask).has_foreground
    assert not renderer.prepare(str(path), 64, 96, segment=False).has_foreground


# ---------------------------------------------------------------- 8. camera mode shares the renderer

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("template", ["1080x1920/video_fullscreen.html", "1080x1920/image_default.html"])
async def test_camera_mode_renders_single_pass(monkeypatch, tmp_path, template):
    from pixelle_video.models.animation import AnimationCharacter, AnimationScene, AnimationScript
    from pixelle_video.models.media import MediaResult
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()

    async def media(prompt, index=1, **kwargs):
        p = tmp_path / f"i{index}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=576x1024",
                        "-frames:v", "1", str(p)], check=True)
        return MediaResult(media_type="image", url=f"file://{p}")

    async def tts(text, output_path=None, **kwargs):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=d=0.8", output_path],
                       check=True)
        return output_path

    monkeypatch.setattr(core, "media", media)
    monkeypatch.setattr(core, "tts", tts)
    script = AnimationScript(title="t", characters=[AnimationCharacter(name="Fox", appearance="red fox")],
                             scenes=[AnimationScene(narration="一", visual="Fox runs", characters=["Fox"]),
                                     AnimationScene(narration="二", visual="Fox sleeps", characters=["Fox"])])
    try:
        result = await core.generate_video(text="", pipeline="animation", script=script, motion_mode="camera",
                                           frame_template=template, media_workflow="api/cloudflare/x")
    except Exception as e:
        if "playwright" in str(e).lower() or "executable" in str(e).lower():
            pytest.skip(f"Playwright browser unavailable: {e}")
        raise
    try:
        frames_dir = Path(result.video_path).parent / "frames"
        assert len(list(frames_dir.glob("*_segment.mp4"))) == 2
        assert not list(frames_dir.glob("*_video.mp4"))      # no intermediate clip encode
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                                "-of", "csv=p=0", result.video_path], capture_output=True, text=True).stdout
        assert "video,1080,1920" in probe and "audio" in probe
    finally:
        shutil.rmtree(Path(result.video_path).parent, ignore_errors=True)



async def test_segmentation_is_serialized_during_prefetch(monkeypatch, tmp_path):
    """Network work is concurrent, but at most one segmentation runs at a time."""
    import time

    from pixelle_video.models.animation import AnimationCharacter, AnimationScene, AnimationScript
    from pixelle_video.models.media import MediaResult
    from pixelle_video.pipelines.linear import PipelineContext
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    pipeline = core.pipelines["animation"]

    async def media(prompt, index=1, **kwargs):
        p = tmp_path / f"i{index}.png"
        Image.new("RGB", (32, 32)).save(p)
        return MediaResult(media_type="image", url=f"file://{p}")

    async def tts(text, output_path=None, **kwargs):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=d=0.3", output_path],
                       check=True)
        return output_path

    async def compose(frame, storyboard, config):
        frame.composed_image_path = frame.image_path

    active = {"now": 0, "max": 0}

    def segment(image_path, w, h):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        active["now"] -= 1
        return None

    monkeypatch.setattr(core, "media", media)
    monkeypatch.setattr(core, "tts", tts)
    monkeypatch.setattr(core.frame_processor, "_step_compose_frame", compose)
    monkeypatch.setattr(pipeline._parallax_renderer(), "segment", segment)
    script = AnimationScript(title="t", characters=[AnimationCharacter(name="A", appearance="a")],
                             scenes=[AnimationScene(narration=f"s{i}", visual="v") for i in range(4)])
    ctx = PipelineContext(input_text="", params={"script": script, "media_workflow": "api/cloudflare/x"})
    await pipeline.setup_environment(ctx)
    await pipeline.generate_content(ctx)
    await pipeline.determine_title(ctx)
    await pipeline.plan_visuals(ctx)
    await pipeline.initialize_storyboard(ctx)
    try:
        await pipeline._prefetch_scene_assets(ctx)
        assert active["max"] == 1
        assert sorted(ctx.extras["masks"]) == [0, 1, 2, 3]
    finally:
        shutil.rmtree(ctx.task_dir, ignore_errors=True)


def test_scene_error_reports_quota_exhaustion():
    from pixelle_video.services.api_services.image_cloudflare import CloudflareQuotaExceeded

    try:
        try:
            raise CloudflareQuotaExceeded("Cloudflare Workers AI daily free allocation is used up")
        except CloudflareQuotaExceeded as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        err = SceneMediaError(0, "p", outer, None)
    assert err.quota_exhausted and not err.safety_blocked
    assert "resets daily" in err.public_message and "Edit scene" not in err.public_message

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

"""Tests for the animation generator (prompts, motion helpers, pipeline)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pixelle_video.models.animation import (
    AnimationCharacter,
    AnimationScene,
    AnimationScript,
    CameraMotion,
)
from pixelle_video.prompts.animation_script import (
    ANIMATION_STYLE_PRESETS,
    build_animation_script_prompt,
    build_scene_image_prompt,
    build_scene_video_prompt,
    get_style_prompt,
)
from pixelle_video.utils.motion_util import (
    build_zoompan_expressions,
    clamp_intensity,
    frame_count,
    supersample_size,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _script() -> AnimationScript:
    return AnimationScript(
        title="Little Fox",
        logline="A fox finds the moon",
        characters=[
            AnimationCharacter(name="Momo", appearance="small orange fox with a blue scarf"),
            AnimationCharacter(name="momo", appearance="duplicate should be dropped"),
            AnimationCharacter(name="Owl", appearance="old grey owl with round glasses"),
        ],
        scenes=[
            AnimationScene(narration="Momo looks at the sky", visual="a fox on a hill at night",
                           characters=["Momo"], motion="tail sways", camera="push-in"),
            AnimationScene(narration="The owl speaks", visual="owl on a branch",
                           characters=["Owl", "Unknown"], camera="weird"),
            AnimationScene(narration="   ", visual="empty scene is dropped"),
        ],
    )


# ---------------------------------------------------------------- models

def test_camera_motion_parse_aliases_and_fallback():
    assert CameraMotion.parse("push-in") == CameraMotion.ZOOM_IN
    assert CameraMotion.parse("Pan Left") == CameraMotion.PAN_LEFT
    assert CameraMotion.parse("tilt_up") == CameraMotion.PAN_UP
    assert CameraMotion.parse("???") == CameraMotion.ZOOM_IN
    assert CameraMotion.parse(None, CameraMotion.STATIC) == CameraMotion.STATIC


def test_scene_camera_is_normalized():
    scene = AnimationScene(narration="n", visual="v", camera="dolly out")
    assert scene.camera == "zoom_out"


# ---------------------------------------------------------------- prompts

def test_style_prompt_resolution():
    assert get_style_prompt("pixel_art") == ANIMATION_STYLE_PRESETS["pixel_art"]["prompt"]
    assert get_style_prompt("missing") == ANIMATION_STYLE_PRESETS["anime"]["prompt"]
    assert get_style_prompt("anime", "  my style  ") == "my style"


def test_script_prompt_modes():
    p = build_animation_script_prompt("a fox", n_scenes=4, mode="generate")
    assert "exactly 4 scenes" in p and "a fox" in p
    p = build_animation_script_prompt("Once upon a time", mode="adapt")
    assert "Do NOT invent new plot" in p
    with pytest.raises(ValueError):
        build_animation_script_prompt("x", mode="bad")


def test_scene_prompt_includes_style_and_character_anchor():
    script = _script()
    prompt = build_scene_image_prompt(script.scenes[1], script.characters, "STYLE")
    assert prompt.startswith("STYLE, owl on a branch")
    assert "Owl (old grey owl with round glasses)" in prompt
    assert "Unknown" not in prompt
    video_prompt = build_scene_video_prompt(script.scenes[0], script.characters, "STYLE")
    assert "Motion: tail sways" in video_prompt and "Camera: zoom in" in video_prompt


# ---------------------------------------------------------------- motion

def test_motion_helpers():
    assert clamp_intensity(9) == 0.5
    assert clamp_intensity("bad") == 0.15
    assert frame_count(1.0, 30) == 30
    assert frame_count(0.01, 30) == 2
    with pytest.raises(ValueError):
        frame_count(0, 30)
    w, h = supersample_size(512, 288)
    assert w % 2 == 0 and h % 2 == 0 and w <= 4096 and w >= 512


@pytest.mark.parametrize("motion", [m.value for m in CameraMotion])
def test_zoompan_expressions_have_no_commas(motion):
    expr = build_zoompan_expressions(motion, 60, 0.2)
    assert set(expr) == {"z", "x", "y"}
    assert all("," not in v for v in expr.values())


# ---------------------------------------------------------------- pipeline

def test_normalize_script():
    from pixelle_video.pipelines.animation import AnimationPipeline

    script = AnimationPipeline._normalize_script(_script(), n_scenes=1, mode="generate")
    assert len(script.scenes) == 1
    assert [c.name for c in script.characters] == ["Momo", "Owl"]


def _make_png(path: Path, color: str):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"color=c={color}:s=512x288", "-frames:v", "1", str(path)],
        check=True,
    )


def _make_audio(path: Path, seconds: float = 1.2):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"sine=d={seconds}", str(path)],
        check=True,
    )


def _frames_differ(video: str) -> bool:
    """Return True if the first and last frames of a clip differ (motion happened)."""
    def grab(select: str) -> bytes:
        return subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", video, "-vf", select,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            check=True, capture_output=True,
        ).stdout
    first = grab("select=eq(n\\,0)")
    last = grab("reverse")
    return first != last


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_motion_clip_moves(tmp_path):
    from pixelle_video.services.video import VideoService

    src = tmp_path / "src.png"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=512x288",
         "-frames:v", "1", str(src)],
        check=True,
    )
    out = VideoService().create_motion_clip(str(src), str(tmp_path / "z.mp4"), duration=1.0,
                                            motion="zoom_in", intensity=0.3)
    assert _frames_differ(out)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
async def test_animation_pipeline_end_to_end_with_mocks(tmp_path, monkeypatch):
    """Run the full camera-mode pipeline with mocked LLM / image / TTS services."""
    from pixelle_video.models.media import MediaResult
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    pipeline = core.pipelines["animation"]

    llm_calls = []

    async def fake_llm(prompt, response_type=None, **kwargs):
        llm_calls.append(prompt)
        assert response_type is AnimationScript
        return _script()

    media_prompts = []

    async def fake_media(prompt, media_type="image", output_path=None, index=1, **kwargs):
        media_prompts.append(prompt)
        path = tmp_path / f"img_{index}.png"
        _make_png(path, ["red", "blue", "green"][index % 3])
        return MediaResult(media_type="image", url=f"file://{path}")

    async def fake_tts(text, output_path=None, **kwargs):
        _make_audio(Path(output_path))
        return output_path

    monkeypatch.setattr(pipeline, "llm", fake_llm)
    monkeypatch.setattr(core, "media", fake_media)
    monkeypatch.setattr(core, "tts", fake_tts)

    from pixelle_video.services.animation_fx import ParallaxRenderer

    class CenterSegmenter:
        """Offline stand-in for the ONNX model: a centered ellipse is the 'character'."""

        def predict(self, image):
            import numpy as np
            from PIL import ImageDraw

            mask = Image.new("L", image.size, 0)
            w, h = image.size
            ImageDraw.Draw(mask).ellipse((w * 0.3, h * 0.3, w * 0.7, h * 0.95), fill=255)
            return np.asarray(mask, dtype=np.float32) / 255.0

    pipeline._renderer = ParallaxRenderer(CenterSegmenter())

    events = []
    try:
        result = await core.generate_video(
            text="A little fox who wants to touch the moon",
            pipeline="animation",
            n_scenes=2,
            art_style="watercolor",
            frame_template="1080x1920/video_default.html",
            media_workflow="selfhost/image_flux.json",
            progress_callback=events.append,
        )
    except Exception as e:  # Headless Chromium may be unavailable in some CI images
        if "playwright" in str(e).lower() or "executable" in str(e).lower():
            pytest.skip(f"Playwright browser unavailable: {e}")
        raise

    try:
        assert Path(result.video_path).exists()
        assert len(result.storyboard.frames) == 2
        assert result.storyboard.title == "Little Fox"
        assert all(f.media_type == "video" for f in result.storyboard.frames)
        assert len(llm_calls) == 1
        assert media_prompts[0].startswith(ANIMATION_STYLE_PRESETS["watercolor"]["prompt"])
        assert "Momo (small orange fox with a blue scarf)" in media_prompts[0]
        assert result.duration > 2.0
        assert events and events[-1].event_type == "completed"

        script_file = Path(result.video_path).parent / "animation_script.json"
        data = json.loads(script_file.read_text(encoding="utf-8"))
        assert data["camera_motions"] == ["zoom_in", "zoom_in"]
        assert data["motion_mode"] == "parallax"
        assert data["ambient_effects"] == ["none", "none"]
        assert len(data["scene_prompts"]) == 2
    finally:
        shutil.rmtree(Path(result.video_path).parent, ignore_errors=True)


async def test_pipeline_validation_errors():
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    with pytest.raises(ValueError, match="video_\\* frame template"):
        await core.generate_video(text="x", pipeline="animation", motion_mode="ai_video",
                                  frame_template="1080x1920/image_default.html",
                                  media_workflow="runninghub/video_wan2.2.json")
    with pytest.raises(ValueError, match="parallax"):
        await core.generate_video(text="x", pipeline="animation",
                                  frame_template="1080x1920/image_default.html")
    with pytest.raises(ValueError, match="static"):
        await core.generate_video(text="x", pipeline="animation",
                                  frame_template="1080x1920/static_default.html")
    with pytest.raises(ValueError, match="non-empty"):
        await core.generate_video(text="  ", pipeline="animation")


# ---------------------------------------------------------------- API

def test_api_styles_and_validation():
    from fastapi.testclient import TestClient

    from api.app import app

    client = TestClient(app)
    styles = client.get("/api/animation/styles").json()["styles"]
    assert {s["key"] for s in styles} == set(ANIMATION_STYLE_PRESETS)

    r = client.post("/api/animation/generate/sync", json={
        "text": "x", "media_workflow": "a", "motion_mode": "ai_video",
        "frame_template": "1080x1920/image_default.html",
    })
    assert r.status_code == 400 and "video_*" in r.json()["detail"]

    r = client.post("/api/animation/generate/sync", json={
        "text": "x", "media_workflow": "a", "min_narration_words": 50, "max_narration_words": 10,
    })
    assert r.status_code == 400

    r = client.post("/api/animation/generate/sync", json={"text": "x"})
    assert r.status_code == 422  # media_workflow is required


async def test_script_llm_fails_fast_on_client_errors(monkeypatch):
    """A 404/401 from the LLM must not be retried; parse errors must be."""
    from pixelle_video.pipelines import animation as anim
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    pipeline = core.pipelines["animation"]

    async def no_sleep(*_):
        return None

    monkeypatch.setattr(anim.asyncio, "sleep", no_sleep)

    class StatusError(Exception):
        def __init__(self, status_code):
            super().__init__(f"HTTP {status_code}")
            self.status_code = status_code

    calls = {"n": 0}

    async def not_found(**kwargs):
        calls["n"] += 1
        raise StatusError(404)

    monkeypatch.setattr(pipeline, "llm", not_found)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        await pipeline._call_script_llm("prompt")
    assert calls["n"] == 1

    calls["n"] = 0

    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("Failed to parse LLM response")
        return _script()

    monkeypatch.setattr(pipeline, "llm", flaky)
    assert (await pipeline._call_script_llm("prompt")).title == "Little Fox"
    assert calls["n"] == 3


# ---------------------------------------------------------------- names & composition

def test_narration_uses_display_names():
    from pixelle_video.pipelines.animation import AnimationPipeline

    script = AnimationScript(
        title="t",
        characters=[
            AnimationCharacter(name="Pipi", display_name="皮皮", appearance="orange fox"),
            AnimationCharacter(name="Lumi", appearance="firefly"),  # no display_name -> falls back
        ],
        scenes=[
            AnimationScene(narration="Pipi 和 PIPI 看著月亮，Lumi 飛過來", visual="Pipi sits",
                           characters=["Pipi", "Lumi"]),
            AnimationScene(narration="Pipiland is not a name match", visual="x"),
        ],
    )
    normalized = AnimationPipeline._normalize_script(script, n_scenes=5, mode="generate")
    assert normalized.scenes[0].narration == "皮皮 和 皮皮 看著月亮，Lumi 飛過來"
    assert normalized.scenes[1].narration == "Pipiland is not a name match"
    assert normalized.characters[1].display_name == "Lumi"
    # Visual prompts keep ASCII names so image models are not fed CJK text
    prompt = build_scene_image_prompt(normalized.scenes[0], normalized.characters, "S")
    assert "Pipi (orange fox)" in prompt and "皮皮" not in prompt


def test_script_prompt_requires_display_name():
    p = build_animation_script_prompt("a fox", n_scenes=3)
    assert '"display_name"' in p and "ASCII" in p


def test_composition_hint():
    from pixelle_video.prompts.animation_script import composition_hint

    assert "vertical" in composition_hint(576, 1024)
    assert "wide" in composition_hint(1024, 576)
    assert composition_hint(512, 512) == ""
    assert composition_hint(None, 10) == ""
    prompt = build_scene_image_prompt(_script().scenes[0], _script().characters, "S",
                                      composition_hint(576, 1024))
    assert "vertical portrait composition" in prompt


@pytest.mark.parametrize("template,size", [
    ("1080x1920/video_fullscreen.html", (576, 1024)),
    ("1920x1080/video_fullscreen.html", (1024, 576)),
])
def test_fullscreen_templates(template, size):
    from pixelle_video.services.frame_html import HTMLFrameGenerator
    from pixelle_video.utils.template_util import get_template_type, resolve_template_path

    assert get_template_type(template) == "video"
    generator = HTMLFrameGenerator(resolve_template_path(template))
    assert generator.get_media_size() == size
    params = generator.parse_template_parameters()
    assert {"title_size", "subtitle_size", "accent_color"} <= set(params)

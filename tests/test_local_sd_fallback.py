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

"""Local stable-diffusion.cpp client and quota-exhaustion fallback."""

import json
import shutil
import stat
from pathlib import Path

import pytest
from PIL import Image

from pixelle_video.services.api_services.image_sdcpp import LocalSDError, SdCppImageClient

FAKE_SD_CLI = """#!/bin/sh
# Fake sd-cli: copies sample.png to the -o path and records its arguments
echo "$@" > "$(dirname "$0")/last_args.txt"
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-o" ]; then out="$2"; fi
  shift
done
cp "$(dirname "$0")/sample.png" "$out"
"""


@pytest.fixture
def sd_dir(tmp_path):
    d = tmp_path / "sd"
    d.mkdir()
    cli = d / "sd-cli"
    cli.write_text(FAKE_SD_CLI)
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    (d / "DreamShaper8_LCM_q8_0.gguf").write_bytes(b"weights")
    (d / "taesd_full.safetensors").write_bytes(b"taesd")
    Image.new("RGB", (512, 896), (10, 120, 200)).save(d / "sample.png")
    return d


def test_build_command(sd_dir):
    client = SdCppImageClient(str(sd_dir))
    cmd = client.build_command("dreamshaper-8-lcm", "a penguin", "/tmp/o.png", 576, 1024, seed=7)
    joined = " ".join(cmd)
    assert "--taesd" in cmd and "--sampling-method lcm" in joined and "--steps 4" in joined
    assert cmd[cmd.index("-W") + 1] == "512" and cmd[cmd.index("-H") + 1] == "896"
    assert cmd[cmd.index("--seed") + 1] == "7"
    with pytest.raises(LocalSDError, match="Unknown local SD model"):
        client.build_command("nope", "p", "/tmp/o.png", 576, 1024)
    (sd_dir / "taesd_full.safetensors").unlink()
    assert "--taesd" not in client.build_command("dreamshaper-8-lcm", "p", "/tmp/o.png", 576, 1024)


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(LocalSDError, match="not found"):
        SdCppImageClient(str(tmp_path)).build_command("dreamshaper-8-lcm", "p", "/tmp/o.png", 512, 512)


def test_generate_image_fits_target_size(sd_dir, tmp_path):
    out = SdCppImageClient(str(sd_dir)).generate_image("a penguin", str(tmp_path / "f.png"), width=576, height=1024)
    assert Image.open(out).size == (576, 1024)
    assert not list(tmp_path.glob("*.sdraw.png"))     # raw output cleaned up
    assert "a penguin" in (sd_dir / "last_args.txt").read_text()


def test_generate_image_failure(sd_dir, tmp_path):
    (sd_dir / "sd-cli").write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(LocalSDError, match="exit 3"):
        SdCppImageClient(str(sd_dir)).generate_image("p", str(tmp_path / "f.png"), width=512, height=512)


async def test_media_service_routes_local_sd(sd_dir, tmp_path):
    from pixelle_video.config import config_manager
    from pixelle_video.service import PixelleVideoCore

    original = config_manager.get_api_providers_config()["local_sd"]
    config_manager.set_api_provider_config("local_sd", {"sd_dir": str(sd_dir)})
    try:
        core = PixelleVideoCore()
        await core.initialize()
        assert "api/local_sd/dreamshaper-8-lcm" in [w["key"] for w in core.api_media.list_workflows()]
        result = await core.media(prompt="a penguin", workflow="api/local_sd/dreamshaper-8-lcm", media_type="image",
                                  width=576, height=1024, output_path=str(tmp_path / "frame.png"))
        assert result.is_image and Image.open(result.url).size == (576, 1024)
        config_manager.set_api_provider_config("local_sd", {"sd_dir": ""})
        with pytest.raises(LocalSDError, match="not configured"):
            await core.media(prompt="x", workflow="api/local_sd/dreamshaper-8-lcm", media_type="image",
                             width=64, height=64, output_path=str(tmp_path / "x.png"))
    finally:
        config_manager.set_api_provider_config("local_sd", original)


# ---------------------------------------------------------------- pipeline fallback

async def _prefetch_with(monkeypatch, tmp_path, media, params):
    from pixelle_video.models.animation import AnimationCharacter, AnimationScene, AnimationScript
    from pixelle_video.pipelines.linear import PipelineContext
    from pixelle_video.service import PixelleVideoCore

    core = PixelleVideoCore()
    await core.initialize()
    pipeline = core.pipelines["animation"]

    async def tts(text, output_path=None, **kwargs):
        shutil.copy(Path(__file__).with_name("conftest.py"), output_path)   # any file; duration patched below
        return output_path

    async def duration(path):
        return 1.0

    async def compose(frame, storyboard, config):
        frame.composed_image_path = frame.image_path

    monkeypatch.setattr(core, "media", media)
    monkeypatch.setattr(core, "tts", tts)
    monkeypatch.setattr(core.frame_processor, "_get_audio_duration", duration)
    monkeypatch.setattr(core.frame_processor, "_step_compose_frame", compose)
    monkeypatch.setattr(pipeline._parallax_renderer(), "segment", lambda *a: None)
    script = AnimationScript(title="t", characters=[AnimationCharacter(name="A", appearance="a")],
                             scenes=[AnimationScene(narration=f"s{i}", visual="v") for i in range(4)])
    ctx = PipelineContext(input_text="", params={"script": script, **params})
    await pipeline.setup_environment(ctx)
    await pipeline.generate_content(ctx)
    await pipeline.determine_title(ctx)
    await pipeline.plan_visuals(ctx)
    await pipeline.initialize_storyboard(ctx)
    return pipeline, ctx


def _fake_media(tmp_path, calls, primary_error):
    from pixelle_video.models.media import MediaResult

    async def media(prompt, workflow=None, index=1, **kwargs):
        calls.append(workflow)
        if workflow.startswith("api/cloudflare/"):
            raise primary_error
        path = tmp_path / f"img_{index}.png"
        Image.new("RGB", (32, 32)).save(path)
        return MediaResult(media_type="image", url=str(path))
    return media


async def test_quota_exhaustion_switches_remaining_scenes_to_fallback(monkeypatch, tmp_path):
    from pixelle_video.services.api_services.image_cloudflare import CloudflareQuotaExceeded

    calls = []
    media = _fake_media(tmp_path, calls, CloudflareQuotaExceeded("daily free allocation is used up"))
    pipeline, ctx = await _prefetch_with(monkeypatch, tmp_path, media, {
        "media_workflow": "api/cloudflare/flux-1-schnell",
        "fallback_media_workflow": "api/local_sd/dreamshaper-8-lcm",
        "prefetch_concurrency": 1,
    })
    try:
        await pipeline._prefetch_scene_assets(ctx)
        assert calls.count("api/cloudflare/flux-1-schnell") == 1          # no repeated quota hits
        assert calls.count("api/local_sd/dreamshaper-8-lcm") == 4
        assert all(f.image_path for f in ctx.storyboard.frames)
        pipeline._save_animation_script(ctx)
        saved = json.loads(Path(ctx.extras["script_path"]).read_text(encoding="utf-8"))
        assert saved["image_fallback_used"] is True
        assert saved["image_workflows"] == ["api/local_sd/dreamshaper-8-lcm"] * 4
    finally:
        shutil.rmtree(ctx.task_dir, ignore_errors=True)


@pytest.mark.parametrize("error,fallback", [
    (RuntimeError("Input prompt contains NSFW content"), "api/local_sd/dreamshaper-8-lcm"),  # safety: no fallback
    ("quota", None),                                                                          # fallback disabled
])
async def test_no_fallback_for_safety_blocks_or_when_disabled(monkeypatch, tmp_path, error, fallback):
    from pixelle_video.pipelines.animation import SceneMediaError
    from pixelle_video.services.api_services.image_cloudflare import CloudflareQuotaExceeded

    if error == "quota":
        error = CloudflareQuotaExceeded("daily free allocation is used up")
    calls = []
    pipeline, ctx = await _prefetch_with(monkeypatch, tmp_path, _fake_media(tmp_path, calls, error), {
        "media_workflow": "api/cloudflare/flux-1-schnell", "fallback_media_workflow": fallback,
    })
    try:
        with pytest.raises(SceneMediaError):
            await pipeline._prefetch_scene_assets(ctx)
        assert not any(c.startswith("api/local_sd/") for c in calls)
    finally:
        shutil.rmtree(ctx.task_dir, ignore_errors=True)


async def test_local_images_are_generated_one_at_a_time(monkeypatch, tmp_path):
    import asyncio

    from pixelle_video.models.media import MediaResult

    active = {"now": 0, "max": 0}

    async def media(prompt, workflow=None, index=1, **kwargs):
        active["now"] += 1
        active["max"] = max(active["max"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        path = tmp_path / f"i{index}.png"
        Image.new("RGB", (32, 32)).save(path)
        return MediaResult(media_type="image", url=str(path))

    pipeline, ctx = await _prefetch_with(monkeypatch, tmp_path, media, {
        "media_workflow": "api/local_sd/dreamshaper-8-lcm", "prefetch_concurrency": 3,
    })
    try:
        await pipeline._prefetch_scene_assets(ctx)
        assert active["max"] == 1
    finally:
        shutil.rmtree(ctx.task_dir, ignore_errors=True)

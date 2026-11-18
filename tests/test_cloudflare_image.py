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

"""Tests for the Cloudflare Workers AI image client (HTTP is mocked)."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from pixelle_video.services.api_services import image_cloudflare as cf
from pixelle_video.services.api_services.image_cloudflare import (
    CloudflareAPIError,
    CloudflareImageClient,
    fit_image,
    request_size,
    resolve_model_id,
)


def _png(w=1024, h=1024, color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _patch_transport(monkeypatch, handler):
    """Route every httpx.AsyncClient created by the module through a mock transport."""
    real = httpx.AsyncClient

    def factory(**kwargs):
        kwargs.pop("proxy", None)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(cf.httpx, "AsyncClient", factory)
    monkeypatch.setattr(cf.asyncio, "sleep", _no_sleep)


async def _no_sleep(*_):
    return None


def test_resolve_and_request_size():
    assert resolve_model_id("flux-1-schnell") == "@cf/black-forest-labs/flux-1-schnell"
    assert resolve_model_id("@cf/custom/x") == "@cf/custom/x"
    assert request_size(512, 288) == (1024, 576)
    assert request_size(1080, 1920) == (576, 1024)
    assert request_size(None, None) == (1024, 1024)


def test_fit_image_center_crop():
    image = fit_image(_png(1024, 1024), 512, 288)
    assert image.size == (512, 288)
    with pytest.raises(CloudflareAPIError):
        fit_image(b"not an image", 10, 10)


def test_build_payload_per_model():
    client = CloudflareImageClient("acc", "tok")
    flux = client.build_payload("@cf/black-forest-labs/flux-1-schnell", "a fox", 512, 288, steps=99, seed=7)
    assert flux == {"prompt": "a fox", "steps": 8, "seed": 7}  # no size, steps clamped
    sdxl = client.build_payload("@cf/bytedance/stable-diffusion-xl-lightning", "a fox", 512, 288,
                                negative_prompt="blurry")
    assert sdxl["width"] == 1024 and sdxl["height"] == 576 and sdxl["negative_prompt"] == "blurry"
    long_prompt = client.build_payload("@cf/black-forest-labs/flux-1-schnell", "x" * 5000)
    assert len(long_prompt["prompt"]) == 2048
    with pytest.raises(ValueError):
        client.build_payload("@cf/black-forest-labs/flux-1-schnell", "  ")


def test_client_requires_credentials():
    with pytest.raises(ValueError):
        CloudflareImageClient("", "tok")
    with pytest.raises(ValueError):
        CloudflareImageClient("acc", " ")


async def test_generate_json_base64_response(monkeypatch, tmp_path):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        payload = {"success": True, "result": {"image": base64.b64encode(_png()).decode()}}
        return httpx.Response(200, json=payload)

    _patch_transport(monkeypatch, handler)
    client = CloudflareImageClient("acc123", "secret-token")
    path = await client.generate_image("a fox", str(tmp_path / "out.png"), model="flux-1-schnell",
                                       width=512, height=288)
    assert seen["url"].endswith("/accounts/acc123/ai/run/@cf/black-forest-labs/flux-1-schnell")
    assert seen["auth"] == "Bearer secret-token"
    assert seen["body"]["prompt"] == "a fox"
    assert Image.open(path).size == (512, 288)


@pytest.mark.parametrize("transient_status", [429, 409, 503])
async def test_generate_binary_response_and_retry(monkeypatch, tmp_path, transient_status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(transient_status, text="transient: AiError: Cog prediction failed")
        return httpx.Response(200, content=_png(1024, 576), headers={"content-type": "image/png"})

    _patch_transport(monkeypatch, handler)
    client = CloudflareImageClient("acc", "tok")
    path = await client.generate_image("a fox", str(tmp_path / "out.jpg"), model="sdxl-lightning",
                                       width=1080, height=1920)
    assert calls["n"] == 2
    assert Image.open(path).size == (1080, 1920)


async def test_auth_error_is_not_retried_and_hides_token(monkeypatch, tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(403, json={"success": False, "errors": [{"message": "forbidden"}]})

    _patch_transport(monkeypatch, handler)
    client = CloudflareImageClient("account-abcdef", "super-secret-token")
    with pytest.raises(CloudflareAPIError) as exc:
        await client.generate_image("a fox", str(tmp_path / "o.png"))
    assert calls["n"] == 1
    assert exc.value.status_code == 403
    assert "super-secret-token" not in str(exc.value)


async def test_api_error_payload(monkeypatch, tmp_path):
    _patch_transport(monkeypatch, lambda r: httpx.Response(
        200, json={"success": False, "errors": [{"message": "bad prompt"}]}))
    client = CloudflareImageClient("acc", "tok", max_retries=0)
    with pytest.raises(CloudflareAPIError, match="bad prompt"):
        await client.generate_image("a fox", str(tmp_path / "o.png"))


async def test_media_service_routes_cloudflare(monkeypatch, tmp_path):
    """core.media('api/cloudflare/...') reaches the Cloudflare client with config credentials."""
    from pixelle_video.config import config_manager
    from pixelle_video.service import PixelleVideoCore

    def handler(request):
        assert request.headers["authorization"] == "Bearer cfg-token"
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    _patch_transport(monkeypatch, handler)
    original = config_manager.get_api_providers_config()["cloudflare"]
    config_manager.set_api_provider_config("cloudflare", {"account_id": "cfg-acc", "api_token": "cfg-token"})
    try:
        core = PixelleVideoCore()
        await core.initialize()
        keys = [w["key"] for w in core.api_media.list_workflows()]
        assert "api/cloudflare/flux-1-schnell" in keys
        result = await core.media(prompt="a fox", workflow="api/cloudflare/sdxl-lightning",
                                  media_type="image", width=512, height=288,
                                  output_path=str(tmp_path / "frame.png"))
        assert result.is_image
        assert Image.open(result.url).size == (512, 288)
    finally:
        config_manager.set_api_provider_config("cloudflare", original)


# ---------------------------------------------------------------- daily quota

async def test_daily_quota_fails_fast_without_retries(monkeypatch, tmp_path):
    from pixelle_video.services.api_services.image_cloudflare import CloudflareQuotaExceeded

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"errors": [{"message": "AiError: you have used up your daily free "
                                                                "allocation of 10,000 neurons, please upgrade"}]})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(CloudflareQuotaExceeded, match="resets daily"):
        await CloudflareImageClient("acc", "tok").generate_image("a fox", str(tmp_path / "o.png"))
    assert calls["n"] == 1


async def test_plain_rate_limit_is_still_retried(monkeypatch, tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="Too many requests")
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    _patch_transport(monkeypatch, handler)
    await CloudflareImageClient("acc", "tok").generate_image("a fox", str(tmp_path / "o.png"))
    assert calls["n"] == 2


async def test_usage_counter_records_images(monkeypatch, tmp_path):
    from pixelle_video.services.api_services import cloudflare_usage

    _patch_transport(monkeypatch, lambda r: httpx.Response(200, content=_png(), headers={"content-type": "image/png"}))
    client = CloudflareImageClient("acc", "tok")
    for i in range(2):
        await client.generate_image("a fox", str(tmp_path / f"o{i}.png"), model="flux-1-schnell")
    usage = cloudflare_usage.today_usage()
    assert usage["images"] == 2
    expected = 2 * cloudflare_usage.estimate_neurons(1024, 1024, 6)
    assert usage["neurons_estimate"] == pytest.approx(expected, abs=0.2)


def test_estimate_and_warning(monkeypatch, tmp_path):
    from pixelle_video.services.api_services import cloudflare_usage

    assert cloudflare_usage.estimate_neurons(1024, 1024, 6) == pytest.approx(4 * 4.8 + 6 * 9.6)
    monkeypatch.setattr(cloudflare_usage, "FREE_DAILY_NEURONS", 100)
    path = tmp_path / "u.json"
    day = cloudflare_usage.record_image(1024, 1024, 6, path=path)
    assert day["ratio"] > 0.7 and json.loads(path.read_text())[day["date"]]["images"] == 1

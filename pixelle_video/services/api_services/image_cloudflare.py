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
Cloudflare Workers AI text-to-image client.

REST endpoint:
    POST {base_url}/accounts/{account_id}/ai/run/{model}
    Authorization: Bearer <api_token>

Response formats differ per model and are both handled:
    - JSON  {"success": true, "result": {"image": "<base64>"}}   (e.g. FLUX.1 schnell)
    - raw image bytes (Content-Type: image/png)                    (e.g. SDXL models)

The generated image is center-cropped and resized locally to the requested
width/height so models with a fixed output size still fit any template.
"""

import asyncio
import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from loguru import logger
from PIL import Image

DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
# 409 is returned by Workers AI for transient inference failures (e.g. "Cog prediction failed")
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CloudflareImageModel:
    """Per-model request capabilities."""

    model_id: str
    supports_size: bool = False
    max_steps: int = 8
    default_steps: int = 4
    supports_negative_prompt: bool = False
    max_prompt_chars: int = 2048


CLOUDFLARE_IMAGE_MODELS: Dict[str, CloudflareImageModel] = {
    m.model_id: m
    for m in (
        CloudflareImageModel("@cf/black-forest-labs/flux-1-schnell", max_steps=8, default_steps=6),
        CloudflareImageModel(
            "@cf/bytedance/stable-diffusion-xl-lightning",
            supports_size=True, max_steps=20, default_steps=8, supports_negative_prompt=True,
        ),
        CloudflareImageModel(
            "@cf/stabilityai/stable-diffusion-xl-base-1.0",
            supports_size=True, max_steps=20, default_steps=20, supports_negative_prompt=True,
        ),
        CloudflareImageModel(
            "@cf/lykon/dreamshaper-8-lcm",
            supports_size=True, max_steps=20, default_steps=8, supports_negative_prompt=True,
        ),
    )
}
DEFAULT_CLOUDFLARE_IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"

# Short names used in Pixelle workflow keys: api/cloudflare/<alias>
CLOUDFLARE_MODEL_ALIASES: Dict[str, str] = {
    "flux-1-schnell": "@cf/black-forest-labs/flux-1-schnell",
    "sdxl-lightning": "@cf/bytedance/stable-diffusion-xl-lightning",
    "sdxl-base-1.0": "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    "dreamshaper-8-lcm": "@cf/lykon/dreamshaper-8-lcm",
}


def resolve_model_id(name: str) -> str:
    """Map a short alias (or a full @cf/... id) to the Workers AI model id."""
    if not name:
        return DEFAULT_CLOUDFLARE_IMAGE_MODEL
    return CLOUDFLARE_MODEL_ALIASES.get(name, name)


class CloudflareQuotaExceeded(RuntimeError):
    """The account's daily Workers AI allowance is used up (not retried)."""

    quota_exhausted = True


def is_daily_quota_error(status_code: int, body: str) -> bool:
    text = (body or "").lower()
    return status_code == 429 and ("daily free allocation" in text or "neurons" in text)


class CloudflareAPIError(RuntimeError):
    """Raised for non-retryable or exhausted Cloudflare API failures."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _mask(value: str) -> str:
    return f"{value[:4]}…{value[-4:]}" if value and len(value) > 8 else "***"


def request_size(width: Optional[int], height: Optional[int], long_side: int = 1024) -> tuple[int, int]:
    """
    Generation size for models that accept width/height: keep aspect ratio,
    scale the longest side to `long_side`, round to multiples of 64, clamp 256-2048.
    """
    if not width or not height:
        return long_side, long_side
    scale = long_side / float(max(width, height))

    def fit(v: float) -> int:
        return int(max(256, min(2048, round(v * scale / 64) * 64)))

    return fit(width), fit(height)


def fit_image(data: bytes, width: Optional[int], height: Optional[int]) -> Image.Image:
    """Decode bytes, then center-crop + resize to (width, height) if given."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as e:
        raise CloudflareAPIError(f"Cloudflare returned data that is not a valid image: {e}") from e
    image = image.convert("RGB")
    if not width or not height:
        return image

    src_w, src_h = image.size
    target_ratio = width / float(height)
    if src_w / float(src_h) > target_ratio:
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(round(src_w / target_ratio))
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))
    return image.resize((int(width), int(height)), Image.LANCZOS)


def extract_image_bytes(content_type: str, body: bytes) -> bytes:
    """Return raw image bytes from either a JSON (base64) or binary response."""
    if content_type.startswith("image/"):
        return body
    try:
        import json

        payload = json.loads(body)
    except Exception as e:
        raise CloudflareAPIError(f"Unexpected Cloudflare response ({content_type}): {body[:200]!r}") from e

    if not payload.get("success", True):
        raise CloudflareAPIError(f"Cloudflare API error: {payload.get('errors') or payload}")
    result = payload.get("result") or {}
    image_b64 = result.get("image") if isinstance(result, dict) else None
    if not image_b64:
        raise CloudflareAPIError(f"Cloudflare response has no image: {str(payload)[:200]}")
    try:
        return base64.b64decode(image_b64)
    except Exception as e:
        raise CloudflareAPIError(f"Failed to decode base64 image from Cloudflare: {e}") from e


class CloudflareImageClient:
    """Async client for Cloudflare Workers AI text-to-image models."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        base_url: Optional[str] = None,
        local_proxy: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        if not account_id or not account_id.strip():
            raise ValueError("Cloudflare account_id is not configured")
        if not api_token or not api_token.strip():
            raise ValueError("Cloudflare api_token is not configured")
        self.account_id = account_id.strip()
        self._api_token = api_token.strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.local_proxy = local_proxy or None
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))

    def _endpoint(self, model: str) -> str:
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{model}"

    def build_payload(
        self,
        model: str,
        prompt: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = CLOUDFLARE_IMAGE_MODELS.get(model) or CloudflareImageModel(model)
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if len(prompt) > spec.max_prompt_chars:
            logger.warning(f"Prompt truncated from {len(prompt)} to {spec.max_prompt_chars} chars")
            prompt = prompt[: spec.max_prompt_chars]

        payload: Dict[str, Any] = {"prompt": prompt}
        n_steps = max(1, min(spec.max_steps, int(steps or spec.default_steps)))
        if model == DEFAULT_CLOUDFLARE_IMAGE_MODEL:
            payload["steps"] = n_steps
        else:
            payload["num_steps"] = n_steps
        if seed is not None:
            payload["seed"] = int(seed)
        if spec.supports_size:
            payload["width"], payload["height"] = request_size(width, height)
        if negative_prompt and spec.supports_negative_prompt:
            payload["negative_prompt"] = negative_prompt
        return payload

    async def _post(self, model: str, payload: Dict[str, Any]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_token}", "Content-Type": "application/json"}
        client_kwargs: Dict[str, Any] = {"timeout": self.timeout}
        if self.local_proxy:
            client_kwargs["proxy"] = self.local_proxy

        last_error: Optional[Exception] = None
        async with httpx.AsyncClient(**client_kwargs) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(self._endpoint(model), json=payload, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = e
                    logger.warning(f"Cloudflare request error (attempt {attempt + 1}): {e}")
                else:
                    if response.status_code < 400:
                        return response
                    detail = response.text[:300]
                    if is_daily_quota_error(response.status_code, response.text):
                        raise CloudflareQuotaExceeded(
                            "Cloudflare Workers AI daily free allocation is used up; it resets daily (exact reset time is set by Cloudflare). "
                            f"Detail: {detail}"
                        )
                    if response.status_code in (401, 403):
                        raise CloudflareAPIError(
                            f"Cloudflare authentication failed ({response.status_code}). Check that the "
                            f"API token has 'Workers AI' permission and the account ID "
                            f"({_mask(self.account_id)}) is correct. Detail: {detail}",
                            response.status_code,
                        )
                    if response.status_code not in RETRYABLE_STATUS:
                        raise CloudflareAPIError(
                            f"Cloudflare API error {response.status_code} for {model}: {detail}",
                            response.status_code,
                        )
                    last_error = CloudflareAPIError(
                        f"Cloudflare API error {response.status_code}: {detail}", response.status_code
                    )
                    logger.warning(f"Cloudflare returned {response.status_code} (attempt {attempt + 1})")

                if attempt < self.max_retries:
                    await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))

        raise CloudflareAPIError(f"Cloudflare request failed after retries: {last_error}")

    async def generate_image(
        self,
        prompt: str,
        output_path: str,
        model: str = DEFAULT_CLOUDFLARE_IMAGE_MODEL,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate one image and save it as PNG at `output_path`.

        Returns:
            The absolute output path.
        """
        model = resolve_model_id(model)
        payload = self.build_payload(model, prompt, width, height, steps, seed, negative_prompt)
        logger.info(
            f"Cloudflare Workers AI image: model={model}, account={_mask(self.account_id)}, "
            f"target={width}x{height}, params={ {k: v for k, v in payload.items() if k != 'prompt'} }"
        )
        response = await self._post(model, payload)
        content_type = response.headers.get("content-type", "").lower()
        image = fit_image(extract_image_bytes(content_type, response.content), width, height)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            path = path.with_suffix(".png")
        image.save(path)
        logger.success(f"Cloudflare image saved: {path} ({image.size[0]}x{image.size[1]})")
        try:
            from pixelle_video.services.api_services.cloudflare_usage import record_image

            gen_w, gen_h = (payload.get("width"), payload.get("height"))
            record_image(gen_w or 1024, gen_h or 1024, int(payload.get("steps") or payload.get("num_steps") or 1))
        except Exception as e:  # usage tracking must never break generation
            logger.debug(f"Cloudflare usage tracking failed: {e}")
        return str(path.resolve())

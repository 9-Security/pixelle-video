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
Foreground (character) segmentation for 2.5D parallax.

Uses the open-source IS-Net anime segmentation model (ONNX, ~176 MB, CPU) from
the rembg project releases. The model is downloaded once into a cache directory
and verified with SHA-256 before use. Any failure (no network, bad checksum,
missing onnxruntime) is reported as SegmentationUnavailable so callers can fall
back to single-layer rendering instead of failing the whole video.
"""

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from loguru import logger
from PIL import Image


@dataclass(frozen=True)
class SegmentationModel:
    name: str
    url: str
    sha256: str
    input_size: int = 1024


ISNET_ANIME = SegmentationModel(
    name="isnet-anime.onnx",
    url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx",
    sha256="f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99",
)


class SegmentationUnavailable(RuntimeError):
    """Raised when the segmentation model cannot be loaded or run."""


def default_model_dir() -> Path:
    """Model cache dir: $PIXELLE_MODEL_DIR or ~/.cache/pixelle-video/models."""
    env = os.environ.get("PIXELLE_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "pixelle-video" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(model: SegmentationModel = ISNET_ANIME, model_dir: Optional[Path] = None,
                 timeout: float = 600.0) -> Path:
    """
    Return the local model path, downloading and verifying it if needed.

    The download goes to a temp file in the same directory and is renamed only
    after the checksum matches, so a partial or tampered file is never used.
    """
    model_dir = Path(model_dir or default_model_dir())
    path = model_dir / model.name
    if path.exists():
        if _sha256(path) == model.sha256:
            return path
        logger.warning(f"Checksum mismatch for cached {path}; re-downloading")
        path.unlink()

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading segmentation model {model.name} (~176 MB) to {model_dir}")
    fd, tmp_name = tempfile.mkstemp(prefix=model.name + ".", suffix=".part", dir=model_dir)
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as f, httpx.stream(
            "GET", model.url, follow_redirects=True, timeout=timeout
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(1024 * 1024):
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != model.sha256:
            raise SegmentationUnavailable(
                f"Checksum mismatch for {model.name}: expected {model.sha256}, got {digest.hexdigest()}"
            )
        tmp.replace(path)
        logger.success(f"Segmentation model ready: {path}")
        return path
    except SegmentationUnavailable:
        raise
    except Exception as e:
        raise SegmentationUnavailable(f"Failed to download {model.name}: {e}") from e
    finally:
        if tmp.exists():
            tmp.unlink()


class ForegroundSegmenter:
    """Lazy, thread-safe ONNX segmenter returning a float mask in [0, 1]."""

    _lock = threading.Lock()
    _sessions: dict = {}

    def __init__(self, model: SegmentationModel = ISNET_ANIME, model_dir: Optional[Path] = None):
        self.model = model
        self.model_dir = model_dir

    def _session(self):
        key = (self.model.name, str(self.model_dir))
        with self._lock:
            if key not in self._sessions:
                try:
                    import onnxruntime as ort
                except ImportError as e:
                    raise SegmentationUnavailable("onnxruntime is not installed") from e
                path = ensure_model(self.model, self.model_dir)
                options = ort.SessionOptions()
                options.log_severity_level = 3
                try:
                    self._sessions[key] = ort.InferenceSession(
                        str(path), sess_options=options, providers=["CPUExecutionProvider"]
                    )
                except Exception as e:
                    raise SegmentationUnavailable(f"Failed to load {path}: {e}") from e
            return self._sessions[key]

    def preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Letterbox to a square canvas (keeps aspect ratio) and scale to [0, 1].

        Matches the anime-segmentation reference inference: no mean/std
        normalization, and no stretching (stretching visibly degrades masks).

        Returns:
            (NCHW tensor, (offset_x, offset_y, scaled_w, scaled_h))
        """
        size = self.model.input_size
        w, h = image.size
        scale = size / float(max(w, h))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        ox, oy = (size - nw) // 2, (size - nh) // 2
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(image.convert("RGB").resize((nw, nh), Image.BILINEAR), (ox, oy))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)[None, ...].astype(np.float32), (ox, oy, nw, nh)

    def predict(self, image: Image.Image) -> np.ndarray:
        """Return an (H, W) float32 mask in [0, 1] at the image's resolution."""
        session = self._session()
        tensor, (ox, oy, nw, nh) = self.preprocess(image)
        try:
            output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        except Exception as e:
            raise SegmentationUnavailable(f"Segmentation inference failed: {e}") from e
        mask = np.asarray(output, dtype=np.float32).reshape(output.shape[-2], output.shape[-1])
        mask = np.clip(mask[oy:oy + nh, ox:ox + nw], 0.0, 1.0)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        return np.asarray(mask_img, dtype=np.float32) / 255.0

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
Split a still image into parallax layers.

- foreground: the character(s), premultiplied RGB + feathered alpha
- background: the original image with the foreground removed and the hole
  filled by OpenCV inpainting, slightly blurred for a depth-of-field look

When there is no usable foreground (mask too small or covering most of the
frame), only the background layer is produced and the renderer falls back to
single-layer motion.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image

MIN_FG_AREA = 0.02
MAX_FG_AREA = 0.75


@dataclass
class SceneLayers:
    background: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    foreground: Optional[np.ndarray] = None     # (H, W, 3) float32, premultiplied by alpha
    alpha: Optional[np.ndarray] = None          # (H, W) float32 in [0, 1]
    anchor: Tuple[float, float] = (0.0, 0.0)    # bottom-center of the foreground bbox (px)
    fg_area: float = 0.0

    @property
    def has_foreground(self) -> bool:
        return self.foreground is not None and self.alpha is not None

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.background.shape[:2]
        return w, h


def to_float_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def build_layers(
    image: Image.Image,
    mask: Optional[np.ndarray],
    threshold: float = 0.5,
    feather_px: float = 1.5,
    dof_sigma: float = 1.2,
) -> SceneLayers:
    """
    Build parallax layers from an RGB image and a soft foreground mask.

    Args:
        image: Source image (already at output resolution)
        mask: (H, W) float mask in [0, 1], or None for single-layer output
        threshold: Mask level treated as foreground for area checks/inpainting
        feather_px: Gaussian feather applied to the alpha edge
        dof_sigma: Background blur (0 disables) to separate planes
    """
    rgb = to_float_rgb(image)
    h, w = rgb.shape[:2]
    if mask is None:
        return SceneLayers(background=rgb)
    if mask.shape != (h, w):
        raise ValueError(f"Mask shape {mask.shape} does not match image size {(h, w)}")

    hard = (mask >= threshold).astype(np.uint8)
    area = float(hard.mean())
    if not MIN_FG_AREA <= area <= MAX_FG_AREA:
        logger.info(f"Foreground area {area:.1%} outside [{MIN_FG_AREA:.0%}, {MAX_FG_AREA:.0%}]; single layer")
        return SceneLayers(background=rgb, fg_area=area)

    # Keep only reasonably large connected regions (drops speckles)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    keep = np.zeros_like(hard)
    min_region = max(64, int(0.002 * h * w))
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_region:
            keep[labels == label] = 1
    if keep.sum() == 0:
        return SceneLayers(background=rgb, fg_area=area)

    alpha = np.clip(mask, 0.0, 1.0) * cv2.dilate(keep, np.ones((5, 5), np.uint8)).astype(np.float32)
    if feather_px > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_px)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    background = inpaint_background(rgb, keep)
    if dof_sigma > 0:
        background = cv2.GaussianBlur(background, (0, 0), dof_sigma)

    ys, xs = np.nonzero(keep)
    anchor = (float(xs.min() + xs.max()) / 2.0, float(ys.max()))
    foreground = rgb * alpha[..., None]
    return SceneLayers(
        background=background.astype(np.float32),
        foreground=foreground.astype(np.float32),
        alpha=alpha,
        anchor=anchor,
        fg_area=float(keep.mean()),
    )


def inpaint_background(rgb: np.ndarray, hard_mask: np.ndarray, work_long_side: int = 512) -> np.ndarray:
    """
    Fill the foreground hole. Inpainting runs at reduced resolution (fast and
    smooth), is upscaled, and replaces only the dilated hole region.
    """
    h, w = rgb.shape[:2]
    dilate_px = max(3, int(round(0.015 * max(h, w))))
    hole = cv2.dilate(hard_mask, np.ones((dilate_px * 2 + 1,) * 2, np.uint8))

    scale = min(1.0, work_long_side / float(max(h, w)))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize((rgb * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(hole, (sw, sh), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_hole, 7, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0

    blend = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), max(1.0, dilate_px / 2.0))[..., None]
    return rgb * (1.0 - blend) + filled * blend

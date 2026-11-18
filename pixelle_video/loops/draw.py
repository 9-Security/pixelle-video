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

"""Small anti-aliased, sub-pixel drawing helpers on top of OpenCV (BGR uint8 frames)."""

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

SHIFT = 4  # fixed-point bits for sub-pixel drawing
_SCALE = 1 << SHIFT

Color = Tuple[int, int, int]


def _fp(v: float) -> int:
    return int(round(v * _SCALE))


def _pt(p: Sequence[float]) -> Tuple[int, int]:
    return _fp(p[0]), _fp(p[1])


def hex_bgr(code: str) -> Color:
    code = code.lstrip("#")
    r, g, b = (int(code[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def mix(a: Color, b: Color, k: float) -> Color:
    return tuple(int(round(a[i] + (b[i] - a[i]) * k)) for i in range(3))


def vertical_gradient(w: int, h: int, top: Color, bottom: Color) -> np.ndarray:
    k = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    img = np.array(top, np.float32) * (1 - k) + np.array(bottom, np.float32) * k
    return np.repeat(img, w, axis=1).astype(np.uint8)


def vignette(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    d = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
    mask = 1 - strength * np.clip(d / 2, 0, 1) ** 1.2
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)


def circle(img: np.ndarray, c: Sequence[float], r: float, color: Color, thickness: int = -1) -> None:
    cv2.circle(img, _pt(c), max(1, _fp(r)), color, thickness, cv2.LINE_AA, SHIFT)


def line(img: np.ndarray, a: Sequence[float], b: Sequence[float], color: Color, thickness: int = 1) -> None:
    cv2.line(img, _pt(a), _pt(b), color, max(1, int(round(thickness))), cv2.LINE_AA, SHIFT)


def ellipse(img: np.ndarray, c: Sequence[float], axes: Sequence[float], angle: float, color: Color,
            thickness: int = -1, start: float = 0, end: float = 360) -> None:
    cv2.ellipse(img, _pt(c), (max(1, _fp(axes[0])), max(1, _fp(axes[1]))), angle, start, end, color,
                thickness, cv2.LINE_AA, SHIFT)


def polygon(img: np.ndarray, pts: Iterable[Sequence[float]], color: Color, closed: bool = True,
            thickness: int = -1) -> None:
    arr = np.array([_pt(p) for p in pts], dtype=np.int32).reshape(-1, 1, 2)
    if thickness < 0:
        cv2.fillPoly(img, [arr], color, cv2.LINE_AA, SHIFT)
    else:
        cv2.polylines(img, [arr], closed, color, thickness, cv2.LINE_AA, SHIFT)


def polyline(img: np.ndarray, pts: Iterable[Sequence[float]], color: Color, thickness: int = 2) -> None:
    polygon(img, pts, color, closed=False, thickness=max(1, int(round(thickness))))


def rotated_rect(center: Sequence[float], w: float, h: float, angle_rad: float,
                 pivot: Optional[Sequence[float]] = None) -> np.ndarray:
    """Corners of a w x h rectangle centred at `center`, rotated by angle around `pivot` (default centre)."""
    cx, cy = center
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) + [cx, cy]
    px, py = pivot if pivot is not None else (cx, cy)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rel = corners - [px, py]
    return np.stack([rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1) + [px, py]


def blend_layer(dst: np.ndarray, layer: np.ndarray, alpha: float, roi: Optional[Tuple[int, int, int, int]] = None) -> None:
    """In-place dst = dst*(1-alpha) + layer*alpha, optionally only inside roi (x0, y0, x1, y1)."""
    if alpha <= 0:
        return
    if roi is None:
        cv2.addWeighted(layer, alpha, dst, 1 - alpha, 0, dst)
        return
    x0, y0, x1, y1 = clip_roi(roi, dst.shape)
    if x1 <= x0 or y1 <= y0:
        return
    sub = dst[y0:y1, x0:x1]
    cv2.addWeighted(layer[y0:y1, x0:x1], alpha, sub, 1 - alpha, 0, sub)


def clip_roi(roi: Tuple[float, float, float, float], shape) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    return max(0, int(x0)), max(0, int(y0)), min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))


def soft_shadow(dst: np.ndarray, c: Sequence[float], axes: Sequence[float], strength: float = 0.35,
                blur: float = 12.0) -> None:
    """Darken an elliptical blurred area (drop shadow), only touching its bounding box."""
    pad = blur * 3
    x0, y0 = c[0] - axes[0] - pad, c[1] - axes[1] - pad
    x1, y1 = c[0] + axes[0] + pad, c[1] + axes[1] + pad
    bx0, by0, bx1, by1 = clip_roi((x0, y0, x1, y1), dst.shape)
    if bx1 <= bx0 or by1 <= by0:
        return
    mask = np.zeros((by1 - by0, bx1 - bx0), np.float32)
    ellipse(mask, (c[0] - bx0, c[1] - by0), axes, 0, 1.0)
    k = max(3, int(blur) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), blur)
    sub = dst[by0:by1, bx0:bx1].astype(np.float32)
    sub *= (1 - strength * mask)[..., None]
    dst[by0:by1, bx0:bx1] = sub.astype(np.uint8)


def glow(dst: np.ndarray, c: Sequence[float], r: float, color: Color, strength: float = 0.5) -> None:
    """Additive soft glow around a point."""
    pad = r * 2
    bx0, by0, bx1, by1 = clip_roi((c[0] - pad, c[1] - pad, c[0] + pad, c[1] + pad), dst.shape)
    if bx1 <= bx0 or by1 <= by0:
        return
    y, x = np.ogrid[by0:by1, bx0:bx1]
    d2 = ((x - c[0]) ** 2 + (y - c[1]) ** 2) / (r * r)
    m = (np.exp(-d2) * strength)[..., None]
    sub = dst[by0:by1, bx0:bx1].astype(np.float32)
    sub += m * np.array(color, np.float32)
    dst[by0:by1, bx0:bx1] = np.clip(sub, 0, 255).astype(np.uint8)


def periodic_noise_texture(w: int, h: int, rng: np.random.Generator, cells: Tuple[int, int] = (8, 6),
                           octaves: int = 3) -> np.ndarray:
    """Smooth value noise in [0, 1] that tiles horizontally (wraps at x = w)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    cx, cy = cells
    for _ in range(octaves):
        grid = rng.random((cy + 1, cx)).astype(np.float32)
        grid = np.concatenate([grid, grid[:, :1]], axis=1)  # wrap horizontally
        layer = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += layer * amp
        total += amp
        amp *= 0.5
        cx, cy = cx * 2, cy * 2
    return np.clip(out / total, 0, 1)

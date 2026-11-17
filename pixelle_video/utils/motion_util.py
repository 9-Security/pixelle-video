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
Camera motion helpers (Ken Burns effect) for FFmpeg's zoompan filter.

Pure functions only, so they can be unit-tested without FFmpeg.
Expressions intentionally avoid commas so they survive filter-graph escaping.
"""

import math
from typing import Dict, Tuple

from pixelle_video.models.animation import CameraMotion

MIN_INTENSITY = 0.0
MAX_INTENSITY = 0.5
DEFAULT_INTENSITY = 0.15


def clamp_intensity(intensity: float) -> float:
    """Clamp zoom/pan intensity into a safe range."""
    try:
        value = float(intensity)
    except (TypeError, ValueError):
        return DEFAULT_INTENSITY
    if math.isnan(value):
        return DEFAULT_INTENSITY
    return max(MIN_INTENSITY, min(MAX_INTENSITY, value))


def frame_count(duration: float, fps: int) -> int:
    """Number of output frames for a clip (at least 2 so progress is defined)."""
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return max(2, int(math.ceil(duration * fps)))


def even(value: int) -> int:
    """Round down to an even number (libx264 + yuv420p requirement), minimum 2."""
    return max(2, int(value) // 2 * 2)


def build_zoompan_expressions(
    motion: CameraMotion | str,
    n_frames: int,
    intensity: float = DEFAULT_INTENSITY,
) -> Dict[str, str]:
    """
    Build zoompan z/x/y expressions for a camera motion.

    Uses zoompan variables: `on` (output frame index), `iw`/`ih` (input size),
    `zoom` (current zoom). Progress p goes 0 -> 1 linearly across the clip.

    Returns:
        dict with keys "z", "x", "y"
    """
    motion = CameraMotion.parse(motion if isinstance(motion, str) else motion.value)
    k = clamp_intensity(intensity)
    last = max(1, n_frames - 1)
    p = f"(on/{last})"
    center_x = "(iw/2-iw/zoom/2)"
    center_y = "(ih/2-ih/zoom/2)"
    span_x = "(iw-iw/zoom)"
    span_y = "(ih-ih/zoom)"

    if motion == CameraMotion.STATIC or k == 0:
        return {"z": "1", "x": "0", "y": "0"}
    if motion == CameraMotion.ZOOM_IN:
        return {"z": f"1+{k}*{p}", "x": center_x, "y": center_y}
    if motion == CameraMotion.ZOOM_OUT:
        return {"z": f"1+{k}-{k}*{p}", "x": center_x, "y": center_y}

    # Pans keep a constant zoom so there is room to move
    z = f"{1 + k:.4f}"
    if motion == CameraMotion.PAN_LEFT:
        return {"z": z, "x": f"{span_x}*(1-{p})", "y": center_y}
    if motion == CameraMotion.PAN_RIGHT:
        return {"z": z, "x": f"{span_x}*{p}", "y": center_y}
    if motion == CameraMotion.PAN_UP:
        return {"z": z, "x": center_x, "y": f"{span_y}*(1-{p})"}
    if motion == CameraMotion.PAN_DOWN:
        return {"z": z, "x": center_x, "y": f"{span_y}*{p}"}

    return {"z": "1", "x": "0", "y": "0"}  # pragma: no cover - exhaustive above


def supersample_size(width: int, height: int, max_side: int = 4096) -> Tuple[int, int]:
    """
    Size to upscale the source image to before zoompan.

    zoompan crops at integer input pixels; supersampling reduces visible jitter.
    The factor is capped so the longest side stays below `max_side`.
    """
    longest = max(width, height)
    factor = max(1.0, min(4.0, max_side / float(longest)))
    return even(int(width * factor)), even(int(height * factor))

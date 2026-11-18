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
Seamless loop scenes.

A scene is a pure function of time that is periodic by construction: every
moving part is expressed through `phase = (t % period) / period` (or an integer
number of cycles per period), so frame(t) == frame(t + period) and the clip can
repeat forever without a visible seam. Sound follows the same rule
(see `pixelle_video.loops.audio`).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type

import numpy as np

from pixelle_video.loops.audio import SoundDesign


@dataclass(frozen=True)
class SceneInfo:
    name: str
    title_zh: str
    title_en: str
    description: str
    period: int  # seconds; integer so period * fps and period * sample_rate are integers
    kind: str = "soothing"  # soothing | suspense


class LoopScene(ABC):
    """Base class: subclasses set `info` and implement `setup`, `render` and `sound_design`."""

    info: SceneInfo

    def __init__(self, width: int = 1920, height: int = 1080, seed: int = 7):
        if width < 160 or height < 90:
            raise ValueError("Frame size too small")
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        # Scale relative to a 1920x1080 design canvas (uses the short side for portrait frames)
        self.s = min(self.width / 1920.0, self.height / 1080.0)
        self.ox = (self.width - 1920.0 * self.s) / 2
        self.oy = (self.height - 1080.0 * self.s) / 2
        self.setup()

    def P(self, x: float, y: float) -> np.ndarray:
        """Map a point on the 1920x1080 design canvas to frame pixels (scaled and centred)."""
        return np.array([self.ox + x * self.s, self.oy + y * self.s])

    @property
    def period(self) -> int:
        return self.info.period

    def phase(self, t: float) -> float:
        return (t % self.period) / self.period

    def cycles(self, t: float, n: int) -> float:
        """Phase in [0, 1) of a motion that repeats n times per period."""
        return (t * n / self.period) % 1.0

    @abstractmethod
    def setup(self) -> None:
        """Precompute static layers (background, textures, random schedules)."""

    @abstractmethod
    def render(self, t: float) -> np.ndarray:
        """Return the BGR uint8 frame (height, width, 3) at time t."""

    def sound_design(self) -> SoundDesign:
        """Periodic ambience layers and timed sound events (default: soft room tone)."""
        return SoundDesign()


SCENES: Dict[str, Type[LoopScene]] = {}


def register(cls: Type[LoopScene]) -> Type[LoopScene]:
    name = cls.info.name
    if name in SCENES:
        raise ValueError(f"Duplicate loop scene '{name}'")
    SCENES[name] = cls
    return cls


def get_scene(name: str) -> Type[LoopScene]:
    from pixelle_video.loops import scenes  # noqa: F401  (registers built-in scenes)

    try:
        return SCENES[name]
    except KeyError:
        raise KeyError(f"Unknown loop scene '{name}'. Available: {', '.join(sorted(SCENES))}") from None


def list_scenes() -> List[SceneInfo]:
    from pixelle_video.loops import scenes  # noqa: F401

    return [cls.info for cls in SCENES.values()]


# ---------------------------------------------------------------------------
# Easing helpers (all map [0, 1] -> [0, 1])
# ---------------------------------------------------------------------------

def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * np.cos(np.pi * x)


def segment(p: float, start: float, end: float) -> float:
    """Progress of phase p within [start, end], clamped to [0, 1]."""
    if end <= start:
        return 1.0 if p >= end else 0.0
    return min(1.0, max(0.0, (p - start) / (end - start)))


Color = Tuple[int, int, int]

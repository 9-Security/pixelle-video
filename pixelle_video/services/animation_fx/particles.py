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
Stateless ambient particle effects.

Every particle position is a pure function of (seed, time), so any frame can be
rendered independently and output is reproducible for the same seed.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

AMBIENT_EFFECTS = ("none", "fireflies", "stars", "snow", "rain", "petals", "dust", "embers", "bubbles")


@dataclass(frozen=True)
class EffectSpec:
    count: int                      # particles per 1M output pixels
    color: Tuple[float, float, float]
    size: Tuple[float, float]       # radius range, as fraction of frame width
    velocity: Tuple[float, float]   # px/s per 1000 px of width (x, y)
    sway: float                     # horizontal sway amplitude (fraction of width)
    additive: bool                  # glow (add) or paint over (alpha)
    glow: float                     # blur sigma as fraction of width (0 = none)
    pulse: float                    # brightness pulse amount 0..1
    opacity: float


EFFECTS: Dict[str, EffectSpec] = {
    "fireflies": EffectSpec(40, (1.0, 0.92, 0.45), (0.003, 0.006), (0, -8), 0.03, True, 0.012, 0.9, 0.9),
    "stars": EffectSpec(110, (1.0, 1.0, 0.95), (0.0015, 0.003), (0, 0), 0.0, True, 0.004, 0.8, 0.9),
    "snow": EffectSpec(160, (1.0, 1.0, 1.0), (0.002, 0.005), (6, 70), 0.02, False, 0.002, 0.0, 0.85),
    "rain": EffectSpec(220, (0.75, 0.82, 0.95), (0.0008, 0.0012), (-60, 900), 0.0, False, 0.0, 0.0, 0.35),
    "petals": EffectSpec(45, (1.0, 0.72, 0.80), (0.006, 0.010), (25, 55), 0.05, False, 0.0, 0.0, 0.9),
    "dust": EffectSpec(70, (1.0, 0.95, 0.85), (0.0012, 0.0025), (6, -4), 0.02, True, 0.004, 0.5, 0.55),
    "embers": EffectSpec(60, (1.0, 0.55, 0.15), (0.0018, 0.0035), (8, -90), 0.03, True, 0.008, 0.8, 0.95),
    "bubbles": EffectSpec(40, (0.85, 0.95, 1.0), (0.004, 0.009), (0, -60), 0.02, False, 0.0, 0.0, 0.5),
}


def normalize_effect(value) -> str:
    """Map loose LLM output to a known effect name (unknown -> none)."""
    if not value:
        return "none"
    key = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"firefly": "fireflies", "star": "stars", "starry": "stars", "sparkles": "dust",
               "petal": "petals", "sakura": "petals", "leaves": "petals", "ember": "embers",
               "sparks": "embers", "bubble": "bubbles", "rainy": "rain", "snowy": "snow"}
    key = aliases.get(key, key)
    return key if key in AMBIENT_EFFECTS else "none"


class ParticleField:
    """Ambient particles for one clip."""

    def __init__(self, effect: str, width: int, height: int, seed: int = 0, density: float = 1.0):
        self.effect = normalize_effect(effect)
        self.width, self.height = width, height
        self.spec = EFFECTS.get(self.effect)
        if self.spec is None:
            self.n = 0
            return
        rng = np.random.default_rng(seed)
        self.n = max(1, int(self.spec.count * density * width * height / 1_000_000))
        self.x0 = rng.uniform(0, width, self.n)
        self.y0 = rng.uniform(0, height, self.n)
        self.radius = rng.uniform(*self.spec.size, self.n) * width
        speed = rng.uniform(0.6, 1.4, self.n)
        unit = width / 1000.0
        self.vx = self.spec.velocity[0] * unit * speed
        self.vy = self.spec.velocity[1] * unit * speed
        self.phase = rng.uniform(0, 2 * np.pi, self.n)
        self.freq = rng.uniform(0.2, 0.7, self.n)
        self.angle = rng.uniform(0, 180, self.n)

    @property
    def enabled(self) -> bool:
        return self.n > 0

    def positions(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        s = self.spec
        sway = s.sway * self.width * np.sin(2 * np.pi * self.freq * t + self.phase)
        margin = float(self.radius.max()) * 4 + 2
        x = np.mod(self.x0 + self.vx * t + sway + margin, self.width + 2 * margin) - margin
        y = np.mod(self.y0 + self.vy * t + margin, self.height + 2 * margin) - margin
        return x, y

    def brightness(self, t: float) -> np.ndarray:
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * self.freq * 1.7 * t + self.phase)
        return self.spec.opacity * (1.0 - self.spec.pulse + self.spec.pulse * pulse)

    def apply(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Draw particles onto a float32 (H, W, 3) frame in [0, 1] (returns new array)."""
        if not self.enabled:
            return frame
        s = self.spec
        x, y = self.positions(t)
        level = self.brightness(t)
        layer = np.zeros(frame.shape[:2], dtype=np.float32)
        for i in range(self.n):
            cx, cy, r = int(round(x[i])), int(round(y[i])), max(1, int(round(self.radius[i])))
            if self.effect == "rain":
                length = int(r * 18)
                dx = int(round(length * s.velocity[0] / max(1.0, abs(s.velocity[1]))))
                cv2.line(layer, (cx, cy), (cx + dx, cy + length), float(level[i]), max(1, r), cv2.LINE_AA)
            elif self.effect == "petals":
                axes = (r, max(1, int(r * 0.55)))
                angle = float(self.angle[i] + 60 * t * (1 if i % 2 else -1))
                cv2.ellipse(layer, (cx, cy), axes, angle, 0, 360, float(level[i]), -1, cv2.LINE_AA)
            elif self.effect == "bubbles":
                cv2.circle(layer, (cx, cy), r, float(level[i]), max(1, r // 4), cv2.LINE_AA)
            else:
                cv2.circle(layer, (cx, cy), r, float(level[i]), -1, cv2.LINE_AA)

        color = np.array(s.color, dtype=np.float32)
        if s.additive:
            glow = layer
            if s.glow > 0:
                halo = cv2.GaussianBlur(layer, (0, 0), s.glow * self.width)
                glow = np.maximum(layer, halo * 2.2)
            return np.clip(frame + glow[..., None] * color, 0.0, 1.0)
        if s.glow > 0:
            layer = cv2.GaussianBlur(layer, (0, 0), s.glow * self.width)
        a = np.clip(layer, 0.0, 1.0)[..., None]
        return frame * (1.0 - a) + color * a

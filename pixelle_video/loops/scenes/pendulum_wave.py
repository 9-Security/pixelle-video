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

"""Pendulum wave: N pendulums, pendulum i swings (base + i) times per period, so they realign exactly."""

import colorsys

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, register


@register
class PendulumWave(LoopScene):
    info = SceneInfo(
        name="pendulum_wave",
        title_zh="擺錘波",
        title_en="Pendulum Wave",
        description="15 pendulums drift through snake and wave patterns, then realign perfectly",
        period=60,
    )

    count = 15
    base_cycles = 51  # pendulum i completes base_cycles + i swings per period

    def setup(self) -> None:
        w, h, s = self.width, self.height, self.s
        bg = draw.vertical_gradient(w, h, draw.hex_bgr("#0f1626"), draw.hex_bgr("#1d2b45"))
        self.cx = w / 2
        self.pivot_y = h * 0.12
        # Frame bar
        draw.line(bg, (self.cx - 260 * s, self.pivot_y), (self.cx + 260 * s, self.pivot_y),
                  draw.hex_bgr("#8a94a8"), int(8 * s))
        self.background = draw.vignette(bg, 0.4)
        # Longer pendulums swing slower: L ~ 1 / f^2, mapped to screen lengths
        freqs = np.array([self.base_cycles + i for i in range(self.count)], float)
        lengths = 1.0 / freqs ** 2
        lengths = lengths / lengths.max()
        self.lengths = (h * 0.30 + (lengths - lengths.min()) / (lengths.max() - lengths.min()) * h * 0.48)
        self.amplitude = 0.42  # radians
        self.colors = []
        for i in range(self.count):
            r, g, b = colorsys.hls_to_rgb(0.55 - 0.45 * i / (self.count - 1), 0.62, 0.65)
            self.colors.append((int(b * 255), int(g * 255), int(r * 255)))

    def _angle(self, t: float, i: int) -> float:
        return self.amplitude * np.cos(2 * np.pi * self.cycles(t, self.base_cycles + i))

    def render(self, t: float) -> np.ndarray:
        img = self.background.copy()
        s = self.s
        string = draw.hex_bgr("#6b7690")
        bobs = []
        # Pivots spread slightly in depth: draw back to front (longest first)
        for i in range(self.count):
            depth = i / (self.count - 1)
            pivot = (self.cx + (depth - 0.5) * 22 * s, self.pivot_y + depth * 4 * s)
            a = self._angle(t, i)
            length = self.lengths[i]
            bob = (pivot[0] + np.sin(a) * length, pivot[1] + np.cos(a) * length)
            draw.line(img, pivot, bob, string, max(1, int(round(1.6 * s))))
            bobs.append((bob, i))
        for bob, i in bobs:
            r = (18 + 6 * (1 - i / (self.count - 1))) * s
            draw.glow(img, bob, r * 2.2, self.colors[i], 0.35)
            draw.circle(img, bob, r, self.colors[i])
            draw.circle(img, (bob[0] - r * 0.35, bob[1] - r * 0.35), r * 0.3, draw.mix(self.colors[i], (255, 255, 255), 0.6))
        return img

    def sound_design(self) -> SoundDesign:
        # Soft room tone plus a chime when all pendulums line up (start) and at the halfway split
        return SoundDesign(
            layers=[NoiseLayer("brown", 0.035, (30.0, 600.0)), NoiseLayer("pink", 0.004, (2000.0, 9000.0))],
            events=[
                SoundEvent(0.0, "chime", 0.35, 1.0),
                SoundEvent(self.period / 2, "chime", 0.22, 0.75),
            ],
        )

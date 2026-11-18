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
Endless dominoes: the camera tracks the falling front, so the line never ends.

Periodicity: dominoes are spaced d apart and the wave moves at v, so the view
repeats every m*d/v seconds (m = colour pattern length); the background hills
are sums of sines whose wavelength divides the distance they scroll per period.
"""

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, ease_in_out, register

PALETTE = ["#e76f51", "#f4a261", "#e9c46a", "#2a9d8f", "#457b9d"]


@register
class DominoForever(LoopScene):
    info = SceneInfo(
        name="domino_forever",
        title_zh="永遠倒不完的骨牌",
        title_en="Endless Dominoes",
        description="A domino run that never reaches its last piece; a soft clack every step",
        period=30,
        kind="suspense",
    )

    step_seconds = 0.4  # time between two dominoes falling
    fall_seconds = 1.6  # time for one domino to settle

    def setup(self) -> None:
        w, h, s = self.width, self.height, self.s
        if (self.period / self.step_seconds) % len(PALETTE):
            raise ValueError("period must hold a whole number of colour patterns")
        self.d = 104 * s  # spacing
        self.dw = 32 * s  # thickness
        self.dh = 230 * s  # height
        self.v = self.d / self.step_seconds
        self.ground = h * 0.78
        self.front_x = w * 0.62
        self.lie_angle = np.arccos(self.dw / self.d)  # resting angle on the next piece
        self.contact_angle = np.arcsin(min(0.99, (self.d - self.dw) / self.dh))
        # Fraction of the fall at which a piece hits the next one (for the clack timing)
        xs = np.linspace(0, 1, 2001)
        reached = np.nonzero(self.lie_angle * np.array([ease_in_out(x) for x in xs]) >= self.contact_angle)[0]
        self.contact_time = float(xs[reached[0]] * self.fall_seconds) if len(reached) else self.fall_seconds * 0.5
        sky = draw.vertical_gradient(w, h, draw.hex_bgr("#f7d9c4"), draw.hex_bgr("#fbeee0"))
        draw.glow(sky, (w * 0.25, h * 0.28), 150 * s, draw.hex_bgr("#ffc890"), 0.16)
        draw.circle(sky, (w * 0.25, h * 0.28), 70 * s, draw.hex_bgr("#fff1dc"))
        self.sky = sky
        # Hill layers: (speed factor, base y, colour, harmonics)
        self.hills = []
        for speed, base, color in ((0.22, 0.52, "#e8b9a0"), (0.45, 0.62, "#d49a83")):
            length = speed * self.v * self.period  # scrolls exactly one wavelength per period
            harmonics = [(int(n), float(self.rng.uniform(8, 26) * s), float(self.rng.uniform(0, 2 * np.pi)))
                         for n in self.rng.choice(np.arange(1, 7), size=3, replace=False)]
            self.hills.append((speed, base * h, draw.hex_bgr(color), length, harmonics))
        self.colors = [draw.hex_bgr(c) for c in PALETTE]

    def _hill_points(self, t: float, speed: float, base: float, length: float, harmonics) -> list:
        shift = (speed * self.v * t) % length
        xs = np.linspace(0, self.width, 97)
        ys = np.full_like(xs, base)
        for n, amp, ph in harmonics:
            ys += amp * np.sin(2 * np.pi * n * (xs + shift) / length + ph)
        return [(0, self.height)] + list(zip(xs, ys)) + [(self.width, self.height)]

    def _angle(self, age: float) -> float:
        if age <= 0:
            return 0.0
        return self.lie_angle * ease_in_out(min(1.0, age / self.fall_seconds))

    def render(self, t: float) -> np.ndarray:
        img = self.sky.copy()
        for speed, base, color, length, harmonics in self.hills:
            draw.polygon(img, self._hill_points(t, speed, base, length, harmonics), color)
        draw.polygon(img, [(0, self.ground), (self.width, self.ground), (self.width, self.height), (0, self.height)],
                     draw.hex_bgr("#a8755a"))
        draw.line(img, (0, self.ground), (self.width, self.ground), draw.hex_bgr("#8a5c45"), max(1, int(3 * self.s)))

        p = t % self.period
        cam = self.v * p - self.front_x  # world x at screen left
        k0 = int(np.floor((cam - self.dh) / self.d))
        k1 = int(np.ceil((cam + self.width + self.dh) / self.d))
        for k in range(k0, k1 + 1):
            x = k * self.d - cam
            age = p - k * self.step_seconds
            angle = self._angle(age)
            pivot = (x + self.dw / 2, self.ground)
            center = (x, self.ground - self.dh / 2)
            corners = draw.rotated_rect(center, self.dw, self.dh, angle, pivot)
            color = self.colors[k % len(self.colors)]
            # Shadow stretches as the piece leans
            sx = pivot[0] + np.sin(angle) * self.dh * 0.5
            draw.soft_shadow(img, (sx + 10 * self.s, self.ground + 6 * self.s),
                             (self.dw * 0.8 + np.sin(angle) * self.dh * 0.5, 6 * self.s), 0.25, 5 * self.s)
            draw.polygon(img, corners, color)
            # Lit edge on the left face
            edge = draw.rotated_rect((x - self.dw * 0.3, self.ground - self.dh / 2), self.dw * 0.25, self.dh * 0.96,
                                     angle, pivot)
            draw.polygon(img, edge, draw.mix(color, (255, 255, 255), 0.35))
            # Two pips
            for frac in (0.3, 0.7):
                c = draw.rotated_rect((x, self.ground - self.dh * frac), 1, 1, angle, pivot).mean(axis=0)
                draw.circle(img, c, 4.5 * self.s, (250, 250, 250))
        return draw.vignette(img, 0.25)

    def sound_design(self) -> SoundDesign:
        n = int(round(self.period / self.step_seconds))
        rng = np.random.default_rng(self.seed + 1)
        events = [SoundEvent(k * self.step_seconds + self.contact_time, "clack", float(rng.uniform(0.18, 0.26)),
                             float(rng.uniform(0.9, 1.1)), float(rng.uniform(-0.15, 0.25))) for k in range(n)]
        return SoundDesign(layers=[NoiseLayer("brown", 0.03, (30.0, 500.0))], events=events)

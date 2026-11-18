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

"""Rain drops on a calm pond: expanding ripples with a matching drip sound for every drop."""

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, register


@register
class RainRipples(LoopScene):
    info = SceneInfo(
        name="rain_ripples",
        title_zh="雨落池塘",
        title_en="Rain on a Pond",
        description="Drops fall on dark water; every ripple has its own soft drip sound",
        period=40,
    )

    drops_per_second = 1.3
    ripple_life = 4.5  # seconds
    perspective = 0.42  # vertical squash of ripples (oblique view)

    def setup(self) -> None:
        w, h, s = self.width, self.height, self.s
        bg = draw.vertical_gradient(w, h, draw.hex_bgr("#0b2a2f"), draw.hex_bgr("#061719"))
        # Faint horizontal reflection bands
        noise = draw.periodic_noise_texture(w, h, self.rng, cells=(3, 12), octaves=2)
        bg = np.clip(bg.astype(np.float32) * (0.9 + 0.2 * noise[..., None]), 0, 255).astype(np.uint8)
        self.background = draw.vignette(bg, 0.45)
        n = int(self.period * self.drops_per_second)
        self.drops = []
        for _ in range(n):
            t0 = float(self.rng.uniform(0, self.period))
            x = float(self.rng.uniform(0.06, 0.94) * w)
            y = float(self.rng.uniform(0.12, 0.92) * h)
            size = float(self.rng.uniform(0.7, 1.3))
            self.drops.append((t0, x, y, size))
        self.drops.sort()
        # Floating leaves drift on slow closed orbits (integer cycles per period)
        self.leaves = []
        for _ in range(3):
            self.leaves.append({
                "c": (float(self.rng.uniform(0.2, 0.8) * w), float(self.rng.uniform(0.3, 0.75) * h)),
                "r": (float(self.rng.uniform(40, 110) * s), float(self.rng.uniform(15, 40) * s)),
                "off": float(self.rng.random()),
                "rot": float(self.rng.uniform(0, 360)),
                "size": float(self.rng.uniform(55, 80) * s),
            })
        self.ring_color = draw.hex_bgr("#9fd3d6")

    def render(self, t: float) -> np.ndarray:
        img = self.background.copy()
        layer = img.copy()
        s = self.s
        p = t % self.period
        for t0, x, y, size in self.drops:
            age = (p - t0) % self.period  # wraps: drops near the end ripple on into the start
            if age >= self.ripple_life:
                continue
            k = age / self.ripple_life
            fade = (1 - k) ** 1.6
            for j in range(3):
                r = (12 + age * 95 - j * 22) * s * size
                if r <= 0:
                    continue
                shade = draw.mix(img[int(min(y, self.height - 1)), int(min(x, self.width - 1))].tolist(),
                                 self.ring_color, fade * (1 - j * 0.28))
                draw.ellipse(layer, (x, y), (r, r * self.perspective), 0, shade, max(1, int(round(2.2 * s))))
            if age < 0.12:  # splash point
                draw.circle(layer, (x, y), 4 * s * size, self.ring_color)
        draw.blend_layer(img, layer, 0.85)
        # Leaves on top
        for leaf in self.leaves:
            a = 2 * np.pi * (self.phase(t) + leaf["off"])
            cx = leaf["c"][0] + np.cos(a) * leaf["r"][0]
            cy = leaf["c"][1] + np.sin(a) * leaf["r"][1]
            ang = leaf["rot"] + 360 * self.phase(t)  # one slow turn per period
            size = leaf["size"]
            draw.soft_shadow(img, (cx + 6 * s, cy + 8 * s), (size, size * 0.45), 0.35, 6 * s)
            draw.ellipse(img, (cx, cy), (size, size * 0.42), ang, draw.hex_bgr("#c9803a"))
            rad = np.radians(ang)
            dx, dy = np.cos(rad) * size * 0.9, np.sin(rad) * size * 0.9
            draw.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), draw.hex_bgr("#8a4f1f"), max(1, int(2 * s)))
        return img

    def sound_design(self) -> SoundDesign:
        events = []
        for t0, x, _y, size in self.drops:
            events.append(SoundEvent(t0, "drip", 0.16 * size, pitch=1.25 - 0.45 * (size - 0.7),
                                     pan=(x / self.width) * 1.4 - 0.7))
        return SoundDesign(
            layers=[NoiseLayer("pink", 0.03, (300.0, 9000.0)), NoiseLayer("brown", 0.03, (30.0, 400.0))],
            events=events,
        )

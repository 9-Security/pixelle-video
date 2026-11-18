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

"""A flat-style cat laps water from a bowl, tail swaying, ripples on every lap."""

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, register, segment

FUR = "#e9a35c"
FUR_DARK = "#c97a35"
FUR_LIGHT = "#f6d2a4"


@register
class CatDrinking(LoopScene):
    info = SceneInfo(
        name="cat_drinking",
        title_zh="貓咪喝水",
        title_en="Cat Drinking Water",
        description="An orange cat laps water, tail swaying, ripples on every lap",
        period=20,
    )

    laps = 36  # tongue laps per period
    ripple_life = 0.9

    def setup(self) -> None:
        w, h, P, s = self.width, self.height, self.P, self.s
        bg = draw.vertical_gradient(w, h, draw.hex_bgr("#f3e3cf"), draw.hex_bgr("#ead2b6"))
        # Window light on the wall
        draw.polygon(bg, [P(300, 120), P(700, 120), P(620, 520), P(220, 520)], draw.hex_bgr("#f8eedf"))
        draw.line(bg, P(460, 120), P(420, 520), draw.hex_bgr("#ead2b6"), int(10 * s))
        draw.line(bg, P(260, 320), P(660, 320), draw.hex_bgr("#ead2b6"), int(10 * s))
        # Floor
        draw.polygon(bg, [P(-200, 820), P(2200, 820), P(2200, 1300), P(-200, 1300)], draw.hex_bgr("#b98a63"))
        for x in range(-200, 2200, 240):
            draw.line(bg, P(x, 820), P(x - 120, 1300), draw.hex_bgr("#a77a55"), int(3 * s))
        draw.soft_shadow(bg, P(900, 830), (330 * s, 26 * s), 0.3, 14 * s)
        draw.soft_shadow(bg, P(1250, 828), (160 * s, 16 * s), 0.3, 10 * s)
        self.background = draw.vignette(bg, 0.3)
        self.twitch_at = float(self.rng.uniform(0.25, 0.75))

    def _bob(self, t: float) -> float:
        x = self.cycles(t, self.laps)
        return 4 * np.sin(2 * np.pi * x)

    def render(self, t: float) -> np.ndarray:
        img = self.background.copy()
        P, s = self.P, self.s
        fur, fur_dark, fur_light = draw.hex_bgr(FUR), draw.hex_bgr(FUR_DARK), draw.hex_bgr(FUR_LIGHT)
        phase = self.phase(t)

        # Tail: two slow sways per period
        sway = np.sin(2 * np.pi * self.cycles(t, 2))
        tail = []
        for i in range(24):
            u = i / 23
            x = 640 - 160 * u + 40 * sway * u ** 2
            y = 790 - 230 * u ** 1.4 + 20 * np.sin(np.pi * u) - 30 * sway * u ** 3
            tail.append(P(x, y))
        draw.polyline(img, tail, fur_dark, 34 * s)
        draw.polyline(img, tail[:-3], fur, 26 * s)

        # Body and haunch
        draw.ellipse(img, P(860, 715), (250 * s, 135 * s), -6, fur)
        draw.ellipse(img, P(700, 740), (120 * s, 95 * s), 0, fur)
        for k in range(3):  # stripes
            draw.ellipse(img, P(760 + k * 95, 640), (24 * s, 70 * s), 12, fur_dark, -1, 200, 340)
        draw.ellipse(img, P(980, 790), (110 * s, 40 * s), 0, fur_light)  # chest

        bob = self._bob(t)
        hc = P(1095, 668 + bob)
        # Neck
        draw.polygon(img, [P(960, 620), P(1060, 620 + bob), P(1110, 720 + bob), P(990, 780)], fur)
        # Front paws
        draw.ellipse(img, P(1060, 815), (58 * s, 20 * s), 0, fur_light)
        draw.ellipse(img, P(1140, 818), (52 * s, 18 * s), 0, fur_light)

        # Bowl (behind the head's muzzle)
        draw.polygon(img, [P(1110, 752), P(1390, 752), P(1360, 828), P(1140, 828)], draw.hex_bgr("#4f86c6"))
        draw.ellipse(img, P(1250, 752), (140 * s, 22 * s), 0, draw.hex_bgr("#3b6ea8"))
        draw.ellipse(img, P(1250, 754), (124 * s, 16 * s), 0, draw.hex_bgr("#a9dcf2"))

        # Ripples from each lap (wrap within the period)
        lap_seconds = self.period / self.laps
        p = t % self.period
        layer = img.copy()
        tongue_x = 1182
        for j in range(int(self.ripple_life / lap_seconds) + 2):
            age = ((p - lap_seconds * 0.25) % lap_seconds) + j * lap_seconds  # tongue touches at 1/4 lap
            if age > self.ripple_life:
                continue
            k = age / self.ripple_life
            draw.ellipse(layer, P(tongue_x, 754), (min(118, 12 + 120 * k) * s, min(15, 3 + 14 * k) * s), 0,
                         draw.hex_bgr("#e8f7ff"), max(1, int(2 * s)))
        draw.blend_layer(img, layer, 0.7)

        # Head
        draw.circle(img, hc, 92 * s, fur)
        ear_twitch = np.sin(np.pi * segment(phase, self.twitch_at, self.twitch_at + 0.03)) * 8
        for ex, tilt in ((-40, -18 + ear_twitch), (35, 10)):
            base = hc + np.array([ex * s, -60 * s])
            rad = np.radians(tilt)
            tip = base + np.array([np.sin(rad), -np.cos(rad)]) * 75 * s
            draw.polygon(img, [base + [-38 * s, 12 * s], tip, base + [38 * s, 12 * s]], fur)
            inner = base + (tip - base) * 0.55
            draw.polygon(img, [base + [-18 * s, 4 * s], inner, base + [18 * s, 4 * s]], draw.hex_bgr("#f2b8a2"))
        for k in range(2):
            draw.ellipse(img, hc + np.array([(-10 + k * 26) * s, -70 * s]), (6 * s, 20 * s), 0, fur_dark)
        # Muzzle, closed eye, nose, whiskers
        muzzle = hc + np.array([60 * s, 40 * s])
        draw.ellipse(img, muzzle, (46 * s, 32 * s), 15, fur_light)
        draw.ellipse(img, hc + np.array([42 * s, -6 * s]), (18 * s, 9 * s), 15, draw.hex_bgr("#5a3a22"),
                     max(1, int(4 * s)), 10, 170)
        draw.ellipse(img, muzzle + [30 * s, -12 * s], (9 * s, 6 * s), 15, draw.hex_bgr("#d9837a"))
        for dy, ang in ((-4, -8), (6, 4), (16, 16)):
            a = np.radians(ang)
            start = muzzle + [20 * s, dy * s]
            draw.line(img, start, start + np.array([np.cos(a), np.sin(a)]) * 85 * s, draw.hex_bgr("#fff6ea"),
                      max(1, int(2 * s)))

        # Tongue: dips into the water once per lap
        x = self.cycles(t, self.laps)
        reach = max(0.0, np.sin(2 * np.pi * x)) ** 0.8
        mouth = muzzle + [18 * s, 26 * s]
        tongue_tip = P(tongue_x, 700 + bob + (754 - 700 - bob) * reach)
        if reach > 0.05:
            draw.line(img, mouth, tongue_tip, draw.hex_bgr("#e57f8b"), max(1, int(14 * s)))
        return img

    def sound_design(self) -> SoundDesign:
        lap_seconds = self.period / self.laps
        rng = np.random.default_rng(self.seed + 3)
        events = [SoundEvent(i * lap_seconds + lap_seconds * 0.25, "lap", float(rng.uniform(0.12, 0.2)),
                             float(rng.uniform(0.9, 1.1)), 0.2) for i in range(self.laps)]
        return SoundDesign(layers=[NoiseLayer("brown", 0.03, (30.0, 500.0)),
                                   NoiseLayer("pink", 0.003, (1500.0, 6000.0))], events=events)

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
Suspense loop: a cat paw nudges a mug towards the table edge in ever smaller
steps; it teeters, the paw winds up for the final push... and the view blinks
shut and starts again. The answer never comes.

The seam is an intentional blink: the frame is black at both ends of the
period, so the loop is still continuous.
"""

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, ease_in_out, register, segment

TABLE_EDGE = 1300.0
TABLE_Y = 700.0
MUG_W, MUG_H = 150.0, 165.0
# (start, end) phase of each push and the mug centre x it ends at: steps shrink like Zeno's paradox
PUSHES = [((0.10, 0.20), 1010.0), ((0.28, 0.37), 1130.0), ((0.45, 0.53), 1195.0), ((0.61, 0.68), 1238.0)]
START_X = 880.0
TEETER = (0.70, 0.90)
WIND_UP = (0.80, 0.93)
BLINK_IN, BLINK_OUT = 0.05, 0.93


@register
class CupOnEdge(LoopScene):
    info = SceneInfo(
        name="cup_on_edge",
        title_zh="杯子到底會不會掉",
        title_en="Will It Fall?",
        description="A paw pushes a mug to the very edge... and the loop restarts before you find out",
        period=16,
        kind="suspense",
    )

    def setup(self) -> None:
        w, h, P, s = self.width, self.height, self.P, self.s
        bg = draw.vertical_gradient(w, h, draw.hex_bgr("#d7e3ea"), draw.hex_bgr("#c3d3dd"))
        # Far floor beyond the table edge
        draw.polygon(bg, [P(TABLE_EDGE, 930), P(2300, 930), P(2300, 1300), P(TABLE_EDGE, 1300)], draw.hex_bgr("#9fb3c0"))
        # Table top and front edge
        draw.polygon(bg, [P(-300, TABLE_Y), P(TABLE_EDGE, TABLE_Y), P(TABLE_EDGE, TABLE_Y + 46), P(-300, TABLE_Y + 46)],
                     draw.hex_bgr("#8c5a3c"))
        draw.polygon(bg, [P(-300, TABLE_Y + 46), P(TABLE_EDGE, TABLE_Y + 46), P(TABLE_EDGE - 40, 1300),
                          P(-300, 1300)], draw.hex_bgr("#6e4429"))
        draw.line(bg, P(-300, TABLE_Y), P(TABLE_EDGE, TABLE_Y), draw.hex_bgr("#a8714f"), int(4 * s))
        self.background = draw.vignette(bg, 0.35)

    # -- timeline ----------------------------------------------------------
    def _mug_x(self, p: float) -> float:
        x = START_X
        for (a, b), target in PUSHES:
            # the mug slides during the second half of each push (after the paw makes contact)
            x = x + (target - x) * ease_in_out(segment(p, a + (b - a) * 0.35, b))
        return x

    def _paw_x(self, p: float) -> float:
        rest = START_X - MUG_W / 2 - 230
        contact_gap = 38.0
        x = rest
        prev_x = START_X
        for (a, b), target in PUSHES:
            mid = a + (b - a) * 0.35
            touch = prev_x - MUG_W / 2 - contact_gap
            if a <= p < mid:
                return rest + (touch - rest) * ease_in_out(segment(p, a, mid))
            if mid <= p < b:
                return self._mug_x(p) - MUG_W / 2 - contact_gap
            if b <= p < b + 0.05:
                return target - MUG_W / 2 - contact_gap + (rest - (target - MUG_W / 2 - contact_gap)) * \
                    ease_in_out(segment(p, b, b + 0.05))
            prev_x = target
        if p >= WIND_UP[0]:
            touch = prev_x - MUG_W / 2 - contact_gap
            k = segment(p, WIND_UP[0], WIND_UP[1])
            # pull back slowly, then start the final lunge just before the blink
            return rest - 60 * ease_in_out(min(1.0, k / 0.7)) + (touch - rest + 60) * max(0.0, (k - 0.7) / 0.3) ** 2
        return x

    def _tilt(self, p: float) -> float:
        k = segment(p, *TEETER)
        if k <= 0 or k >= 1:
            return 0.0
        return np.radians(3.5) * np.sin(np.pi * k) * np.sin(2 * np.pi * 3 * k) ** 2

    # -- drawing -----------------------------------------------------------
    def _draw_mug(self, img, cx: float, tilt: float) -> None:
        P, s = self.P, self.s
        pivot = P(TABLE_EDGE, TABLE_Y)
        center = P(cx, TABLE_Y - MUG_H / 2)
        body = draw.rotated_rect(center, MUG_W * s, MUG_H * s, tilt, pivot)
        handle_c = draw.rotated_rect(P(cx + MUG_W / 2 + 18, TABLE_Y - MUG_H * 0.55), 1, 1, tilt, pivot).mean(axis=0)
        draw.soft_shadow(img, P(min(cx, TABLE_EDGE - 40), TABLE_Y + 4), (MUG_W * 0.55 * s, 8 * s), 0.35, 6 * s)
        draw.ellipse(img, handle_c, (34 * s, 46 * s), np.degrees(tilt), draw.hex_bgr("#e05d5d"), int(16 * s))
        draw.polygon(img, body, draw.hex_bgr("#e05d5d"))
        rim = draw.rotated_rect(P(cx, TABLE_Y - MUG_H + 8), MUG_W * s, 16 * s, tilt, pivot)
        draw.polygon(img, rim, draw.hex_bgr("#f07a7a"))
        heart = draw.rotated_rect(P(cx, TABLE_Y - MUG_H * 0.5), 1, 1, tilt, pivot).mean(axis=0)
        draw.circle(img, heart, 18 * s, draw.hex_bgr("#fbe3e3"))

    def _draw_paw(self, img, x: float) -> None:
        P, s = self.P, self.s
        y = TABLE_Y - 52
        grey, light = draw.hex_bgr("#7d7f86"), draw.hex_bgr("#c9cbd1")
        dark = draw.hex_bgr("#5f6168")
        shoulder = P(x - 520, y - 330)
        wrist = P(x - 40, y)
        draw.soft_shadow(img, P(x - 60, TABLE_Y + 2), (110 * s, 8 * s), 0.3, 6 * s)
        draw.line(img, shoulder + [0, 10 * s], wrist + [0, 10 * s], dark, int(130 * s))
        draw.line(img, shoulder, wrist, grey, int(120 * s))
        # tabby stripes across the forearm
        axis = (wrist - shoulder) / np.linalg.norm(wrist - shoulder)
        normal = np.array([-axis[1], axis[0]])
        for k in (0.35, 0.55, 0.75):
            c = shoulder + (wrist - shoulder) * k
            draw.line(img, c - normal * 52 * s, c + normal * 30 * s, dark, int(14 * s))
        draw.ellipse(img, P(x, y + 4), (78 * s, 58 * s), 0, light)
        for dy in (-30, 0, 30):
            draw.ellipse(img, P(x + 58, y + 4 + dy), (17 * s, 15 * s), 0, draw.hex_bgr("#f0b7c0"))

    def render(self, t: float) -> np.ndarray:
        p = self.phase(t)
        img = self.background.copy()
        self._draw_mug(img, self._mug_x(p), self._tilt(p))
        self._draw_paw(img, self._paw_x(p))
        # Blink: eyelids close at the end and open at the start (black at both ends of the loop)
        if p < BLINK_IN:
            openness = ease_in_out(p / BLINK_IN)
        elif p > BLINK_OUT:
            openness = 1 - ease_in_out((p - BLINK_OUT) / (1 - BLINK_OUT))
        else:
            openness = 1.0
        if openness < 1.0:
            lid = int(round(self.height / 2 * (1 - openness)))
            if lid > 0:
                img[:lid] = 0
                img[self.height - lid:] = 0
            if openness < 0.999:
                img = (img.astype(np.float32) * (0.25 + 0.75 * openness)).astype(np.uint8)
        return img

    def sound_design(self) -> SoundDesign:
        T = self.period
        events = []
        for (a, b), _ in PUSHES:
            mid = a + (b - a) * 0.35
            events.append(SoundEvent(mid * T, "tap", 0.3, 1.3))
            for k in range(5):  # scrape while sliding
                events.append(SoundEvent((mid + (b - mid) * k / 5) * T, "lap", 0.12, 0.8))
        for k in range(4):  # heartbeat during the teeter
            events.append(SoundEvent((TEETER[0] + 0.02 + k * 0.05) * T, "thud", 0.4, 1.0))
            events.append(SoundEvent((TEETER[0] + 0.035 + k * 0.05) * T, "thud", 0.28, 0.9))
        return SoundDesign(layers=[NoiseLayer("brown", 0.03, (30.0, 500.0))], events=events)

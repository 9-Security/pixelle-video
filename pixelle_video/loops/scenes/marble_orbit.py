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

"""A finger flicks a glass marble around a wooden ring track, once per lap, forever (top view)."""

import numpy as np

from pixelle_video.loops import draw
from pixelle_video.loops.audio import NoiseLayer, SoundDesign, SoundEvent
from pixelle_video.loops.base import LoopScene, SceneInfo, ease_in_out, register, segment


@register
class MarbleOrbit(LoopScene):
    info = SceneInfo(
        name="marble_orbit",
        title_zh="手指推彈珠",
        title_en="Marble Flick",
        description="A finger gives a marble a gentle flick every lap; it rolls round and round",
        period=30,
    )

    laps = 6
    # Speed profile per lap: fast after the flick, slowing down (normalised to one lap)
    accel_a, accel_b = 1.6, 1.2

    def setup(self) -> None:
        w, h, s = self.width, self.height, self.s
        self.c = (w / 2, h / 2)
        self.r = 28 * s
        self.R = self.r * 11  # integer ratio: the marble's spin pattern also repeats every lap
        self.tip_r = 38 * s
        self.flick_angle = np.pi / 2  # bottom of the ring
        self.finger_dir = np.array([0.62, 0.78])  # tip -> knuckle direction (towards lower right)
        # Wooden board with grain
        grain = draw.periodic_noise_texture(w, h, self.rng, cells=(2, 46), octaves=3)
        base = np.array(draw.hex_bgr("#b07a4f"), np.float32)
        dark = np.array(draw.hex_bgr("#7d4f2e"), np.float32)
        board = base * (1 - grain[..., None] * 0.55) + dark * (grain[..., None] * 0.55)
        board = board.astype(np.uint8)
        groove_w = int(round(2 * self.r + 14 * s))
        draw.circle(board, self.c, self.R, draw.hex_bgr("#4a2c18"), groove_w)
        draw.circle(board, self.c, self.R - self.r * 0.2, draw.hex_bgr("#5e3a22"), max(1, int(groove_w * 0.55)))
        draw.circle(board, self.c, self.R + self.r + 7 * s, draw.hex_bgr("#c9956a"), max(1, int(3 * s)))
        draw.circle(board, self.c, self.R - self.r - 7 * s, draw.hex_bgr("#c9956a"), max(1, int(3 * s)))
        self.board = draw.vignette(board, 0.4)

    # -- motion ------------------------------------------------------------
    def _lap(self, t: float):
        x = self.cycles(t, self.laps)
        lap_index = int((t % self.period) * self.laps // self.period)
        return lap_index, x

    def _theta(self, t: float) -> float:
        _, x = self._lap(t)
        a, b = self.accel_a, self.accel_b
        frac = (a * x - b * x * x / 2) / (a - b / 2)
        return self.flick_angle + 2 * np.pi * frac

    def _ball(self, theta: float):
        return np.array([self.c[0] + self.R * np.cos(theta), self.c[1] + self.R * np.sin(theta)])

    def _tangent(self, theta: float):
        return np.array([-np.sin(theta), np.cos(theta)])

    def _tip(self, t: float):
        _, x = self._lap(t)
        ball0 = self._ball(self.flick_angle)
        tangent0 = self._tangent(self.flick_angle)
        radial0 = np.array([np.cos(self.flick_angle), np.sin(self.flick_angle)])
        contact = ball0 - tangent0 * (self.r + self.tip_r * 0.9)
        hover = contact + radial0 * 70 * self.s
        rest = contact + radial0 * 120 * self.s + tangent0 * -60 * self.s
        if x < 0.05:  # push: stay behind the marble
            th = self._theta(t)
            return self._ball(th) - self._tangent(th) * (self.r + self.tip_r * 0.9)
        if x < 0.3:
            th_end = self._theta(t - (x - 0.05) * self.period / self.laps)
            start = self._ball(th_end) - self._tangent(th_end) * (self.r + self.tip_r * 0.9)
            return start + (rest - start) * ease_in_out(segment(x, 0.05, 0.3))
        if x < 0.8:
            return rest
        if x < 0.95:
            return rest + (hover - rest) * ease_in_out(segment(x, 0.8, 0.95))
        return hover + (contact - hover) * ease_in_out(segment(x, 0.95, 1.0))

    # -- drawing -----------------------------------------------------------
    def render(self, t: float) -> np.ndarray:
        img = self.board.copy()
        s = self.s
        th = self._theta(t)
        ball = self._ball(th)
        draw.soft_shadow(img, ball + [8 * s, 10 * s], (self.r * 0.95, self.r * 0.8), 0.5, 7 * s)
        draw.circle(img, ball, self.r, draw.hex_bgr("#3f8fd0"))
        # Swirl rotates with the rolling distance (R / r is an integer, so it is periodic per lap)
        spin = (th - self.flick_angle) * self.R / self.r
        for k, col in ((0, "#9fd6ff"), (np.pi, "#1f5f99")):
            a = spin + k
            p0 = ball + np.array([np.cos(a), np.sin(a)]) * self.r * 0.55
            p1 = ball + np.array([np.cos(a + 1.2), np.sin(a + 1.2)]) * self.r * 0.6
            draw.line(img, p0, p1, draw.hex_bgr(col), max(1, int(round(5 * s))))
        draw.circle(img, ball, self.r, draw.hex_bgr("#9fd6ff"), max(1, int(round(2 * s))))
        draw.circle(img, ball + [-self.r * 0.35, -self.r * 0.4], self.r * 0.22, (255, 255, 255))

        tip = self._tip(t)
        knuckle = tip + self.finger_dir * 520 * s
        draw.soft_shadow(img, tip + [14 * s, 18 * s], (self.tip_r, self.tip_r), 0.35, 10 * s)
        skin, skin_dark = draw.hex_bgr("#f1c7a5"), draw.hex_bgr("#d9a684")
        draw.line(img, tip + [6 * s, 8 * s], knuckle + [6 * s, 8 * s], skin_dark, int(2 * self.tip_r + 4 * s))
        draw.line(img, tip, knuckle, skin, int(2 * self.tip_r))
        nail_c = tip + self.finger_dir * self.tip_r * 0.9
        ang = np.degrees(np.arctan2(self.finger_dir[1], self.finger_dir[0]))
        draw.ellipse(img, nail_c, (self.tip_r * 0.75, self.tip_r * 0.6), ang, draw.hex_bgr("#f7dcd0"))
        return img

    def sound_design(self) -> SoundDesign:
        a, b = self.accel_a, self.accel_b
        lap_seconds = self.period / self.laps

        def rolling(times: np.ndarray) -> np.ndarray:
            x = (times % lap_seconds) / lap_seconds
            speed = (a - b * x) / a
            return 0.25 + 0.75 * speed ** 1.5

        events = [SoundEvent(i * lap_seconds, "tap", 0.35, 1.0, 0.1) for i in range(self.laps)]
        return SoundDesign(
            layers=[NoiseLayer("brown", 0.05, (40.0, 900.0), envelope=rolling),
                    NoiseLayer("brown", 0.02, (30.0, 300.0))],
            events=events,
        )

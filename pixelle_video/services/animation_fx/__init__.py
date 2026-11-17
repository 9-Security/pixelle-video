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

"""2.5D animation effects: layer separation, parallax camera, ambient particles."""

from pixelle_video.services.animation_fx.particles import AMBIENT_EFFECTS, normalize_effect
from pixelle_video.services.animation_fx.renderer import ParallaxOptions, ParallaxRenderer
from pixelle_video.services.animation_fx.segmentation import (
    ForegroundSegmenter,
    SegmentationUnavailable,
)

__all__ = [
    "AMBIENT_EFFECTS",
    "ForegroundSegmenter",
    "ParallaxOptions",
    "ParallaxRenderer",
    "SegmentationUnavailable",
    "normalize_effect",
]

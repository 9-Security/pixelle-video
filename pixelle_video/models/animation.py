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
Animation data models.

These models describe the structured "animation script" produced by the LLM:
a fixed cast of characters (used as visual anchors for consistency across
scenes) and an ordered list of scenes with narration, visuals and camera motion.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CameraMotion(str, Enum):
    """Camera motion applied when animating a still image (Ken Burns style)."""

    STATIC = "static"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"

    @classmethod
    def parse(cls, value: Optional[str], default: "CameraMotion" = None) -> "CameraMotion":
        """Parse a loose string (LLM output) into a CameraMotion, never raising."""
        default = default or cls.ZOOM_IN
        if not value:
            return default
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "zoomin": cls.ZOOM_IN,
            "push_in": cls.ZOOM_IN,
            "dolly_in": cls.ZOOM_IN,
            "zoomout": cls.ZOOM_OUT,
            "pull_out": cls.ZOOM_OUT,
            "dolly_out": cls.ZOOM_OUT,
            "left": cls.PAN_LEFT,
            "right": cls.PAN_RIGHT,
            "up": cls.PAN_UP,
            "tilt_up": cls.PAN_UP,
            "down": cls.PAN_DOWN,
            "tilt_down": cls.PAN_DOWN,
            "none": cls.STATIC,
            "still": cls.STATIC,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return default


class AnimationCharacter(BaseModel):
    """A recurring character with a fixed visual description."""

    name: str = Field(description="Short unique character name")
    appearance: str = Field(
        description=(
            "Fixed English visual description reused in every scene: species/gender, age, "
            "body shape, hair, clothing, colors and distinctive features"
        )
    )


class AnimationScene(BaseModel):
    """A single scene (shot) of the animation."""

    narration: str = Field(description="Narration / voice-over text for this scene")
    visual: str = Field(
        description="English description of what is on screen: setting, action, composition, lighting"
    )
    characters: List[str] = Field(
        default_factory=list, description="Names of characters appearing in this scene"
    )
    motion: str = Field(
        default="",
        description="English description of movement in the shot (used by AI video models)",
    )
    camera: str = Field(
        default=CameraMotion.ZOOM_IN.value,
        description="Camera move: static, zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down",
    )

    @field_validator("camera", mode="before")
    @classmethod
    def _normalize_camera(cls, value):
        return CameraMotion.parse(value).value


class AnimationScript(BaseModel):
    """Complete animation script produced by the LLM."""

    title: str = Field(description="Short title of the animation")
    logline: str = Field(default="", description="One-sentence summary of the story")
    characters: List[AnimationCharacter] = Field(default_factory=list)
    scenes: List[AnimationScene] = Field(default_factory=list)

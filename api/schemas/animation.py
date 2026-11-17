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

"""Animation generation API schemas."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnimationGenerateRequest(BaseModel):
    """Request body for animation generation."""

    text: str = Field("", description="Story idea (generate) or full story (adapt); optional with `script`")
    script: Optional[Dict[str, Any]] = Field(
        None,
        description="Edited animation_script.json content (or its 'script' object). Skips script generation "
                    "and reuses these scenes, e.g. after fixing a scene the image model rejected",
    )
    mode: Literal["generate", "adapt"] = Field("generate", description="Script mode")
    title: Optional[str] = Field(None, description="Title; defaults to the AI-generated title")
    tone: Optional[str] = Field(None, description="Audience / tone hint")
    n_scenes: int = Field(6, ge=1, le=30, description="Target number of scenes")
    max_characters: int = Field(4, ge=0, le=8, description="Max recurring characters")
    min_narration_words: int = Field(10, ge=1, le=200)
    max_narration_words: int = Field(40, ge=1, le=300)

    art_style: str = Field("anime", description="Style preset key, see /api/animation/styles")
    custom_style: Optional[str] = Field(None, description="Free-text style prompt (overrides art_style)")

    motion_mode: Literal["parallax", "camera", "ai_video"] = Field(
        "parallax", description="parallax: 2.5D layers + particles; camera: zoom/pan only; ai_video: video model"
    )
    ambient_effect: str = Field(
        "auto", description="auto (LLM per scene), none, fireflies, stars, snow, rain, petals, dust, embers, bubbles"
    )
    parallax_strength: float = Field(0.6, ge=0.0, le=1.5, description="Extra character movement vs background")
    character_motion: bool = Field(True, description="Subtle character breathing/bob (parallax)")
    camera_motion: str = Field("auto", description="auto or zoom_in/zoom_out/pan_left/pan_right/pan_up/pan_down/static")
    camera_intensity: Optional[float] = Field(
        None, ge=0.0, le=0.5, description="Zoom/pan amount (default 0.12 parallax, 0.15 camera)"
    )
    media_workflow: str = Field(..., description="Image workflow (camera) or video workflow (ai_video)")
    api_video_params: Optional[Dict[str, Any]] = Field(None, description="Extra params for API video models")

    frame_template: str = Field("1080x1920/video_fullscreen.html", description="Subtitle template")
    template_params: Optional[Dict[str, Any]] = None
    video_fps: int = Field(30, ge=10, le=60)

    tts_inference_mode: Literal["local", "comfyui"] = "local"
    tts_voice: Optional[str] = Field("zh-CN-YunjianNeural", description="Edge-TTS voice (local mode)")
    tts_speed: Optional[float] = Field(1.1, ge=0.5, le=2.0)
    tts_workflow: Optional[str] = Field(None, description="TTS workflow (comfyui mode)")
    ref_audio: Optional[str] = None

    bgm_path: Optional[str] = None
    bgm_volume: float = Field(0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _text_or_script(self):
        if not self.text.strip() and not self.script:
            raise ValueError("Either 'text' or 'script' is required")
        return self

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "A little fox afraid of the dark sets out to bring the moon back",
            "n_scenes": 5,
            "art_style": "watercolor",
            "motion_mode": "parallax",
            "media_workflow": "api/cloudflare/flux-1-schnell",
            "frame_template": "1080x1920/video_fullscreen.html",
            "tts_voice": "zh-CN-XiaoxiaoNeural",
        }
    })


class AnimationGenerateResponse(BaseModel):
    success: bool = True
    message: str = "Success"
    video_url: str
    duration: float
    file_size: int
    title: str
    script_url: Optional[str] = Field(None, description="URL of animation_script.json")


class AnimationGenerateAsyncResponse(BaseModel):
    success: bool = True
    message: str = "Task created successfully"
    task_id: str

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
Animation script prompts and art style presets.

The animation pipeline asks the LLM for ONE structured script (characters +
scenes) instead of generating narration and image prompts independently.
Keeping a fixed character sheet and a fixed style prefix is what gives the
generated images visual consistency across scenes.
"""

from typing import Dict, Iterable, List, Optional

from pixelle_video.models.animation import AnimationCharacter, AnimationScene

# ----------------------------------------------------------------------------
# Art style presets
# ----------------------------------------------------------------------------

ANIMATION_STYLE_PRESETS: Dict[str, Dict[str, str]] = {
    "anime": {
        "name_zh": "日系動漫",
        "name_en": "Japanese Anime",
        "prompt": (
            "anime style illustration, cel shading, clean line art, vibrant colors, "
            "expressive characters, detailed background, studio quality animation still"
        ),
    },
    "3d_cartoon": {
        "name_zh": "3D 卡通",
        "name_en": "3D Cartoon",
        "prompt": (
            "3D animated movie style, stylized cartoon characters, soft global illumination, "
            "subsurface scattering, rich colors, cinematic lighting, high quality render"
        ),
    },
    "watercolor": {
        "name_zh": "水彩繪本",
        "name_en": "Watercolor Storybook",
        "prompt": (
            "children's picture book illustration, soft watercolor painting, gentle pastel colors, "
            "paper texture, warm and cozy atmosphere"
        ),
    },
    "chinese_ink": {
        "name_zh": "中國水墨",
        "name_en": "Chinese Ink Painting",
        "prompt": (
            "traditional Chinese ink wash painting, elegant brush strokes, rice paper texture, "
            "minimal colors with subtle accents, poetic atmosphere, generous negative space"
        ),
    },
    "pixel_art": {
        "name_zh": "像素風",
        "name_en": "Pixel Art",
        "prompt": (
            "16-bit pixel art, retro video game style, limited color palette, crisp pixels, "
            "detailed sprite characters"
        ),
    },
    "clay": {
        "name_zh": "黏土定格",
        "name_en": "Claymation",
        "prompt": (
            "claymation stop motion style, handcrafted plasticine characters and props, "
            "visible fingerprints texture, miniature set, soft studio lighting"
        ),
    },
    "flat_vector": {
        "name_zh": "扁平向量",
        "name_en": "Flat Vector",
        "prompt": (
            "flat vector illustration, minimal shapes, bold solid colors, clean geometric design, "
            "motion graphics explainer style"
        ),
    },
    "comic": {
        "name_zh": "美式漫畫",
        "name_en": "American Comic",
        "prompt": (
            "american comic book style, bold ink outlines, halftone shading, dynamic composition, "
            "saturated colors"
        ),
    },
}

DEFAULT_ANIMATION_STYLE = "anime"


def get_style_prompt(style: Optional[str], custom_style: Optional[str] = None) -> str:
    """Resolve the style prefix. A non-empty custom style always wins."""
    if custom_style and custom_style.strip():
        return custom_style.strip()
    preset = ANIMATION_STYLE_PRESETS.get(style or DEFAULT_ANIMATION_STYLE)
    if preset is None:
        preset = ANIMATION_STYLE_PRESETS[DEFAULT_ANIMATION_STYLE]
    return preset["prompt"]


# ----------------------------------------------------------------------------
# Script generation prompt
# ----------------------------------------------------------------------------

_COMMON_RULES = """# Output Rules
1. Language: "title", "logline" and every "narration" MUST be written in the same language as the user's input.
   "appearance", "visual" and "motion" MUST always be written in English (they are fed to image/video models).
2. Characters: define at most {max_characters} recurring characters. Each "appearance" must be a complete,
   self-contained English description (species or gender, age, body shape, hair, outfit, colors,
   distinctive features) so the same character can be redrawn identically in every scene.
   If the story has no characters (e.g. landscapes), return an empty list.
3. Scenes: each scene is one shot.
   - "narration": voice-over text, {min_words}~{max_words} words, natural spoken rhythm, no ending punctuation.
   - "visual": English, 20~50 words, describe setting, what characters are doing, composition, lighting.
     Refer to characters ONLY by their exact "name"; do NOT repeat their appearance here.
   - "characters": the list of character names visible in this shot (exact names from the character list).
   - "motion": English, 5~20 words, the movement happening in the shot (for AI video models).
   - "camera": one of static, zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down.
     Vary the camera across scenes to keep the animation dynamic.
4. Visual style is decided separately; do NOT mention art style, medium or rendering in "visual".
5. Content must be suitable for a general audience.
"""

GENERATE_ANIMATION_SCRIPT_PROMPT = """# Role
You are an animation director and screenwriter who turns ideas into short animated stories
with a clear beginning, development, climax and ending.

# Task
Write an animated short story based on the user's idea, split into exactly {n_scenes} scenes.

# User Idea
{idea}

# Target Audience / Tone
{tone}

""" + _COMMON_RULES

ADAPT_ANIMATION_SCRIPT_PROMPT = """# Role
You are an animation storyboard artist.

# Task
Turn the user's existing story into an animation storyboard.
- Split the story into scenes in its original order (about {n_scenes} scenes; follow the story's natural breaks).
- Keep the user's wording for "narration" as much as possible; only trim or split sentences
  so each narration fits {min_words}~{max_words} words. Do NOT invent new plot.

# User Story
{idea}

# Target Audience / Tone
{tone}

""" + _COMMON_RULES


def build_animation_script_prompt(
    idea: str,
    n_scenes: int = 6,
    mode: str = "generate",
    tone: Optional[str] = None,
    min_words: int = 10,
    max_words: int = 40,
    max_characters: int = 4,
) -> str:
    """
    Build the LLM prompt that produces an AnimationScript.

    Args:
        idea: Story idea (generate mode) or full story text (adapt mode)
        n_scenes: Target number of scenes
        mode: "generate" (write a new story) or "adapt" (storyboard the given story)
        tone: Optional audience / tone hint
        min_words: Min narration length
        max_words: Max narration length
        max_characters: Upper bound of recurring characters
    """
    if mode not in ("generate", "adapt"):
        raise ValueError(f"Unsupported animation script mode: {mode}")
    template = GENERATE_ANIMATION_SCRIPT_PROMPT if mode == "generate" else ADAPT_ANIMATION_SCRIPT_PROMPT
    return template.format(
        idea=idea.strip(),
        n_scenes=n_scenes,
        tone=(tone or "Warm, imaginative, suitable for all ages").strip(),
        min_words=min_words,
        max_words=max_words,
        max_characters=max_characters,
    )


# ----------------------------------------------------------------------------
# Per-scene media prompts
# ----------------------------------------------------------------------------

def _character_anchors(
    scene: AnimationScene, characters: Iterable[AnimationCharacter]
) -> List[str]:
    """Return 'Name: appearance' anchors for characters present in the scene."""
    by_name = {c.name.strip().lower(): c for c in characters}
    anchors = []
    for name in scene.characters:
        character = by_name.get(str(name).strip().lower())
        if character:
            anchors.append(f"{character.name} ({character.appearance.strip()})")
    return anchors


def build_scene_image_prompt(
    scene: AnimationScene,
    characters: Iterable[AnimationCharacter],
    style_prompt: str,
) -> str:
    """
    Compose the final text-to-image prompt for one scene.

    Order: style prefix -> scene visual -> character anchors. Putting the style
    first and repeating the full character description in every scene is the
    simplest model-agnostic way to keep the look consistent.
    """
    parts = [style_prompt.strip(), scene.visual.strip()]
    anchors = _character_anchors(scene, characters)
    if anchors:
        parts.append("Characters: " + "; ".join(anchors))
    parts.append("consistent character design, no text, no watermark")
    return ", ".join(p for p in parts if p)


def build_scene_video_prompt(
    scene: AnimationScene,
    characters: Iterable[AnimationCharacter],
    style_prompt: str,
) -> str:
    """Compose the text-to-video prompt: image prompt plus motion and camera hints."""
    base = build_scene_image_prompt(scene, characters, style_prompt)
    extras = []
    if scene.motion:
        extras.append(f"Motion: {scene.motion.strip()}")
    if scene.camera and scene.camera != "static":
        extras.append(f"Camera: {scene.camera.replace('_', ' ')}")
    extras.append("smooth animation")
    return base + ". " + ". ".join(extras)

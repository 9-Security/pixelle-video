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

import re
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
   Each character has TWO names that refer to the same character:
   - "display_name": the name spoken in narration, in the narration language (e.g. 皮皮 for Chinese input).
   - "name": a short ASCII romanization of display_name used only in English fields (e.g. Pipi).
   - "descriptor": a short English noun phrase WITHOUT any name (e.g. "a small red fox with a teal scarf").
   Narration MUST call each character only by its exact "display_name" (never "name", never a nickname).
   Never alter, shorten or merge a display_name with other words (e.g. do not blend it with the species).
   If the story has no characters (e.g. landscapes), return an empty list.
3. Scenes: each scene is one shot.
   - "narration": voice-over text, {min_words}~{max_words} words, natural spoken rhythm, no ending punctuation.
   - "visual": English, 20~50 words, describe setting, what characters are doing, composition, lighting.
     Refer to characters ONLY by their exact ASCII "name"; do NOT repeat their appearance here.
   - "characters": the ASCII "name" of every character visible in this shot.
   - "motion": English, 5~20 words, the movement happening in the shot (for AI video models).
   - "camera": one of static, zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down.
     Vary the camera across scenes to keep the animation dynamic.
   - "ambient": an atmospheric particle effect that fits the scene: one of none, fireflies, stars, snow,
     rain, petals, dust, embers, bubbles. Only use weather/particles consistent with the setting.
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

NO_TEXT_GUARD = "no text, no letters, no words, no captions, no signature, no watermark"


def _scene_characters(scene: AnimationScene, characters: Iterable[AnimationCharacter]) -> List[AnimationCharacter]:
    by_name = {c.name.strip().lower(): c for c in characters}
    found = []
    for name in scene.characters:
        character = by_name.get(str(name).strip().lower())
        if character and character not in found:
            found.append(character)
    return found


def _strip_names(text: str, characters: Iterable[AnimationCharacter]) -> str:
    """
    Replace character names with name-free visual labels in ONE pass.

    - A single alternation (longest names first) means text inserted for one
      character is never re-matched by another character's name.
    - Names match case-sensitively: the script prompt requires the exact name in
      English fields, so a character named "Moon" does not swallow "the full moon".
    """
    labels = {}
    for character in characters:
        for name in (character.name.strip(), character.spoken_name()):
            if name and name not in labels:
                labels[name] = character.visual_label()
    if not labels:
        return text
    alternation = "|".join(re.escape(n) for n in sorted(labels, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9])({alternation})('s)?(?![A-Za-z0-9])")
    return pattern.sub(lambda m: labels[m.group(1)] + (m.group(2) or ""), text)


def _character_anchors(
    scene: AnimationScene, characters: Iterable[AnimationCharacter]
) -> List[str]:
    """Full appearance of each character in the scene, without names."""
    return [c.appearance.strip() for c in _scene_characters(scene, characters) if c.appearance.strip()]


def composition_hint(width: Optional[int], height: Optional[int]) -> str:
    """
    Framing hint derived from the output aspect ratio.

    Some models (e.g. FLUX.1 schnell on Workers AI) only output square images
    that are later center-cropped, so the subject should stay centered.
    """
    if not width or not height:
        return ""
    ratio = width / float(height)
    if ratio < 0.85:
        return "vertical portrait composition, main subject centered in the middle of the frame"
    if ratio > 1.2:
        return "wide cinematic composition, main subject centered"
    return ""


def build_scene_image_prompt(
    scene: AnimationScene,
    characters: Iterable[AnimationCharacter],
    style_prompt: str,
    composition: str = "",
) -> str:
    """
    Compose the final text-to-image prompt for one scene.

    Order: style prefix -> scene visual -> character anchors. Putting the style
    first and repeating the full character description in every scene is the
    simplest model-agnostic way to keep the look consistent.
    """
    characters = list(characters)
    parts = [style_prompt.strip(), _strip_names(scene.visual.strip(), characters)]
    anchors = _character_anchors(scene, characters)
    if anchors:
        label = "Character" if len(anchors) == 1 else f"{len(anchors)} characters"
        parts.append(f"{label}: " + "; ".join(anchors))
    if composition:
        parts.append(composition)
    parts.append(f"consistent character design, {NO_TEXT_GUARD}")
    return ", ".join(p for p in parts if p)


def build_scene_video_prompt(
    scene: AnimationScene,
    characters: Iterable[AnimationCharacter],
    style_prompt: str,
    composition: str = "",
) -> str:
    """Compose the text-to-video prompt: image prompt plus motion and camera hints."""
    base = build_scene_image_prompt(scene, characters, style_prompt, composition)
    extras = []
    if scene.motion:
        extras.append(f"Motion: {_strip_names(scene.motion.strip(), list(characters))}")
    if scene.camera and scene.camera != "static":
        extras.append(f"Camera: {scene.camera.replace('_', ' ')}")
    extras.append("smooth animation")
    return base + ". " + ". ".join(extras)


PROOFREAD_PROMPT = """# Task
Proofread the narration lines of an animated story. Fix ONLY:
- character names that are misspelled, shortened, merged with other words or replaced by nicknames:
  each character must be called exactly by its official name below;
- obvious typos.
Do NOT rewrite, reorder, add or remove content, and keep each line's language, length and punctuation.
Return the same number of lines, in the same order.

# Official character names
{names}

# Narration lines (JSON array)
{lines}
"""


def build_proofread_prompt(narrations: List[str], display_names: List[str]) -> str:
    import json

    return PROOFREAD_PROMPT.format(
        names="\n".join(f"- {n}" for n in display_names) or "- (none)",
        lines=json.dumps(narrations, ensure_ascii=False, indent=1),
    )

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
Animation Generation Pipeline

Turns a story idea (or an existing story) into a narrated animated short.

Workflow:
1. LLM writes ONE structured AnimationScript: character sheet + scenes
   (narration, English visual, characters, motion, camera move).
2. Per-scene media prompts = art-style prefix + scene visual + full character
   descriptions, so characters stay visually consistent across scenes.
3. Per scene, one of two motion modes:
   - "camera"   : text-to-image, then animate the still with a camera move
                  (zoom / pan, Ken Burns). Works with any image workflow and is
                  the cheapest option.
   - "ai_video" : text-to-video workflow (e.g. WAN 2.x / API video model) using
                  a prompt enriched with motion and camera hints.
4. Narration TTS, subtitle template overlay, concatenation and optional BGM are
   reused from the standard pipeline.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger

from pixelle_video.models.animation import AnimationScene, AnimationScript, CameraMotion
from pixelle_video.models.progress import ProgressEvent
from pixelle_video.models.storyboard import StoryboardFrame
from pixelle_video.pipelines.linear import PipelineContext
from pixelle_video.pipelines.standard import StandardPipeline
from pixelle_video.prompts.animation_script import (
    build_animation_script_prompt,
    build_scene_image_prompt,
    build_scene_video_prompt,
    composition_hint,
    get_style_prompt,
)
from pixelle_video.services.animation_fx import ParallaxOptions, ParallaxRenderer, normalize_effect
from pixelle_video.services.video import VideoService
from pixelle_video.utils.os_util import get_task_frame_path
from pixelle_video.utils.template_util import get_template_type

MOTION_MODES = ("parallax", "camera", "ai_video")
DEFAULT_MOTION_MODE = "parallax"
SCRIPT_MODES = ("generate", "adapt")
DEFAULT_TEMPLATE = "1080x1920/video_fullscreen.html"
MAX_SCENES = 30
# Text baked into image templates moves with the camera, so keep it gentler
IMAGE_TEMPLATE_INTENSITY_FACTOR = 0.5


class SceneMediaError(RuntimeError):
    """
    A scene's image/video could not be generated.

    Carries what a person needs to fix it: the scene number, the exact prompt,
    whether the model's safety filter rejected it, and where the script was
    saved so it can be edited and passed back with `script=...`.
    """

    def __init__(self, scene_index: int, prompt: str, cause: Exception, script_path: str | None):
        self.scene_index = scene_index
        self.prompt = prompt
        self.script_path = script_path
        self.safety_blocked = "nsfw" in str(cause).lower()
        reason = (
            "rejected by the image model's safety filter (this can be a false positive)"
            if self.safety_blocked else f"failed: {cause}"
        )
        hint = (
            f" The script was saved to {script_path}; edit scene {scene_index + 1} "
            f"(its 'visual' text or the character 'appearance') and regenerate with script=<that file>."
            if script_path else ""
        )
        super().__init__(f"Scene {scene_index + 1} image generation {reason}. Prompt: {prompt[:400]!r}.{hint}")


def load_animation_script(source) -> AnimationScript:
    """
    Load a script from an AnimationScript, a dict, or a path to animation_script.json.

    Accepts both the full saved payload ({"script": {...}, ...}) and a bare script.
    """
    if isinstance(source, AnimationScript):
        return source.model_copy(deep=True)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Animation script file not found: {path}")
        source = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("script must be an AnimationScript, a dict or a path to animation_script.json")
    data = source.get("script", source)
    try:
        return AnimationScript.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid animation script: {e}") from e


class AnimationPipeline(StandardPipeline):
    """
    Animated short generator.

    Main parameters (kwargs):
        mode: "generate" (idea -> new story) | "adapt" (storyboard the given story)
        n_scenes: target scene count (1-30)
        tone: optional audience / tone hint for the script
        art_style: key of ANIMATION_STYLE_PRESETS (default "anime")
        custom_style: free-text style prompt, overrides art_style
        max_characters: max recurring characters (default 4)
        motion_mode: "parallax" (2.5D layers + particles, default) | "camera" | "ai_video"
        camera_motion: "auto" (use LLM choice) or a fixed CameraMotion value
        camera_intensity: 0.0-0.5 zoom/pan amount (default 0.15; 0.12 for parallax)
        ambient_effect: "auto" (LLM choice per scene), "none" or a fixed particle effect (parallax)
        parallax_strength: 0.0-1.5 extra character movement vs background (default 0.6)
        character_motion: subtle character breathing/bob in parallax mode (default True)
        media_workflow: image workflow (parallax/camera) or video workflow (ai_video)
        frame_template: subtitle template; video_* recommended
        plus all TTS / BGM / output params of StandardPipeline
    """

    # ==================== Validation & setup ====================

    async def setup_environment(self, ctx: PipelineContext):
        self._validate_params(ctx)
        await super().setup_environment(ctx)

    def _validate_params(self, ctx: PipelineContext):
        params = ctx.params
        if params.get("script") is not None:
            ctx.extras["animation_script"] = load_animation_script(params["script"])
            if not isinstance(params["script"], (str, Path)):
                params["script"] = "(inline script)"  # keep task metadata JSON-serializable
            else:
                params["script"] = str(params["script"])
            if not (ctx.input_text or "").strip():
                ctx.input_text = ctx.extras["animation_script"].title or "animation"
        elif not ctx.input_text or not ctx.input_text.strip():
            raise ValueError("Animation pipeline requires a non-empty story idea or story text")

        mode = params.get("mode", "generate")
        if mode not in SCRIPT_MODES:
            raise ValueError(f"mode must be one of {SCRIPT_MODES}, got '{mode}'")

        motion_mode = params.setdefault("motion_mode", DEFAULT_MOTION_MODE)
        if motion_mode not in MOTION_MODES:
            raise ValueError(f"motion_mode must be one of {MOTION_MODES}, got '{motion_mode}'")

        n_scenes = int(params.get("n_scenes", 6))
        if not 1 <= n_scenes <= MAX_SCENES:
            raise ValueError(f"n_scenes must be between 1 and {MAX_SCENES}, got {n_scenes}")
        params["n_scenes"] = n_scenes

        params.setdefault("frame_template", DEFAULT_TEMPLATE)
        template_type = get_template_type(params["frame_template"])
        if template_type == "static":
            raise ValueError("Animation pipeline needs an image_* or video_* template, not static_*")
        if motion_mode == "ai_video" and template_type != "video":
            raise ValueError("motion_mode='ai_video' requires a video_* frame template")
        if motion_mode == "parallax" and template_type != "video":
            raise ValueError("motion_mode='parallax' requires a video_* frame template")
        if motion_mode == "ai_video" and not params.get("media_workflow"):
            raise ValueError("motion_mode='ai_video' requires a video media_workflow")

        if not params.get("media_width") or not params.get("media_height"):
            from pixelle_video.services.frame_html import HTMLFrameGenerator
            from pixelle_video.utils.template_util import resolve_template_path

            generator = HTMLFrameGenerator(resolve_template_path(params["frame_template"]))
            width, height = generator.get_media_size()
            params["media_width"] = params.get("media_width") or width
            params["media_height"] = params.get("media_height") or height

        # Narration length defaults suited for storytelling
        params.setdefault("min_narration_words", 10)
        params.setdefault("max_narration_words", 40)

    # ==================== Content ====================

    async def generate_content(self, ctx: PipelineContext):
        """Step 2: Generate the structured animation script."""
        params = ctx.params
        if ctx.extras.get("animation_script") is not None:
            # Edited / saved script: keep its scenes as-is, no LLM call
            script = self._normalize_script(
                ctx.extras["animation_script"], len(ctx.extras["animation_script"].scenes), "adapt"
            )
            ctx.extras["animation_script"] = script
            ctx.narrations = [scene.narration for scene in script.scenes]
            logger.info(f"✅ Using provided animation script: '{script.title}' ({len(script.scenes)} scenes)")
            return

        self._report_progress(ctx.progress_callback, "generating_animation_script", 0.05)

        prompt = build_animation_script_prompt(
            idea=ctx.input_text,
            n_scenes=params["n_scenes"],
            mode=params.get("mode", "generate"),
            tone=params.get("tone"),
            min_words=params["min_narration_words"],
            max_words=params["max_narration_words"],
            max_characters=int(params.get("max_characters", 4)),
        )

        script = await self._call_script_llm(prompt)
        script = self._normalize_script(script, params["n_scenes"], params.get("mode", "generate"))

        ctx.extras["animation_script"] = script
        ctx.narrations = [scene.narration for scene in script.scenes]
        logger.info(
            f"✅ Animation script: '{script.title}' - {len(script.characters)} characters, "
            f"{len(script.scenes)} scenes"
        )

    async def _call_script_llm(self, prompt: str, retries: int = 2) -> AnimationScript:
        """Call LLM with structured output, retrying on parse failures."""
        last_error = None
        for attempt in range(1, retries + 2):
            try:
                return await self.llm(
                    prompt=prompt,
                    response_type=AnimationScript,
                    temperature=0.8,
                    max_tokens=6000,
                )
            except Exception as e:
                if not self._is_retryable_llm_error(e):
                    raise RuntimeError(f"Failed to generate animation script: {e}") from e
                last_error = e
                logger.warning(f"Animation script generation failed (attempt {attempt}): {e}")
                if attempt <= retries:
                    await asyncio.sleep(min(10.0, 2.0 * attempt))
        raise RuntimeError(f"Failed to generate animation script: {last_error}")

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        """Retry parse failures, timeouts, rate limits and 5xx; fail fast on other 4xx."""
        status = getattr(error, "status_code", None)
        if status is None:
            return True  # parse/validation errors, timeouts, connection errors
        return status in (408, 409, 429) or status >= 500

    @staticmethod
    def _normalize_script(script: AnimationScript, n_scenes: int, mode: str) -> AnimationScript:
        """Drop empty scenes, enforce scene count in generate mode, dedupe characters."""
        scenes = [s for s in script.scenes if s.narration and s.narration.strip()]
        if not scenes:
            raise RuntimeError("LLM returned an animation script without any usable scene")
        for scene in scenes:
            scene.narration = scene.narration.strip()
            if not scene.visual.strip():
                scene.visual = scene.narration
        if mode == "generate" and len(scenes) > n_scenes:
            logger.warning(f"LLM returned {len(scenes)} scenes, truncating to {n_scenes}")
            scenes = scenes[:n_scenes]
        elif len(scenes) > MAX_SCENES:
            scenes = scenes[:MAX_SCENES]

        seen = set()
        characters = []
        for character in script.characters:
            key = character.name.strip().lower()
            if key and key not in seen and character.appearance.strip():
                seen.add(key)
                characters.append(character)

        for character in characters:
            character.name = character.name.strip()
            character.display_name = character.spoken_name()
        for scene in scenes:
            scene.narration = AnimationPipeline._use_spoken_names(scene.narration, characters)

        script.scenes = scenes
        script.characters = characters
        return script

    @staticmethod
    def _use_spoken_names(narration: str, characters) -> str:
        """
        Replace ASCII character names that leaked into narration with their display names,
        so narration always uses one name per character (e.g. 'Pipi' -> '皮皮').
        """
        for character in characters:
            spoken = character.spoken_name()
            if not character.name or spoken == character.name:
                continue
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(character.name)}(?![A-Za-z0-9])", re.IGNORECASE)
            narration = pattern.sub(spoken, narration)
        return narration

    async def determine_title(self, ctx: PipelineContext):
        """Step 3: Prefer user title, then the script title."""
        script: AnimationScript = ctx.extras["animation_script"]
        ctx.title = (ctx.params.get("title") or script.title or "").strip()
        if not ctx.title:
            await super().determine_title(ctx)
        logger.info(f"   Title: '{ctx.title}'")

    # ==================== Visual planning ====================

    async def plan_visuals(self, ctx: PipelineContext):
        """Step 4: Build per-scene prompts (no extra LLM calls)."""
        self._report_progress(ctx.progress_callback, "generating_image_prompts", 0.15)
        script: AnimationScript = ctx.extras["animation_script"]
        style_prompt = get_style_prompt(ctx.params.get("art_style"), ctx.params.get("custom_style"))
        ctx.extras["style_prompt"] = style_prompt

        builder = (
            build_scene_video_prompt
            if ctx.params.get("motion_mode") == "ai_video"
            else build_scene_image_prompt
        )
        composition = composition_hint(ctx.params.get("media_width"), ctx.params.get("media_height"))
        ctx.image_prompts = [
            builder(scene, script.characters, style_prompt, composition) for scene in script.scenes
        ]
        ctx.extras["camera_motions"] = [self._resolve_camera(ctx, scene) for scene in script.scenes]
        ctx.extras["ambient_effects"] = [self._resolve_ambient(ctx, scene) for scene in script.scenes]
        # Save before any media is generated, so a failed run still leaves an editable script
        self._save_animation_script(ctx)
        logger.info(f"✅ Built {len(ctx.image_prompts)} scene prompts (style: {style_prompt[:60]}...)")

    @staticmethod
    def _resolve_camera(ctx: PipelineContext, scene: AnimationScene) -> CameraMotion:
        fixed = ctx.params.get("camera_motion", "auto")
        if fixed and fixed != "auto":
            return CameraMotion.parse(fixed)
        return CameraMotion.parse(scene.camera)

    @staticmethod
    def _resolve_ambient(ctx: PipelineContext, scene: AnimationScene) -> str:
        fixed = ctx.params.get("ambient_effect", "auto")
        if fixed and fixed != "auto":
            return normalize_effect(fixed)
        return normalize_effect(scene.ambient)

    def _parallax_renderer(self) -> ParallaxRenderer:
        """One renderer per pipeline so the segmentation model loads once."""
        if getattr(self, "_renderer", None) is None:
            self._renderer = ParallaxRenderer()
        return self._renderer

    # ==================== Asset production ====================

    async def produce_assets(self, ctx: PipelineContext):
        """Step 6: Produce audio + animated media per scene."""
        if ctx.params.get("motion_mode") == "ai_video":
            # AI video: standard frame processor already does TTS -> video -> overlay
            await super().produce_assets(ctx)
            return
        await self._run_frames(ctx, self._process_camera_frame)

    async def _run_frames(
        self,
        ctx: PipelineContext,
        worker: Callable[[PipelineContext, StoryboardFrame, Callable], Awaitable[StoryboardFrame]],
    ):
        """Run a per-frame worker with the same concurrency policy as StandardPipeline."""
        storyboard = ctx.storyboard
        config = ctx.config
        total = len(storyboard.frames)

        from pixelle_video.config import config_manager

        is_runninghub = bool(
            (config.tts_workflow and config.tts_workflow.startswith("runninghub/"))
            or (config.media_workflow and config.media_workflow.startswith("runninghub/"))
        )
        limit = config_manager.config.comfyui.runninghub_concurrent_limit or 1
        concurrency = limit if is_runninghub else 1
        semaphore = asyncio.Semaphore(max(1, concurrency))
        completed = 0
        base, span = 0.2, 0.6
        per_frame = span / total

        async def run(frame: StoryboardFrame):
            nonlocal completed

            def on_step(step: int, action: str, local_progress: float):
                if ctx.progress_callback:
                    ctx.progress_callback(ProgressEvent(
                        event_type="frame_step",
                        progress=base + per_frame * (completed + local_progress),
                        frame_current=frame.index + 1,
                        frame_total=total,
                        step=step,
                        action=action,
                    ))

            async with semaphore:
                processed = await worker(ctx, frame, on_step)
                completed += 1
                logger.info(f"✅ Scene {frame.index + 1}/{total} done ({processed.duration:.2f}s)")
                return processed

        results = await asyncio.gather(*(run(f) for f in storyboard.frames))
        storyboard.frames = sorted(results, key=lambda f: f.index)
        storyboard.total_duration = sum(f.duration for f in storyboard.frames)

    async def _process_camera_frame(
        self, ctx: PipelineContext, frame: StoryboardFrame, on_step: Callable
    ) -> StoryboardFrame:
        """TTS -> text-to-image -> camera-motion clip -> subtitle overlay -> segment."""
        processor = self.core.frame_processor
        config = ctx.config
        storyboard = ctx.storyboard
        video_service = VideoService()
        motion: CameraMotion = ctx.extras["camera_motions"][frame.index]
        parallax = ctx.params.get("motion_mode") == "parallax"
        intensity = float(ctx.params.get("camera_intensity", 0.12 if parallax else 0.15))
        template_type = get_template_type(config.frame_template)

        on_step(1, "audio", 0.0)
        await processor._step_generate_audio(frame, config)

        on_step(2, "media", 0.25)
        try:
            result = await self.core.media(
                prompt=frame.image_prompt,
                workflow=config.media_workflow,
                media_type="image",
                width=config.media_width,
                height=config.media_height,
                output_path=get_task_frame_path(config.task_id, frame.index, "image"),
                index=frame.index + 1,
            )
        except Exception as e:
            raise SceneMediaError(frame.index, frame.image_prompt, e, ctx.extras.get("script_path")) from e
        if not result.is_image:
            raise RuntimeError(
                "Camera motion mode needs an image workflow, but the workflow returned "
                f"'{result.media_type}'. Use motion_mode='ai_video' for video workflows."
            )
        frame.image_path = await processor._download_media(
            result.url, frame.index, config.task_id, media_type="image"
        )

        if template_type == "video":
            # Animate the raw image, then overlay the transparent subtitle template
            clip_path = get_task_frame_path(config.task_id, frame.index, "video")
            if parallax:
                options = ParallaxOptions(
                    motion=motion.value,
                    intensity=intensity,
                    parallax=max(0.0, min(1.5, float(ctx.params.get("parallax_strength", 0.6)))),
                    ambient=ctx.extras["ambient_effects"][frame.index],
                    breathing=bool(ctx.params.get("character_motion", True)),
                    fps=config.video_fps,
                    seed=frame.index + 1,
                )
                await asyncio.to_thread(
                    self._parallax_renderer().render,
                    frame.image_path,
                    clip_path,
                    frame.duration,
                    config.media_width,
                    config.media_height,
                    options,
                )
            else:
                await asyncio.to_thread(
                    video_service.create_motion_clip,
                    image=frame.image_path,
                    output=clip_path,
                    duration=frame.duration,
                    motion=motion.value,
                    intensity=intensity,
                    fps=config.video_fps,
                    width=config.media_width,
                    height=config.media_height,
                )
            frame.media_type = "video"
            frame.video_path = clip_path

            on_step(3, "compose", 0.5)
            await processor._step_compose_frame(frame, storyboard, config)
            on_step(4, "video", 0.75)
            await processor._step_create_video_segment(frame, config)
        else:
            # Image template: compose first, then animate the composed frame gently
            frame.media_type = "image"
            on_step(3, "compose", 0.5)
            await processor._step_compose_frame(frame, storyboard, config)

            on_step(4, "video", 0.75)
            segment_path = get_task_frame_path(config.task_id, frame.index, "segment")
            await asyncio.to_thread(
                video_service.create_motion_clip,
                image=frame.composed_image_path,
                output=segment_path,
                audio=frame.audio_path,
                motion=motion.value,
                intensity=intensity * IMAGE_TEMPLATE_INTENSITY_FACTOR,
                fps=config.video_fps,
            )
            frame.video_segment_path = segment_path

        return frame

    # ==================== Persistence ====================

    async def _persist_task_data(self, ctx: PipelineContext):
        """Persist standard metadata plus the structured animation script."""
        ctx.params["pipeline"] = "animation"
        await super()._persist_task_data(ctx)
        self._save_animation_script(ctx)

    def _save_animation_script(self, ctx: PipelineContext):
        """Write animation_script.json (script, prompts, motions) into the task directory."""
        script: AnimationScript = ctx.extras.get("animation_script")
        if not script or not ctx.task_dir:
            return
        try:
            payload = {
                "script": script.model_dump(),
                "style_prompt": ctx.extras.get("style_prompt"),
                "motion_mode": ctx.params.get("motion_mode"),
                "camera_motions": [m.value for m in ctx.extras.get("camera_motions", [])],
                "ambient_effects": ctx.extras.get("ambient_effects", []),
                "scene_prompts": ctx.image_prompts,
            }
            path = Path(ctx.task_dir) / "animation_script.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            ctx.extras["script_path"] = str(path)
            logger.info(f"💾 Saved animation script: {path}")
        except Exception as e:
            logger.error(f"Failed to save animation script: {e}")

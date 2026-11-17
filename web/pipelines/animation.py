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
Animation Pipeline UI

Three-column layout:
- Left: story input, art style, BGM
- Middle: motion mode, media workflow, subtitle template, narration voice
- Right: generate button, progress, video preview and generated script

All widget keys are prefixed with "anim_" because every pipeline tab is
rendered on the same Streamlit page.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from loguru import logger

from pixelle_video.config import config_manager
from pixelle_video.models.progress import ProgressEvent
from pixelle_video.prompts.animation_script import (
    ANIMATION_STYLE_PRESETS,
    DEFAULT_ANIMATION_STYLE,
)
from web.components.content_input import render_bgm_section, render_version_info
from web.i18n import get_language, tr
from web.pipelines.api_workflows import (
    list_api_media_workflows,
    list_local_media_workflows,
    render_api_video_controls,
    workflow_select_help,
    workflow_source_help,
    workflow_source_label,
)
from web.pipelines.base import PipelineUI, register_pipeline_ui
from web.utils.async_helpers import run_async
from web.utils.streamlit_helpers import check_and_warn_selfhost_workflow

AMBIENT_OPTIONS = ["auto", "none", "fireflies", "stars", "snow", "rain", "petals", "dust", "embers", "bubbles"]
CAMERA_OPTIONS = ["auto", "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "static"]
KEY = "anim_"


def _style_label(key: str) -> str:
    preset = ANIMATION_STYLE_PRESETS[key]
    return preset["name_zh"] if get_language() == "zh_CN" else preset["name_en"]


class AnimationPipelineUI(PipelineUI):
    """UI for the Animation (animated short) pipeline."""

    name = "animation"
    icon = "🎞️"

    @property
    def display_name(self):
        return tr("pipeline.animation.name")

    @property
    def description(self):
        return tr("pipeline.animation.description")

    def render(self, pixelle_video: Any):
        left_col, middle_col, right_col = st.columns([1, 1, 1])

        with left_col:
            story_params = self._render_story_input()
            bgm_params = render_bgm_section(key_prefix=KEY)
            render_version_info()

        with middle_col:
            motion_params = self._render_motion_config(pixelle_video)
            voice_params = self._render_voice_config(pixelle_video)

        with right_col:
            params = {**story_params, **bgm_params, **motion_params, **voice_params}
            self._render_output(pixelle_video, params)

    # ------------------------------------------------------------------ left

    def _render_story_input(self) -> dict:
        with st.container(border=True):
            st.markdown(f"**{tr('animation.section.story')}**")
            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(f"**{tr('help.what')}**")
                st.markdown(tr("animation.story.what"))
                st.markdown(f"**{tr('help.how')}**")
                st.markdown(tr("animation.story.how"))

            mode = st.radio(
                tr("animation.mode"),
                ["generate", "adapt"],
                format_func=lambda m: tr(f"animation.mode.{m}"),
                horizontal=True,
                key=f"{KEY}mode",
            )
            text = st.text_area(
                tr("animation.story.input"),
                placeholder=tr(f"animation.story.placeholder_{mode}"),
                height=160,
                key=f"{KEY}text",
            )
            title = st.text_input(tr("animation.title"), placeholder=tr("animation.title_placeholder"),
                                  key=f"{KEY}title")
            tone = st.text_input(tr("animation.tone"), placeholder=tr("animation.tone_placeholder"),
                                 key=f"{KEY}tone")

            col1, col2 = st.columns(2)
            with col1:
                n_scenes = st.slider(tr("animation.n_scenes"), 2, 20, 6, key=f"{KEY}n_scenes")
            with col2:
                max_characters = st.slider(tr("animation.max_characters"), 0, 6, 3,
                                           key=f"{KEY}max_characters")

        with st.container(border=True):
            st.markdown(f"**{tr('animation.section.style')}**")
            style_keys = list(ANIMATION_STYLE_PRESETS.keys()) + ["custom"]
            art_style = st.selectbox(
                tr("animation.art_style"),
                style_keys,
                index=style_keys.index(DEFAULT_ANIMATION_STYLE),
                format_func=lambda k: tr("animation.art_style.custom") if k == "custom" else _style_label(k),
                key=f"{KEY}art_style",
            )
            custom_style = None
            if art_style == "custom":
                custom_style = st.text_area(tr("animation.custom_style"), height=80,
                                            placeholder="e.g. Ghibli-like hand painted background, soft light",
                                            key=f"{KEY}custom_style")
            else:
                st.caption(ANIMATION_STYLE_PRESETS[art_style]["prompt"])

        return {
            "mode": mode,
            "text": text,
            "title": title.strip() or None,
            "tone": tone.strip() or None,
            "n_scenes": n_scenes,
            "max_characters": max_characters,
            "art_style": DEFAULT_ANIMATION_STYLE if art_style == "custom" else art_style,
            "custom_style": custom_style,
        }

    # ---------------------------------------------------------------- middle

    def _render_motion_config(self, pixelle_video) -> dict:
        from pixelle_video.services.frame_html import HTMLFrameGenerator
        from pixelle_video.utils.template_util import (
            get_all_templates_with_info,
            get_template_type,
            resolve_template_path,
        )

        api_video_params = None
        camera_motion, camera_intensity = "auto", 0.15
        ambient_effect, parallax_strength, character_motion = "auto", 0.6, True

        with st.container(border=True):
            st.markdown(f"**{tr('animation.section.motion')}**")
            with st.expander(tr("help.feature_description"), expanded=False):
                st.markdown(tr("animation.motion.what"))

            motion_mode = st.radio(
                tr("animation.motion_mode"),
                ["parallax", "camera", "ai_video"],
                format_func=lambda m: tr(f"animation.motion_mode.{m}"),
                horizontal=True,
                key=f"{KEY}motion_mode",
            )
            media_type = "video" if motion_mode == "ai_video" else "image"

            # Subtitle template: video_* always; image_* only for plain camera mode
            allowed_types = {"video", "image"} if motion_mode == "camera" else {"video"}
            templates = [
                t.template_path for t in get_all_templates_with_info()
                if get_template_type(t.template_path) in allowed_types
            ]
            # Full-screen video templates first: they suit animation best
            templates.sort(key=lambda p: (
                get_template_type(p) != "video", "fullscreen" not in p, not p.startswith("1080x1920"), p
            ))
            frame_template = st.selectbox(tr("animation.template"), templates, key=f"{KEY}template")
            if get_template_type(frame_template) == "image":
                st.caption(tr("animation.template.image_hint"))

            media_width = media_height = None
            try:
                generator = HTMLFrameGenerator(resolve_template_path(frame_template))
                media_width, media_height = generator.get_media_size()
            except Exception as e:
                logger.warning(f"Failed to read template media size: {e}")

            # Media workflow
            source = st.radio(
                tr("animation.source"),
                ["runninghub", "selfhost", "api"],
                format_func=workflow_source_label,
                horizontal=True,
                key=f"{KEY}source_{media_type}",
                help=workflow_source_help(),
            )
            if source == "api":
                if media_type == "video":
                    workflows = list_api_media_workflows(
                        pixelle_video, "video",
                        required_adapter_abilities=["text_to_video"], verified_only=True,
                    )
                else:
                    workflows = list_api_media_workflows(pixelle_video, "image")
            elif media_type == "video":
                workflows = list_local_media_workflows(pixelle_video, "video", source, key_contains="video_")
            else:
                workflows = list_local_media_workflows(pixelle_video, "image", source)

            workflow_key, workflow_info = None, None
            if workflows:
                labels = [w["display_name"] for w in workflows]
                saved = config_manager.get_comfyui_config().get(media_type, {}).get("default_workflow")
                keys = [w["key"] for w in workflows]
                index = keys.index(saved) if saved in keys else 0
                label = st.selectbox(tr("animation.workflow"), labels, index=index,
                                     key=f"{KEY}workflow_{media_type}_{source}", help=workflow_select_help())
                workflow_info = workflows[labels.index(label)]
                workflow_key = workflow_info["key"]
                if not workflow_key.startswith("api/"):
                    check_and_warn_selfhost_workflow(workflow_key)
            else:
                st.warning(tr("animation.workflow.none"))

            if media_width and media_height:
                st.info(f"📐 {media_width}x{media_height}")

            if motion_mode in ("parallax", "camera"):
                col1, col2 = st.columns(2)
                with col1:
                    camera_motion = st.selectbox(
                        tr("animation.camera"), CAMERA_OPTIONS,
                        format_func=lambda c: tr(f"animation.camera.{c}"),
                        key=f"{KEY}camera",
                    )
                with col2:
                    default_intensity = 0.12 if motion_mode == "parallax" else 0.15
                    camera_intensity = st.slider(tr("animation.intensity"), 0.0, 0.4, default_intensity, 0.01,
                                                 key=f"{KEY}intensity_{motion_mode}")
            if motion_mode == "parallax":
                col1, col2 = st.columns(2)
                with col1:
                    ambient_effect = st.selectbox(
                        tr("animation.ambient"), AMBIENT_OPTIONS,
                        format_func=lambda a: tr(f"animation.ambient.{a}"),
                        key=f"{KEY}ambient",
                    )
                with col2:
                    parallax_strength = st.slider(tr("animation.parallax_strength"), 0.0, 1.5, 0.6, 0.1,
                                                  key=f"{KEY}parallax_strength")
                character_motion = st.checkbox(tr("animation.character_motion"), value=True,
                                               key=f"{KEY}character_motion")
                st.caption(tr("animation.parallax.model_hint"))
            elif workflow_key and workflow_key.startswith("api/"):
                ratio = "9:16" if (media_height or 0) > (media_width or 0) else "16:9"
                api_video_params = render_api_video_controls(
                    workflow_info, key_prefix=f"{KEY}video", default_duration=5,
                    allow_audio_driven=False, show_duration=False, default_ratio=ratio,
                )

        return {
            "motion_mode": motion_mode,
            "frame_template": frame_template,
            "media_workflow": workflow_key,
            "media_width": media_width,
            "media_height": media_height,
            "camera_motion": camera_motion,
            "camera_intensity": camera_intensity,
            "ambient_effect": ambient_effect if motion_mode == "parallax" else None,
            "parallax_strength": parallax_strength if motion_mode == "parallax" else None,
            "character_motion": character_motion if motion_mode == "parallax" else None,
            "api_video_params": api_video_params,
        }

    def _render_voice_config(self, pixelle_video) -> dict:
        from pixelle_video.tts_voices import EDGE_TTS_VOICES, get_voice_display_name

        tts_config = config_manager.get_comfyui_config()["tts"]
        with st.container(border=True):
            st.markdown(f"**{tr('section.tts')}**")
            tts_mode = st.radio(
                tr("tts.inference_mode"), ["local", "comfyui"],
                format_func=lambda x: tr(f"tts.mode.{x}"), horizontal=True,
                key=f"{KEY}tts_mode",
            )
            if tts_mode == "local":
                voice_ids = [v["id"] for v in EDGE_TTS_VOICES]
                saved_voice = tts_config.get("local", {}).get("voice", "zh-CN-YunjianNeural")
                voice = st.selectbox(
                    tr("tts.voice_selector"), voice_ids,
                    index=voice_ids.index(saved_voice) if saved_voice in voice_ids else 0,
                    format_func=lambda v: get_voice_display_name(v, tr, get_language()),
                    key=f"{KEY}voice",
                )
                speed = st.slider(tr("tts.speed"), 0.5, 2.0,
                                  float(tts_config.get("local", {}).get("speed", 1.1)), 0.1,
                                  format="%.1fx", key=f"{KEY}speed")
                return {"tts_inference_mode": "local", "tts_voice": voice, "tts_speed": speed}

            workflows = pixelle_video.tts.list_workflows()
            if not workflows:
                st.warning("No TTS workflows found")
                return {"tts_inference_mode": "comfyui", "tts_workflow": None}
            labels = [w["display_name"] for w in workflows]
            label = st.selectbox("TTS Workflow", labels, key=f"{KEY}tts_workflow")
            workflow_key = workflows[labels.index(label)]["key"]
            check_and_warn_selfhost_workflow(workflow_key)
            return {"tts_inference_mode": "comfyui", "tts_workflow": workflow_key}

    # ----------------------------------------------------------------- right

    def _render_output(self, pixelle_video, params: dict):
        with st.container(border=True):
            st.markdown(f"**{tr('section.video_generation')}**")
            if not config_manager.validate():
                st.warning(tr("settings.not_configured"))

            if not st.button(tr("animation.generate"), type="primary", use_container_width=True,
                             key=f"{KEY}generate"):
                return

            error = self._validate(params)
            if error:
                st.error(error)
                return

            progress_bar = st.progress(0)
            status_text = st.empty()
            start = time.time()

            def on_progress(event: ProgressEvent):
                if event.event_type == "frame_step":
                    message = tr("progress.frame_step", current=event.frame_current,
                                 total=event.frame_total, step=event.step,
                                 action=tr(f"progress.step_{event.action}"))
                elif event.event_type == "processing_frame":
                    message = tr("progress.frame", current=event.frame_current, total=event.frame_total)
                else:
                    message = tr(f"progress.{event.event_type}")
                status_text.text(message)
                progress_bar.progress(min(int(event.progress * 100), 99))

            text = params.pop("text")
            call_params = {k: v for k, v in params.items() if v is not None}
            try:
                result = run_async(pixelle_video.generate_video(
                    text=text, pipeline="animation", progress_callback=on_progress, **call_params
                ))
            except Exception as e:
                progress_bar.empty()
                status_text.text("")
                st.error(tr("status.error", error=str(e)))
                logger.exception(e)
                return

            progress_bar.progress(100)
            status_text.text(tr("status.success"))
            st.success(tr("status.video_generated", path=result.video_path))
            st.caption(
                f"⏱️ {time.time() - start:.1f}s   📦 {result.file_size / 1048576:.2f}MB   "
                f"🎬 {len(result.storyboard.frames)}   ⏳ {result.duration:.1f}s"
            )
            if os.path.exists(result.video_path):
                st.video(result.video_path)
                with open(result.video_path, "rb") as f:
                    st.download_button("⬇️ Download", f.read(), file_name=os.path.basename(result.video_path),
                                       mime="video/mp4", use_container_width=True, key=f"{KEY}download")
            self._render_script(Path(result.video_path).parent / "animation_script.json")

    @staticmethod
    def _validate(params: dict) -> Optional[str]:
        if not config_manager.validate():
            return tr("settings.not_configured")
        if not (params.get("text") or "").strip():
            return tr("error.input_required")
        if not params.get("media_workflow"):
            return tr("animation.workflow.none")
        if params.get("art_style") and params.get("custom_style") is not None and not params["custom_style"].strip():
            return tr("animation.custom_style.required")
        return None

    @staticmethod
    def _render_script(path: Path):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read animation script: {e}")
            return
        script = data.get("script", {})
        with st.expander(tr("animation.script"), expanded=False):
            if script.get("logline"):
                st.markdown(f"*{script['logline']}*")
            for character in script.get("characters", []):
                spoken = character.get("display_name") or character["name"]
                label = spoken if spoken == character["name"] else f"{spoken} ({character['name']})"
                st.markdown(f"- **{label}**: {character['appearance']}")
            for i, scene in enumerate(script.get("scenes", []), 1):
                camera = (data.get("camera_motions") or [""] * i)[i - 1]
                st.markdown(f"**{i}.** {scene['narration']}  \n`{camera}` · {scene['visual']}")
            st.download_button("animation_script.json", path.read_bytes(), file_name="animation_script.json",
                               mime="application/json", key=f"{KEY}script_download")


register_pipeline_ui(AnimationPipelineUI)

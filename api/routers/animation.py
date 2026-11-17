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
Animation generation endpoints.

- GET  /animation/styles          list art style presets
- POST /animation/generate/sync   generate and wait
- POST /animation/generate/async  create a background task (poll /tasks/{id})
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from api.dependencies import PixelleVideoDep
from api.routers.video import path_to_url
from api.schemas.animation import (
    AnimationGenerateAsyncResponse,
    AnimationGenerateRequest,
    AnimationGenerateResponse,
)
from api.tasks import TaskType, task_manager
from pixelle_video.pipelines.animation import SceneMediaError
from pixelle_video.prompts.animation_script import ANIMATION_STYLE_PRESETS

router = APIRouter(prefix="/animation", tags=["Animation Generation"])


@router.get("/styles")
async def list_styles():
    """List available art style presets."""
    return {
        "styles": [
            {"key": key, "name_zh": v["name_zh"], "name_en": v["name_en"], "prompt": v["prompt"]}
            for key, v in ANIMATION_STYLE_PRESETS.items()
        ]
    }


def _build_params(body: AnimationGenerateRequest) -> dict:
    if body.min_narration_words > body.max_narration_words:
        raise ValueError("min_narration_words must be <= max_narration_words")
    params = body.model_dump(exclude_none=True)
    params.pop("text")
    return params


def _progress_message(event) -> str:
    """Short machine-friendly progress label, e.g. 'media 2/5' or 'concatenating'."""
    label = event.action or event.event_type
    if event.frame_current and event.frame_total:
        label += f" {event.frame_current}/{event.frame_total}"
    return label


async def _run(pixelle_video, body: AnimationGenerateRequest, request: Request, progress_callback=None) -> dict:
    result = await pixelle_video.generate_video(
        text=body.text, pipeline="animation", progress_callback=progress_callback, **_build_params(body)
    )
    script_path = Path(result.video_path).parent / "animation_script.json"
    return {
        "output_id": Path(result.video_path).parent.name,
        "video_url": path_to_url(request, result.video_path),
        "duration": result.duration,
        "file_size": result.file_size,
        "title": result.storyboard.title,
        "script_url": path_to_url(request, str(script_path)) if script_path.exists() else None,
    }


@router.post("/generate/sync", response_model=AnimationGenerateResponse)
async def generate_animation_sync(
    body: AnimationGenerateRequest, pixelle_video: PixelleVideoDep, request: Request
):
    """Generate an animation and block until it is done (prefer async for > 3 scenes)."""
    try:
        logger.info(f"Sync animation generation: {body.text[:50]}...")
        return AnimationGenerateResponse(**await _run(pixelle_video, body, request))
    except SceneMediaError as e:
        raise HTTPException(status_code=422, detail={
            "message": e.public_message,
            "scene": e.scene_index + 1,
            "prompt": e.prompt,
            "safety_blocked": e.safety_blocked,
            "quota_exhausted": e.quota_exhausted,
            "script_url": path_to_url(request, e.script_path) if e.script_path else None,
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Sync animation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/async", response_model=AnimationGenerateAsyncResponse)
async def generate_animation_async(
    body: AnimationGenerateRequest, pixelle_video: PixelleVideoDep, request: Request
):
    """Create a background animation task; poll /api/tasks/{task_id} for the result."""
    try:
        _build_params(body)  # fail fast on invalid combinations
        task = task_manager.create_task(
            task_type=TaskType.VIDEO_GENERATION,
            request_params={"pipeline": "animation", **body.model_dump()},
        )

        def on_progress(event):
            task_manager.update_progress(task.task_id, int(round(event.progress * 100)), 100,
                                         _progress_message(event))

        async def execute():
            try:
                return await _run(pixelle_video, body, request, progress_callback=on_progress)
            except SceneMediaError as e:
                # Task errors are returned to API clients: use the path-free message
                raise RuntimeError(e.public_message) from e

        await task_manager.execute_task(task_id=task.task_id, coro_func=execute)
        return AnimationGenerateAsyncResponse(task_id=task.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Async animation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

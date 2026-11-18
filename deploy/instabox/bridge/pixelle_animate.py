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
/animate command for an OpenAI-compatible chat bridge (e.g. webui_bridge.py).

Standard library only, so it can be imported by a bridge running on the system
Python. The bridge hooks in with three calls:

    import pixelle_animate
    # chat route, before other commands:
    if pixelle_animate.ANIMATE_PATTERN.search(last_user):
        pixelle_animate.handle_animate(self, last_user, is_stream, stop_llm=stop_llm); return
    # GET route:
    if pixelle_animate.serve(self, path_only): return

The handler object must provide `send_sse_delta(cid, text)` and
`send_file_range(path, content_type)` like webui_bridge.BridgeHandler.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Optional

PIXELLE_API = os.environ.get("PIXELLE_API_URL", "http://127.0.0.1:8000")
PIXELLE_OUTPUT_DIR = Path(os.environ.get("PIXELLE_OUTPUT_DIR", "/data/workspace/pixelle-video/output"))
IMAGE_WORKFLOW = os.environ.get("PIXELLE_ANIMATE_WORKFLOW", "api/cloudflare/flux-1-schnell")
VOICE = os.environ.get("PIXELLE_ANIMATE_VOICE", "zh-CN-XiaoxiaoNeural")
TIMEOUT_SECONDS = int(os.environ.get("PIXELLE_ANIMATE_TIMEOUT", "1200"))
POLL_SECONDS = 3.0
ROUTE = "/animations/"

ANIMATE_PATTERN = re.compile(
    r"(^\s*/animate\b|(做|生成|產生|产生|製作|制作)(一[支部段個个])?(動畫|动画)|\bmake (an )?animation\b)",
    re.IGNORECASE,
)

STYLE_KEYWORDS = [
    (r"水彩|繪本|绘本|watercolor", "watercolor"),
    (r"水墨|國畫|国画|ink", "chinese_ink"),
    (r"像素|pixel", "pixel_art"),
    (r"3d|三維|三维|皮克斯", "3d_cartoon"),
    (r"黏土|粘土|clay", "clay"),
    (r"美漫|漫畫|漫画|comic", "comic"),
    (r"扁平|向量|flat", "flat_vector"),
    (r"日系|動漫|动漫|anime", "anime"),
]

STAGE_LABELS = {
    "generating_animation_script": "撰寫劇本",
    "generating_image_prompts": "規劃分鏡",
    "audio": "配音",
    "media": "產圖",
    "compose": "字幕",
    "video": "渲染",
    "concatenating": "合併影片",
    "completed": "完成",
}

_OUTPUT_ID = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{4}$")
_SERVABLE = {"final.mp4": "video/mp4", "animation_script.json": "application/json"}
_OPTION_PHRASES = re.compile(
    r"(^\s*/animate\b|(做|生成|產生|产生|製作|制作)(一[支部段個个])?(動畫|动画)[:：，,]?|"
    r"\d+\s*(個|个)?\s*(分鏡|分镜|幕|scenes?)|直式|橫式|横式|9:16|16:9)",
    re.IGNORECASE,
)


def parse_request(text: str) -> dict:
    """Extract the story idea and options (scenes, style, orientation) from chat text."""
    params = {"n_scenes": 5, "art_style": "anime", "frame_template": "1080x1920/video_fullscreen.html"}
    match = re.search(r"(\d+)\s*(個|个)?\s*(分鏡|分镜|幕|scenes?)", text, re.IGNORECASE)
    if match:
        params["n_scenes"] = max(2, min(10, int(match.group(1))))
    idea = text
    for pattern, style in STYLE_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            params["art_style"] = style
            # Drop the keyword from the story only when it is a standalone option token
            idea = re.sub(rf"(^|\s)({pattern})(?=\s|$)", " ", idea, flags=re.IGNORECASE)
            break
    if re.search(r"橫式|横式|16:9|landscape", text, re.IGNORECASE):
        params["frame_template"] = "1920x1080/video_fullscreen.html"
    idea = _OPTION_PHRASES.sub(" ", idea)
    params["text"] = re.sub(r"\s+", " ", idea).strip(" ，,。:：")
    return params


class PixelleClient:
    """Minimal client for the Pixelle-Video animation API."""

    def __init__(self, base_url: str = PIXELLE_API, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("detail", body)
            except ValueError:
                detail = body
            raise RuntimeError(f"Pixelle API {e.code}: {detail}") from e

    def start(self, params: dict) -> str:
        body = dict(params, motion_mode="parallax", media_workflow=IMAGE_WORKFLOW,
                    tts_inference_mode="local", tts_voice=VOICE)
        return self._request("POST", "/api/animation/generate/async", body)["task_id"]

    def status(self, task_id: str) -> dict:
        return self._request("GET", f"/api/tasks/{task_id}")


def describe_progress(task: dict) -> str:
    """'產圖 2/5 (35%)' style label from a Pixelle task."""
    progress = task.get("progress") or {}
    message = progress.get("message") or ""
    stage, _, frames = message.partition(" ")
    label = STAGE_LABELS.get(stage, stage or "處理中")
    if frames:
        label += f" {frames}"
    return f"{label}（{int(progress.get('percentage') or 0)}%）"


def result_markdown(result: dict, elapsed: int) -> str:
    output_id = result.get("output_id", "")
    video = f"{ROUTE}{output_id}/final.mp4"
    script = f"{ROUTE}{output_id}/animation_script.json"
    note = ("\n\nℹ️ 雲端產圖額度已用完，部分分鏡改用本機模型產生（較慢、畫風可能略有差異）。"
            if result.get("image_fallback_used") else "")
    return (
        f"\n\n✅ **完成！**「{result.get('title', '')}」· {result.get('duration', 0):.1f} 秒 · 耗時 `{elapsed}s`{note}\n\n"
        f"<video src=\"{video}\" controls style=\"max-width:100%; border-radius:8px;\"></video>\n\n"
        f"🔗 [觀看 / 下載影片]({video})  |  📜 [劇本 JSON（可編輯後重新生成）]({script})"
    )


def _finish_stream(handler, cid: str):
    done = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "pixelle-animate",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    handler.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode("utf-8"))
    handler.wfile.flush()


def _reply_json(handler, cid: str, text: str):
    out = json.dumps({
        "id": cid, "object": "chat.completion", "created": int(time.time()), "model": "pixelle-animate",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }, ensure_ascii=False).encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(out)))
    handler.end_headers()
    handler.wfile.write(out)


def handle_animate(handler, user_text: str, is_stream: bool, stop_llm: Optional[Callable[[], None]] = None,
                   client: Optional[PixelleClient] = None, sleep: Callable[[float], None] = time.sleep):
    """Start a Pixelle animation from chat text and stream progress until it finishes."""
    client = client or PixelleClient()
    cid = f"chatcmpl-anim-{uuid.uuid4().hex[:8]}"
    params = parse_request(user_text)
    if not params["text"]:
        message = "請在 `/animate` 後面寫下故事點子，例如：`/animate 一隻怕黑的小狐狸找回月亮 5個分鏡 水彩`"
        if is_stream:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.end_headers()
            handler.send_sse_delta(cid, message)
            _finish_stream(handler, cid)
        else:
            _reply_json(handler, cid, message)
        return

    if is_stream:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()

    if stop_llm:
        stop_llm()  # free RAM for rendering; the bridge restarts the LLM on the next chat
    try:
        task_id = client.start(params)
    except Exception as e:
        text = f"❌ **無法建立動畫任務**：`{e}`"
        if is_stream:
            handler.send_sse_delta(cid, text)
            _finish_stream(handler, cid)
        else:
            _reply_json(handler, cid, text)
        return

    header = (
        f"🎞️ **動畫任務已建立**（`{task_id[:8]}`）\n\n"
        f"> {params['text']}\n\n"
        f"🎨 畫風 `{params['art_style']}` · {params['n_scenes']} 個分鏡 · "
        f"{'橫式' if params['frame_template'].startswith('1920') else '直式'}\n\n"
    )
    if not is_stream:
        _reply_json(handler, cid, header + f"⏳ 背景製作中，進度：[{ROUTE}status/{task_id}]({ROUTE}status/{task_id})")
        return

    handler.send_sse_delta(cid, header + "⏳ ")
    start, last_label = time.time(), ""
    task = {}
    while time.time() - start < TIMEOUT_SECONDS:
        sleep(POLL_SECONDS)
        try:
            task = client.status(task_id)
        except Exception as e:
            handler.send_sse_delta(cid, f"\n\n⚠️ 查詢進度失敗：`{e}`")
            break
        if task.get("status") in ("completed", "failed", "cancelled"):
            break
        label = describe_progress(task)
        handler.send_sse_delta(cid, f"\n▸ {label}" if label != last_label else "·")
        last_label = label

    status = task.get("status")
    if status == "completed":
        handler.send_sse_delta(cid, result_markdown(task.get("result") or {}, int(time.time() - start)))
    elif status in ("failed", "cancelled"):
        handler.send_sse_delta(cid, f"\n\n❌ **製作失敗**：{task.get('error') or status}")
    else:
        handler.send_sse_delta(cid, f"\n\n⏳ 仍在背景製作中，稍後查看：[{ROUTE}status/{task_id}]({ROUTE}status/{task_id})")
    _finish_stream(handler, cid)


def serve(handler, path_only: str, client: Optional[PixelleClient] = None) -> bool:
    """
    Serve /animations/<output_id>/<file> (allowlisted files only) and
    /animations/status/<task_id>. Returns True when the request was handled.
    """
    if not path_only.startswith(ROUTE):
        return False
    rest = path_only[len(ROUTE):]
    parts = rest.split("/")
    if len(parts) == 2 and parts[0] == "status" and re.fullmatch(r"[0-9a-fA-F-]{8,64}", parts[1]):
        try:
            payload = (client or PixelleClient()).status(parts[1])
        except Exception as e:
            payload = {"error": str(e)}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    if len(parts) == 2 and _OUTPUT_ID.fullmatch(parts[0]) and parts[1] in _SERVABLE:
        handler.send_file_range(PIXELLE_OUTPUT_DIR / parts[0] / parts[1], _SERVABLE[parts[1]])
        return True
    handler.send_response(404)
    handler.send_header("Content-Length", "9")
    handler.end_headers()
    handler.wfile.write(b"Not found")
    return True

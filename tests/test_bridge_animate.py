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

"""Tests for the /animate chat-bridge module and its installer."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

BRIDGE_DIR = Path(__file__).resolve().parents[1] / "deploy" / "instabox" / "bridge"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BRIDGE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pa = _load("pixelle_animate")
hooks = _load("install_bridge_hooks")


class FakeHandler:
    def __init__(self):
        self.wfile = io.BytesIO()
        self.status = None
        self.deltas = []
        self.files = []

    def send_response(self, code):
        self.status = code

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass

    def send_sse_delta(self, cid, text):
        self.deltas.append(text)

    def send_file_range(self, path, ctype):
        self.files.append((str(path), ctype))


class FakeClient:
    def __init__(self, statuses, fail_start=None):
        self.statuses = list(statuses)
        self.fail_start = fail_start
        self.started = None

    def start(self, params):
        if self.fail_start:
            raise RuntimeError(self.fail_start)
        self.started = params
        return "0123456789abcdef"

    def status(self, task_id):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]


# ---------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,expected", [
    ("/animate 一隻怕黑的小狐狸找回月亮", True),
    ("幫我做一支動畫：小貓學游泳", True),
    ("生成动画 小猫", True),
    ("畫一張貓", False),
    ("做影片 海邊", False),
])
def test_trigger_pattern(text, expected):
    assert bool(pa.ANIMATE_PATTERN.search(text)) is expected


def test_parse_request_options():
    p = pa.parse_request("/animate 一隻怕黑的小狐狸找回月亮 8個分鏡 水彩 橫式")
    assert p["text"] == "一隻怕黑的小狐狸找回月亮"
    assert p["n_scenes"] == 8 and p["art_style"] == "watercolor"
    assert p["frame_template"] == "1920x1080/video_fullscreen.html"
    d = pa.parse_request("做一支動畫：小貓 30 scenes")
    assert d["n_scenes"] == 10 and d["art_style"] == "anime" and d["frame_template"].startswith("1080x1920")
    assert d["text"] == "小貓"
    # a style word inside the story is kept in the story (and still picks the style)
    w = pa.parse_request("/animate 喜歡畫水彩的小熊")
    assert w["text"] == "喜歡畫水彩的小熊" and w["art_style"] == "watercolor"


def test_describe_progress():
    task = {"progress": {"percentage": 35.2, "message": "media 2/5"}}
    assert pa.describe_progress(task) == "產圖 2/5（35%）"
    assert pa.describe_progress({}) == "處理中（0%）"


# ---------------------------------------------------------------- chat flow

def test_stream_flow_reports_progress_and_video():
    handler, stopped = FakeHandler(), []
    client = FakeClient([
        {"status": "running", "progress": {"percentage": 20, "message": "audio 1/5"}},
        {"status": "running", "progress": {"percentage": 20, "message": "audio 1/5"}},
        {"status": "running", "progress": {"percentage": 60, "message": "video 3/5"}},
        {"status": "completed", "result": {"output_id": "20261117_144752_9683", "title": "小狐狸", "duration": 36.8}},
    ])
    pa.handle_animate(handler, "/animate 小狐狸 3個分鏡", True, stop_llm=lambda: stopped.append(1),
                      client=client, sleep=lambda s: None)
    text = "".join(handler.deltas)
    assert stopped == [1] and client.started["n_scenes"] == 3
    assert "配音 1/5（20%）" in text and "渲染 3/5（60%）" in text and "·" in text
    assert '<video src="/animations/20261117_144752_9683/final.mp4"' in text
    assert b"[DONE]" in handler.wfile.getvalue()


def test_stream_flow_reports_failure():
    handler = FakeHandler()
    client = FakeClient([{"status": "failed", "error": "Scene 2 image generation rejected by the image model's "
                                                        "safety filter"}])
    pa.handle_animate(handler, "/animate 小狐狸", True, client=client, sleep=lambda s: None)
    assert "❌" in "".join(handler.deltas) and "safety filter" in "".join(handler.deltas)


def test_start_error_and_empty_idea():
    handler = FakeHandler()
    pa.handle_animate(handler, "/animate 小狐狸", True, client=FakeClient([], fail_start="Pixelle API 400: bad"),
                      sleep=lambda s: None)
    assert "無法建立動畫任務" in "".join(handler.deltas)

    handler = FakeHandler()
    pa.handle_animate(handler, "/animate", False, client=FakeClient([]), sleep=lambda s: None)
    body = json.loads(handler.wfile.getvalue())
    assert "故事點子" in body["choices"][0]["message"]["content"]


def test_non_stream_returns_status_link():
    handler = FakeHandler()
    pa.handle_animate(handler, "/animate 小狐狸", False, client=FakeClient([{}]), sleep=lambda s: None)
    content = json.loads(handler.wfile.getvalue())["choices"][0]["message"]["content"]
    assert "/animations/status/0123456789abcdef" in content


# ---------------------------------------------------------------- file route

@pytest.mark.parametrize("path,served", [
    ("/animations/20261117_144752_9683/final.mp4", True),
    ("/animations/20261117_144752_9683/animation_script.json", True),
    ("/animations/20261117_144752_9683/frames/01_image.png", False),
    ("/animations/../../etc/passwd", False),
    ("/animations/20261117_144752_9683/../secrets.env", False),
    ("/animations/x/final.mp4", False),
])
def test_serve_allowlist(path, served):
    handler = FakeHandler()
    assert pa.serve(handler, path) is True
    assert bool(handler.files) is served
    if served:
        assert handler.files[0][0].startswith(str(pa.PIXELLE_OUTPUT_DIR))
    else:
        assert handler.status == 404


def test_serve_ignores_other_paths_and_proxies_status():
    assert pa.serve(FakeHandler(), "/generated/x.png") is False
    handler = FakeHandler()
    assert pa.serve(handler, "/animations/status/0123456789abcdef", client=FakeClient([{"status": "running"}]))
    assert json.loads(handler.wfile.getvalue())["status"] == "running"


# ---------------------------------------------------------------- installer

STUB_BRIDGE = '''import sys
DEFAULT_NEGATIVE = "x"

class BridgeHandler:
    def do_GET(self):
        path_only = "/"
        # 3. SD Generated image serving
        return None

    def do_POST(self):
        last_user, data = "", {}
        if True:
            # Check for Video generation request
            if last_user:
                pass
'''


def test_installer_is_idempotent_and_compiles(tmp_path):
    bridge = tmp_path / "webui_bridge.py"
    bridge.write_text(STUB_BRIDGE, encoding="utf-8")
    hooks.main(str(bridge))
    patched = bridge.read_text(encoding="utf-8")
    assert patched.count(hooks.MARK) == 3
    assert list(tmp_path.glob("webui_bridge.py.bak-*"))
    hooks.main(str(bridge))
    assert bridge.read_text(encoding="utf-8") == patched
    compile(patched, "bridge", "exec")


def test_installer_refuses_unknown_layout(tmp_path):
    bridge = tmp_path / "webui_bridge.py"
    bridge.write_text("print('hi')\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        hooks.main(str(bridge))
    assert bridge.read_text(encoding="utf-8") == "print('hi')\n"

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

"""Deployment support: ${ENV} config references and system Chromium selection."""

from pathlib import Path

import yaml

from pixelle_video.config.loader import (
    expand_env_refs,
    load_config_dict,
    restore_env_refs,
    save_config_dict,
)
from pixelle_video.config.schema import PixelleVideoConfig
from pixelle_video.services.frame_html import HTMLFrameGenerator

DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "instabox"


def test_expand_env_refs(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "s3cr3t")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    data = {"llm": {"api_key": "${SECRET_KEY}", "model": "${MISSING_VAR:-auto}",
                    "base_url": "http://x/${NOT_WHOLE}"}, "n": 3}
    expanded, refs = expand_env_refs(data)
    assert expanded["llm"]["api_key"] == "s3cr3t"
    assert expanded["llm"]["model"] == "auto"
    assert expanded["llm"]["base_url"] == "http://x/${NOT_WHOLE}"  # partial refs untouched
    assert refs == {("llm", "api_key"): "${SECRET_KEY}", ("llm", "model"): "${MISSING_VAR:-auto}"}


def test_save_keeps_references_not_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("SECRET_KEY", "s3cr3t")
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  api_key: ${SECRET_KEY}\n  base_url: http://a\n  model: m\n", encoding="utf-8")
    refs = {}
    data = load_config_dict(str(path), refs=refs)
    assert data["llm"]["api_key"] == "s3cr3t"

    data["llm"]["base_url"] = "http://b"            # unrelated edit in the UI
    save_config_dict(data, str(path), refs=refs)
    saved = path.read_text(encoding="utf-8")
    assert "s3cr3t" not in saved
    assert yaml.safe_load(saved)["llm"] == {"api_key": "${SECRET_KEY}", "base_url": "http://b", "model": "m"}

    # An explicit new value typed in the UI replaces the reference
    assert restore_env_refs({"llm": {"api_key": "typed"}}, refs) == {"llm": {"api_key": "typed"}}


def test_instabox_config_template_is_valid(monkeypatch):
    monkeypatch.setenv("PIXELLE_LLM_API_KEY", "k")
    monkeypatch.delenv("PIXELLE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("PIXELLE_LLM_MODEL", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    data = load_config_dict(str(DEPLOY / "config.instabox.yaml"))
    config = PixelleVideoConfig(**data)
    assert config.llm.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.llm.api_key == "k" and config.llm.model == "gemini-flash-latest"
    # Switching provider is an env change only
    monkeypatch.setenv("PIXELLE_LLM_BASE_URL", "http://127.0.0.1:3001/v1")
    monkeypatch.setenv("PIXELLE_LLM_MODEL", "auto")
    data = load_config_dict(str(DEPLOY / "config.instabox.yaml"))
    assert data["llm"]["base_url"] == "http://127.0.0.1:3001/v1" and data["llm"]["model"] == "auto"
    assert config.api_providers.cloudflare.api_token == "tok"
    assert config.template.default_template == "1080x1920/video_fullscreen.html"


def test_units_bind_localhost_and_load_secrets():
    for unit in ("pixelle-web.service", "pixelle-api.service"):
        text = (DEPLOY / unit).read_text(encoding="utf-8")
        assert "127.0.0.1" in text and "0.0.0.0" not in text
        assert "EnvironmentFile=-/run/instabox/env/env" in text
        assert "secrets.env" in text and "MemoryMax=" in text


def test_system_chromium_path(monkeypatch):
    monkeypatch.setenv("PIXELLE_CHROMIUM_PATH", "/opt/chrome")
    assert HTMLFrameGenerator._system_chromium_path() == "/opt/chrome"
    monkeypatch.delenv("PIXELLE_CHROMIUM_PATH")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/chromium" if name == "chromium" else None)
    assert HTMLFrameGenerator._system_chromium_path() == "/usr/bin/chromium"

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

"""Edge TTS must honor the system HTTPS proxy (aiohttp ignores it by default)."""

from pixelle_video.utils.tts_util import get_edge_tts_proxy

PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


def _clear(monkeypatch):
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_no_proxy_configured(monkeypatch):
    _clear(monkeypatch)
    assert get_edge_tts_proxy() is None


def test_uses_https_proxy(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    assert get_edge_tts_proxy() == "http://proxy.local:3128"


def test_no_proxy_bypass(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("NO_PROXY", "localhost,.bing.com")
    assert get_edge_tts_proxy() is None

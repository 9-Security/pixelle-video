#!/usr/bin/env python3
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
Idempotently add the Pixelle /animate hooks to webui_bridge.py.

Usage: python3 install_bridge_hooks.py [/data/workspace/webui_bridge.py]

- Backs up the bridge to <file>.bak-<timestamp> before changing it
- Inserts three marked blocks (import, chat route, file route)
- Compiles the result before replacing the original
- Re-running is a no-op; if pixelle_animate cannot be imported the bridge
  keeps working without /animate
"""

import py_compile
import shutil
import sys
import time
from pathlib import Path

MARK = "# >>> pixelle: /animate"
END = "# <<< pixelle: /animate"
BRIDGE_DIR = str(Path(__file__).resolve().parent)

IMPORT_ANCHOR = "DEFAULT_NEGATIVE = "
CHAT_ANCHOR = "            # Check for Video generation request\n"
GET_ANCHOR = "        # 3. SD Generated image serving\n"


def patch(source: str, bridge_dir: str = BRIDGE_DIR) -> str:
    if MARK in source:
        return source
    for anchor in (IMPORT_ANCHOR, CHAT_ANCHOR, GET_ANCHOR):
        if source.count(anchor) != 1:
            raise SystemExit(f"anchor not found exactly once: {anchor.strip()!r}")
    import_block = (
        f"{MARK}\n"
        f"PIXELLE_BRIDGE_DIR = {bridge_dir!r}\n"
        "if PIXELLE_BRIDGE_DIR not in sys.path:\n"
        "    sys.path.insert(0, PIXELLE_BRIDGE_DIR)\n"
        "try:\n"
        "    import pixelle_animate\n"
        "    print(\"[Bridge] Pixelle /animate enabled.\")\n"
        "except Exception as e:\n"
        "    pixelle_animate = None\n"
        "    print(f\"[Bridge] Pixelle /animate not available: {e}\")\n"
        f"{END}\n\n"
    )
    chat_block = (
        f"            {MARK}\n"
        "            if pixelle_animate and last_user and pixelle_animate.ANIMATE_PATTERN.search(last_user):\n"
        "                pixelle_animate.handle_animate(self, last_user, data.get(\"stream\", False), stop_llm=stop_llm)\n"
        "                return\n"
        f"            {END}\n\n"
    )
    get_block = (
        f"        {MARK}\n"
        "        if pixelle_animate and pixelle_animate.serve(self, path_only):\n"
        "            return\n"
        f"        {END}\n\n"
    )
    source = source.replace(IMPORT_ANCHOR, import_block + IMPORT_ANCHOR, 1)
    source = source.replace(CHAT_ANCHOR, chat_block + CHAT_ANCHOR, 1)
    source = source.replace(GET_ANCHOR, get_block + GET_ANCHOR, 1)
    return source


def main(path: str):
    bridge = Path(path)
    original = bridge.read_text(encoding="utf-8")
    patched = patch(original)
    if patched == original:
        print("already patched")
        return
    backup = bridge.with_name(f"{bridge.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(bridge, backup)
    tmp = bridge.with_suffix(".py.patching")
    tmp.write_text(patched, encoding="utf-8")
    py_compile.compile(str(tmp), doraise=True)
    tmp.replace(bridge)
    print(f"patched OK (backup: {backup})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/data/workspace/webui_bridge.py")

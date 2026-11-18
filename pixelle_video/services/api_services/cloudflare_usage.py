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
Local estimate of Cloudflare Workers AI daily usage.

Workers AI's free plan allows a fixed number of "neurons" per UTC day. The API
does not report the remaining allowance per request, so this keeps a small
per-day counter of generated images and an ESTIMATED neuron cost, to warn before
the allowance runs out. The per-image costs are approximations derived from
public per-tile/per-step pricing and are overridable via environment variables.
"""

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

FREE_DAILY_NEURONS = int(os.environ.get("CLOUDFLARE_DAILY_NEURONS", "10000"))
WARN_RATIO = 0.8

# Approximate neurons: FLUX.1 schnell ~4.8 per 512x512 tile + ~9.6 per step (estimate)
NEURONS_PER_TILE = float(os.environ.get("CLOUDFLARE_NEURONS_PER_TILE", "4.8"))
NEURONS_PER_STEP = float(os.environ.get("CLOUDFLARE_NEURONS_PER_STEP", "9.6"))

_lock = threading.Lock()


def usage_file() -> Path:
    base = os.environ.get("PIXELLE_USAGE_DIR")
    directory = Path(base).expanduser() if base else Path.home() / ".cache" / "pixelle-video" / "usage"
    return directory / "cloudflare.json"


def estimate_neurons(width: int, height: int, steps: int) -> float:
    """Estimated neurons for one generated image of width x height at `steps` steps."""
    tiles = max(1, math.ceil(width / 512)) * max(1, math.ceil(height / 512))
    return tiles * NEURONS_PER_TILE + max(1, steps) * NEURONS_PER_STEP


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def record_image(width: int, height: int, steps: int, path: Optional[Path] = None) -> dict:
    """Add one image to today's counter and return today's totals."""
    path = path or usage_file()
    with _lock:
        data = _read(path)
        today = _today()
        day = data.setdefault(today, {"images": 0, "neurons_estimate": 0.0})
        day["images"] += 1
        day["neurons_estimate"] = round(day["neurons_estimate"] + estimate_neurons(width, height, steps), 1)
        # keep the last 14 days only
        for key in sorted(data)[:-14]:
            data.pop(key, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".usage.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    ratio = day["neurons_estimate"] / FREE_DAILY_NEURONS if FREE_DAILY_NEURONS else 0
    message = (f"Cloudflare usage today (UTC {today}): {day['images']} images, "
               f"~{day['neurons_estimate']:.0f}/{FREE_DAILY_NEURONS} neurons (estimate)")
    if ratio >= WARN_RATIO:
        logger.warning(message + " - the free daily allowance is nearly used up")
    else:
        logger.info(message)
    return dict(day, date=today, ratio=ratio)


def today_usage(path: Optional[Path] = None) -> dict:
    day = _read(path or usage_file()).get(_today(), {"images": 0, "neurons_estimate": 0.0})
    return dict(day, date=_today(), limit=FREE_DAILY_NEURONS)

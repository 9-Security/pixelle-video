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
Recognize provider content-policy / safety-filter rejections.

Used only to CLASSIFY an error so a person can be told what happened; it does
not alter prompts or retry around a filter.
"""

from typing import Iterable, Optional

# Markers used by DashScope / Alibaba content inspection (shared with api_media)
CONTENT_INSPECTION_MARKERS = (
    "datainspectionfailed",
    "inappropriate content",
    "green net check failed",
    "content inspection",
    "safety inspection",
    "risk control",
)

# Additional provider wordings for safety rejections
CONTENT_POLICY_MARKERS = CONTENT_INSPECTION_MARKERS + (
    "nsfw",                        # Cloudflare Workers AI (FLUX)
    "content_policy_violation",    # OpenAI images
    "safety system",               # OpenAI images
    "sensitivecontent",            # Volcengine ARK / Seedream
    "sensitive content",
    "content moderation",
)


def message_matches(message: str, markers: Iterable[str]) -> bool:
    lowered = (message or "").lower()
    return any(marker in lowered for marker in markers)


def is_content_policy_rejection(error: Optional[BaseException]) -> bool:
    """True if the error, or any error in its cause chain, is a safety-filter rejection."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if getattr(error, "safety_blocked", False) or message_matches(str(error), CONTENT_POLICY_MARKERS):
            return True
        error = error.__cause__ or error.__context__
    return False

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
Procedural, seamlessly looping "satisfying" videos for 24/7 live streams.

    python -m pixelle_video.loops list
    python -m pixelle_video.loops render all --out output/loops
    python -m pixelle_video.loops program all --out output/loops --minutes 30
"""

from pixelle_video.loops.base import LoopScene, SceneInfo, get_scene, list_scenes, register

__all__ = ["LoopScene", "SceneInfo", "get_scene", "list_scenes", "register"]

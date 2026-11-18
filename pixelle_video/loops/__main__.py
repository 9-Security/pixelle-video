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

"""Command line: list scenes, render one-period clips, build long stream programs."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from pixelle_video.loops.base import list_scenes
from pixelle_video.loops.render import EncodeSettings, build_program, render_clip


def _parse_size(value: str):
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1920x1080") from None
    return w, h


def _scene_names(arg: str):
    names = [i.name for i in list_scenes()]
    if arg == "all":
        return names
    chosen = [n.strip() for n in arg.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in names]
    if unknown:
        raise SystemExit(f"Unknown scene(s): {', '.join(unknown)}. Available: {', '.join(names)}")
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pixelle_video.loops", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List available loop scenes")
    for name in ("render", "program"):
        p = sub.add_parser(name, help="Render one-period clips" if name == "render" else
                           "Render (if needed) and build long programs for streaming")
        p.add_argument("scenes", help="Scene name, comma-separated names or 'all'")
        p.add_argument("--out", default="output/loops", help="Output directory")
        p.add_argument("--size", type=_parse_size, default=(1920, 1080))
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--crf", type=int, default=23)
        p.add_argument("--maxrate", default="3500k")
        p.add_argument("--preset", default="medium")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--dither", type=float, default=None,
                       help="Grain strength against gradient banding (default 1.5; 0 = off, lower bitrate)")
        if name == "program":
            p.add_argument("--minutes", type=float, default=30.0, help="Approximate length of each program")
            p.add_argument("--rerender", action="store_true", help="Render clips even if they exist")
    args = parser.parse_args(argv)

    if args.command == "list":
        for info in list_scenes():
            print(f"{info.name:16s} {info.period:3d}s  {info.kind:9s} {info.title_zh} / {info.title_en}  - "
                  f"{info.description}")
        return 0

    settings = EncodeSettings(width=args.size[0], height=args.size[1], fps=args.fps, crf=args.crf,
                              maxrate=args.maxrate, preset=args.preset)
    out = Path(args.out)
    failed = []
    for scene in _scene_names(args.scenes):
        try:
            clip_meta = out / scene / "meta.json"
            if args.command == "render" or args.rerender or not clip_meta.exists():
                render_clip(scene, str(out), settings, seed=args.seed, dither=args.dither)
            if args.command == "program":
                build_program(str(out / scene), str(out / "programs" / f"{scene}.mp4"), args.minutes)
        except Exception as e:  # keep going with the other scenes
            logger.error(f"{scene}: {e}")
            failed.append(scene)
    if failed:
        logger.error(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

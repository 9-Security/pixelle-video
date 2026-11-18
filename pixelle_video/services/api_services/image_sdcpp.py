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
Local text-to-image with stable-diffusion.cpp (`sd-cli`) on CPU.

Free and quota-less, but slow (~1 min per 512x896 image on 4 vCPUs with the
TAESD decoder), so it is meant as a fallback when a hosted provider's daily
allowance is used up. Each call runs one `sd-cli` process; nothing stays
resident in memory between images.
"""

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from pixelle_video.services.api_services.image_cloudflare import fit_image, request_size

DEFAULT_NEGATIVE = "ugly, deformed, disfigured, bad anatomy, blurry, low quality, artifacts, text, watermark, signature"


@dataclass(frozen=True)
class LocalSDModel:
    filename: str
    sampler: str
    cfg_scale: float
    steps: int


LOCAL_SD_MODELS: Dict[str, LocalSDModel] = {
    "dreamshaper-8-lcm": LocalSDModel("DreamShaper8_LCM_q8_0.gguf", "lcm", 1.5, 4),
    "sd-turbo": LocalSDModel("sd_turbo-f16-q8_0.gguf", "euler_a", 1.0, 2),
}


class LocalSDError(RuntimeError):
    """sd-cli is missing, misconfigured or failed."""


class SdCppImageClient:
    """Runs `sd-cli` for one image and fits the result to the requested size."""

    def __init__(self, sd_dir: str, binary: str = "sd-cli", taesd: Optional[str] = "taesd_full.safetensors",
                 threads: int = 4, long_side: int = 896, timeout: float = 600.0):
        self.sd_dir = Path(sd_dir).expanduser()
        self.binary = binary
        self.taesd = taesd
        self.threads = max(1, int(threads))
        self.long_side = long_side
        self.timeout = timeout

    def build_command(self, model: str, prompt: str, output: str, width: Optional[int], height: Optional[int],
                      seed: Optional[int] = None, negative_prompt: Optional[str] = None) -> list:
        spec = LOCAL_SD_MODELS.get(model)
        if spec is None:
            raise LocalSDError(f"Unknown local SD model '{model}'. Available: {', '.join(LOCAL_SD_MODELS)}")
        binary = self.sd_dir / self.binary
        weights = self.sd_dir / spec.filename
        for path in (binary, weights):
            if not path.is_file():
                raise LocalSDError(f"Local SD file not found: {path}")
        gen_w, gen_h = request_size(width, height, long_side=self.long_side)
        cmd = [
            str(binary), "-m", str(weights),
            "-p", prompt, "-n", negative_prompt or DEFAULT_NEGATIVE,
            "--sampling-method", spec.sampler, "--cfg-scale", str(spec.cfg_scale), "--steps", str(spec.steps),
            "-W", str(gen_w), "-H", str(gen_h),
            "--seed", str(seed if seed is not None else random.randint(0, 2**31 - 1)),
            "-t", str(self.threads), "-o", output,
        ]
        if self.taesd and (self.sd_dir / self.taesd).is_file():
            cmd[3:3] = ["--taesd", str(self.sd_dir / self.taesd)]
        return cmd

    def generate_image(self, prompt: str, output_path: str, model: str = "dreamshaper-8-lcm",
                       width: Optional[int] = None, height: Optional[int] = None,
                       seed: Optional[int] = None, negative_prompt: Optional[str] = None) -> str:
        """Blocking: generate one image, fit it to width x height and save PNG at output_path."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        raw = out.with_name(out.stem + ".sdraw.png")
        cmd = self.build_command(model, prompt, str(raw), width, height, seed, negative_prompt)
        logger.info(f"Local SD image: model={model} size={cmd[cmd.index('-W') + 1]}x{cmd[cmd.index('-H') + 1]} "
                    f"(target {width}x{height}); this takes about a minute on CPU")
        try:
            proc = subprocess.run(cmd, cwd=self.sd_dir, env={"LD_LIBRARY_PATH": str(self.sd_dir), "PATH": "/usr/bin:/bin"},
                                  capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise LocalSDError(f"sd-cli timed out after {self.timeout:.0f}s") from e
        if proc.returncode != 0 or not raw.is_file():
            tail = (proc.stderr or proc.stdout or "")[-400:]
            raise LocalSDError(f"sd-cli failed (exit {proc.returncode}): {tail}")
        try:
            image = fit_image(raw.read_bytes(), width, height)
            if out.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                out = out.with_suffix(".png")
            image.save(out)
        finally:
            raw.unlink(missing_ok=True)
        logger.success(f"Local SD image saved: {out} ({image.size[0]}x{image.size[1]})")
        return str(out.resolve())

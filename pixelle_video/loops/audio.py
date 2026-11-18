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
Seamless, copyright-free procedural audio for loop scenes.

- Ambience is noise synthesised in the frequency domain (random phases, shaped
  spectrum, inverse FFT), so the buffer is exactly periodic: its end flows into
  its start with no click.
- Sound events (drips, clacks, chimes...) are added with circular wrap-around,
  so an event near the end of the period continues at the start.
"""

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 48000


@dataclass
class NoiseLayer:
    color: str = "pink"  # white | pink | brown
    gain: float = 0.05
    band: Tuple[float, float] = (40.0, 12000.0)  # Hz
    # Optional periodic gain envelope: f(times_in_seconds) -> gain multiplier
    envelope: Optional[Callable[[np.ndarray], np.ndarray]] = None


@dataclass
class SoundEvent:
    time: float  # seconds within the period
    kind: str  # key of SYNTHS
    gain: float = 0.3
    pitch: float = 1.0
    pan: float = 0.0  # -1 left .. 1 right


@dataclass
class SoundDesign:
    layers: List[NoiseLayer] = field(default_factory=lambda: [NoiseLayer("brown", 0.03, (30.0, 800.0))])
    events: List[SoundEvent] = field(default_factory=list)


def periodic_noise(n: int, sr: int, color: str, band: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Exactly periodic band-limited noise of n samples, normalised to unit RMS."""
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    amp = np.zeros_like(freqs)
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    f = np.maximum(freqs[mask], 1.0)
    if color == "white":
        amp[mask] = 1.0
    elif color == "pink":
        amp[mask] = 1.0 / np.sqrt(f)
    elif color == "brown":
        amp[mask] = 1.0 / f
    else:
        raise ValueError(f"Unknown noise color '{color}'")
    # Soft band edges avoid ringing
    edge = np.clip((freqs - lo) / max(lo * 0.5, 1.0), 0, 1) * np.clip((hi * 1.25 - freqs) / (hi * 0.25), 0, 1)
    amp *= edge
    spectrum = amp * np.exp(2j * np.pi * rng.random(len(freqs)))
    spectrum[0] = 0
    signal = np.fft.irfft(spectrum, n)
    rms = np.sqrt(np.mean(signal ** 2))
    return signal / rms if rms > 0 else signal


# ---------------------------------------------------------------------------
# Event synthesisers: each returns a mono float32 buffer (peak ~1)
# ---------------------------------------------------------------------------

def _t(duration: float, sr: int) -> np.ndarray:
    return np.arange(int(duration * sr)) / sr


def synth_drip(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(0.18, sr)
    f0 = 900 * pitch
    freq = f0 * (1 + 1.2 * np.exp(-t * 40))  # quick upward-to-settle "plip"
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 28) * (1 - np.exp(-t * 3000))


def synth_clack(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(0.09, sr)
    noise = rng.standard_normal(len(t))
    tone = np.sin(2 * np.pi * 1800 * pitch * t) + 0.6 * np.sin(2 * np.pi * 3100 * pitch * t)
    return (0.5 * noise * np.exp(-t * 160) + tone * np.exp(-t * 70)) * 0.7


def synth_chime(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(3.5, sr)
    f = 660 * pitch
    partials = [(1.0, 1.0), (2.76, 0.35), (5.4, 0.12)]
    sig = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-t * (1.1 + m * 0.6)) for m, a in partials)
    return sig * (1 - np.exp(-t * 400)) * 0.6


def synth_tap(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(0.12, sr)
    return (np.sin(2 * np.pi * 220 * pitch * t) * np.exp(-t * 45) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t * 120))


def synth_lap(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(0.16, sr)
    noise = rng.standard_normal(len(t))
    # crude band-pass: difference of two smoothings
    k1, k2 = 6, 40
    lp1 = np.convolve(noise, np.ones(k1) / k1, mode="same")
    lp2 = np.convolve(noise, np.ones(k2) / k2, mode="same")
    env = np.sin(np.pi * np.clip(t / 0.16, 0, 1)) ** 2
    return (lp1 - lp2) * env * 2.5 * pitch


def synth_thud(sr: int, pitch: float, rng: np.random.Generator) -> np.ndarray:
    t = _t(0.25, sr)
    return np.sin(2 * np.pi * 95 * pitch * t * (1 - 0.3 * t)) * np.exp(-t * 18)


SYNTHS: Dict[str, Callable[[int, float, np.random.Generator], np.ndarray]] = {
    "drip": synth_drip,
    "clack": synth_clack,
    "chime": synth_chime,
    "tap": synth_tap,
    "lap": synth_lap,
    "thud": synth_thud,
}


def render_audio(design: SoundDesign, period: float, sr: int = SAMPLE_RATE, seed: int = 7,
                 peak: float = 0.5) -> np.ndarray:
    """Render one exactly-periodic stereo buffer (n, 2) float32 for the design."""
    n = int(round(period * sr))
    if n <= 0:
        raise ValueError("period must be positive")
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 2), dtype=np.float64)
    times = np.arange(n) / sr

    for layer in design.layers:
        # Independent noise per channel gives width; both are periodic
        for ch in range(2):
            sig = periodic_noise(n, sr, layer.color, layer.band, rng) * layer.gain
            if layer.envelope is not None:
                sig = sig * layer.envelope(times)
            out[:, ch] += sig

    for ev in design.events:
        synth = SYNTHS.get(ev.kind)
        if synth is None:
            raise ValueError(f"Unknown sound event '{ev.kind}'")
        sample = synth(sr, ev.pitch, rng) * ev.gain
        start = int(round((ev.time % period) * sr))
        idx = (start + np.arange(len(sample))) % n  # circular wrap keeps the loop seamless
        left = np.sqrt(0.5 * (1 - ev.pan))
        right = np.sqrt(0.5 * (1 + ev.pan))
        np.add.at(out[:, 0], idx, sample * left)
        np.add.at(out[:, 1], idx, sample * right)

    max_abs = np.max(np.abs(out))
    if max_abs > peak:
        out *= peak / max_abs
    return out.astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Write a stereo float buffer as 16-bit PCM WAV."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1] if pcm.ndim == 2 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return str(path)

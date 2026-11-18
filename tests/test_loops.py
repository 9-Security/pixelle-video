"""Tests for seamless loop scenes, periodic audio and clip/program rendering."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from pixelle_video.loops import audio as loop_audio
from pixelle_video.loops.__main__ import main as cli_main
from pixelle_video.loops.audio import (
    NoiseLayer,
    SoundDesign,
    SoundEvent,
    periodic_noise,
    render_audio,
)
from pixelle_video.loops.base import SCENES, LoopScene, SceneInfo, get_scene, list_scenes
from pixelle_video.loops.render import EncodeSettings, build_program, check_seam, render_clip

SCENE_NAMES = sorted(i.name for i in list_scenes())
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_builtin_scenes_registered():
    assert {"cat_drinking", "cup_on_edge", "domino_forever", "marble_orbit", "pendulum_wave",
            "rain_ripples"} <= set(SCENE_NAMES)
    for info in list_scenes():
        assert info.period % 2 == 0, "period must align with the 2 s keyframe interval"
        assert info.title_zh and info.title_en and info.kind in ("soothing", "suspense")


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_is_periodic_and_seamless(name):
    scene = get_scene(name)(320, 180, seed=3)
    first = scene.render(0.0)
    assert first.shape == (180, 320, 3) and first.dtype == np.uint8
    assert np.array_equal(first, scene.render(float(scene.period)))
    assert np.array_equal(scene.render(1.234), scene.render(1.234 + 2 * scene.period))
    report = check_seam(scene, fps=30, samples=4)
    assert report.ok, report


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_moves(name):
    scene = get_scene(name)(320, 180)
    # Fractions avoid multiples of a lap (e.g. marble_orbit repeats 6 identical laps per period)
    frames = [scene.render(scene.period * f) for f in (0.07, 0.29, 0.53, 0.71)]
    assert any(not np.array_equal(frames[0], f) for f in frames[1:])


def test_scene_supports_portrait_frames():
    scene = get_scene("cat_drinking")(180, 320)
    assert scene.render(0.5).shape == (320, 180, 3)


def test_frame_size_validation():
    with pytest.raises(ValueError):
        get_scene("rain_ripples")(100, 50)


def test_unknown_scene():
    with pytest.raises(KeyError, match="Available"):
        get_scene("nope")


def test_periodic_noise_wraps_smoothly():
    rng = np.random.default_rng(1)
    sig = periodic_noise(48000, 48000, "pink", (40.0, 8000.0), rng)
    steps = np.abs(np.diff(sig))
    assert abs(sig[0] - sig[-1]) <= np.percentile(steps, 99.9)
    assert np.isclose(np.sqrt(np.mean(sig ** 2)), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        periodic_noise(100, 48000, "purple", (40.0, 8000.0), rng)


def test_events_wrap_around_the_period():
    design = SoundDesign(layers=[], events=[SoundEvent(0.99, "chime", 0.5)])
    out = render_audio(design, period=1.0, sr=8000)
    assert out.shape == (8000, 2) and out.dtype == np.float32
    assert np.abs(out[:400]).max() > 0.01  # the chime tail continues at the start
    with pytest.raises(ValueError):
        render_audio(SoundDesign(layers=[], events=[SoundEvent(0.1, "boom")]), 1.0, sr=8000)


def test_audio_peak_is_limited_and_envelope_applied():
    design = SoundDesign(layers=[NoiseLayer("white", 5.0, (100.0, 3000.0), envelope=lambda t: (t < 0.5) * 1.0)])
    out = render_audio(design, period=1.0, sr=8000, peak=0.5)
    assert np.abs(out).max() <= 0.5 + 1e-6
    assert np.abs(out[4100:]).max() == 0


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_scene_sound_designs_render(name):
    scene = get_scene(name)(320, 180)
    out = render_audio(scene.sound_design(), scene.period, sr=8000)
    assert out.shape[0] == scene.period * 8000 and np.abs(out).max() > 0


def test_write_wav(tmp_path):
    path = loop_audio.write_wav(str(tmp_path / "a.wav"), np.zeros((800, 2), np.float32), sr=8000)
    import wave

    with wave.open(path) as wf:
        assert (wf.getnchannels(), wf.getframerate(), wf.getnframes()) == (2, 8000, 800)


def test_encode_settings_validation():
    with pytest.raises(ValueError):
        EncodeSettings(width=321).validate()
    with pytest.raises(ValueError):
        EncodeSettings(fps=5).validate()


def test_period_must_align_with_keyframes(tmp_path, monkeypatch):
    class Odd(LoopScene):
        info = SceneInfo("odd_test", "測試", "Test", "odd period", period=3)

        def setup(self):
            pass

        def render(self, t):
            return np.zeros((self.height, self.width, 3), np.uint8)

    monkeypatch.setitem(SCENES, "odd_test", Odd)
    with pytest.raises(ValueError, match="multiple"):
        render_clip("odd_test", str(tmp_path), EncodeSettings(width=320, height=180, fps=10))


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@needs_ffmpeg
def test_render_clip_and_program(tmp_path):
    settings = EncodeSettings(width=320, height=180, fps=10, preset="ultrafast")
    meta = render_clip("cup_on_edge", str(tmp_path), settings, seed=5)
    clip_dir = tmp_path / "cup_on_edge"
    assert (clip_dir / "clip.mp4").exists() and (clip_dir / "clip.wav").exists()
    saved = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
    assert saved == meta and meta["period"] == 16 and meta["seam"]["ok"]
    info = _probe(clip_dir / "clip.mp4")
    assert abs(float(info["format"]["duration"]) - 16) < 0.2
    codecs = {s["codec_name"] for s in info["streams"]}
    assert codecs == {"h264", "aac"}

    program = build_program(str(clip_dir), str(tmp_path / "programs" / "cup.mp4"), minutes=0.5)
    pinfo = _probe(program)
    assert abs(float(pinfo["format"]["duration"]) - 32) < 0.3  # whole periods: 2 x 16 s
    assert not list((tmp_path / "programs").glob("*.tmp*"))


@needs_ffmpeg
def test_cli_list_render_and_program(tmp_path, capsys):
    assert cli_main(["list"]) == 0
    assert "cat_drinking" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main(["render", "nope", "--out", str(tmp_path)])
    args = ["--out", str(tmp_path), "--size", "320x180", "--fps", "10", "--preset", "ultrafast"]
    assert cli_main(["program", "cup_on_edge", "--minutes", "0.2", *args]) == 0
    assert (tmp_path / "programs" / "cup_on_edge.mp4").exists()
    assert cli_main(["render", "cup_on_edge", "--size", "321x180", "--out", str(tmp_path)]) == 1


def test_dither_is_static_small_and_optional():
    scene = get_scene("rain_ripples")(320, 180, seed=2)
    raw = scene.render(3.0)
    framed = scene.frame(3.0)
    diff = framed.astype(int) - raw.astype(int)
    assert np.abs(diff).max() <= 3 and np.abs(diff).mean() > 0.3
    assert np.array_equal(scene.frame(0.0), scene.frame(float(scene.period)))  # still periodic
    scene.dither_strength = 0
    scene._dither = scene._make_dither()
    assert np.array_equal(scene.frame(3.0), raw)


@needs_ffmpeg
def test_render_clip_records_dither_and_rejects_negative(tmp_path):
    settings = EncodeSettings(width=320, height=180, fps=10, preset="ultrafast")
    meta = render_clip("cup_on_edge", str(tmp_path), settings, dither=0)
    assert meta["dither"] == 0
    with pytest.raises(ValueError):
        render_clip("cup_on_edge", str(tmp_path), settings, dither=-1)


def test_dominoes_never_overlap():
    import cv2

    from pixelle_video.loops.draw import rotated_rect

    sc = get_scene("domino_forever")(640, 360)
    for t in np.linspace(0, sc.period, 61):
        p = t % sc.period
        angles, nxt = {}, sc._angle(p - 31 * sc.step_seconds)
        for k in range(30, -1, -1):
            nxt = min(sc._angle(p - k * sc.step_seconds), float(np.interp(nxt, sc._next_grid, sc._limit)))
            angles[k] = nxt
        for k in range(30):
            polys = [rotated_rect((j * sc.d, -sc.dh / 2), sc.dw, sc.dh, angles[j], (j * sc.d + sc.dw / 2, 0))
                     .astype(np.float32) for j in (k, k + 1)]
            area, _ = cv2.intersectConvexConvex(polys[0], polys[1])
            assert area < 1.0

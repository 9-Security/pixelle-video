"""Tests for the stdlib loop streamer (deploy/livestream/streamer.py)."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("loop_streamer", ROOT / "deploy" / "livestream" / "streamer.py")
streamer = importlib.util.module_from_spec(_spec)
sys.modules["loop_streamer"] = streamer
_spec.loader.exec_module(streamer)

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
KEY = "test-key-9f8e-7d6c"


def _make_program(path: Path, size: str = "160x90", seconds: int = 2) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(seconds),
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ac", "2", str(path)], check=True)
    return path


def test_redactor_hides_key():
    red = streamer.Redactor([KEY, "", "abc"])
    assert red(f"rtmps://host/live2/{KEY} failed") == "rtmps://host/live2/*** failed"
    assert red("abc") == "abc"  # too-short secrets are ignored (avoid mangling logs)


def test_build_command_network_and_file(tmp_path):
    playlist = tmp_path / "p.ffconcat"
    net = streamer.build_command("ffmpeg", playlist, "rtmps://x/live2/k", True, None)
    assert net[-5:] == ["-f", "flv", "-flvflags", "no_duration_filesize", "rtmps://x/live2/k"]
    assert ["-c", "copy"] == net[net.index("-c"):net.index("-c") + 2]
    assert "-re" in net and "-stream_loop" in net and "-t" not in net
    local = streamer.build_command("ffmpeg", playlist, "/tmp/o.flv", False, 5)
    assert local[-2:] == ["-y", "/tmp/o.flv"] and "-t" in local


def test_write_playlist_escapes_quotes(tmp_path):
    files = [tmp_path / "a'b.mp4", tmp_path / "c.mp4"]
    out = tmp_path / "list.ffconcat"
    streamer.write_playlist(files, out, shuffle=False, rng=streamer.random.Random(0))
    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "ffconcat version 1.0"
    assert text[1].endswith("a'\\''b.mp4'") and text[2].endswith("c.mp4'")


@needs_ffmpeg
def test_scan_library_skips_incompatible_and_temp(tmp_path):
    _make_program(tmp_path / "a.mp4")
    _make_program(tmp_path / "b.mp4")
    _make_program(tmp_path / "c.mp4", size="320x180")
    _make_program(tmp_path / "d.tmp.mp4")
    (tmp_path / "broken.mp4").write_bytes(b"not a video")
    found = streamer.scan_library(tmp_path, "ffprobe")
    assert [p.name for p in found] == ["a.mp4", "b.mp4"]


def test_main_requires_key_or_output(tmp_path, monkeypatch):
    monkeypatch.delenv(streamer.KEY_ENV, raising=False)
    assert streamer.main(["--library", str(tmp_path)]) == 2
    assert streamer.main(["--library", str(tmp_path / "missing"), "--output", "x.flv"]) == 2
    monkeypatch.delenv("STREAM_LIBRARY", raising=False)
    assert streamer.main([]) == 2


@needs_ffmpeg
def test_check_mode(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(streamer.KEY_ENV, KEY)
    assert streamer.main(["--library", str(tmp_path), "--check"]) == 1  # empty library
    _make_program(tmp_path / "a.mp4")
    caplog.set_level("INFO")
    assert streamer.main(["--library", str(tmp_path), "--check"]) == 0
    assert KEY not in caplog.text and "***" in caplog.text


@needs_ffmpeg
def test_streams_to_file_and_writes_status(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    _make_program(lib / "a.mp4")
    _make_program(lib / "b.mp4")
    out = tmp_path / "out.flv"
    state = tmp_path / "state"
    code = streamer.main(["--library", str(lib), "--output", str(out), "--duration", "3",
                          "--state-dir", str(state)])
    assert code == 0 and out.stat().st_size > 0
    status = json.loads((state / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "finished" and status["programs"] == 2
    assert (state / "playlist.ffconcat").exists()


@needs_ffmpeg
def test_failed_connection_restarts_and_never_logs_key(tmp_path, monkeypatch, caplog):
    lib = tmp_path / "lib"
    lib.mkdir()
    _make_program(lib / "a.mp4")
    monkeypatch.setenv(streamer.KEY_ENV, KEY)
    args = streamer.parse_args(["--library", str(lib), "--ingest", "rtmp://127.0.0.1:1/live2",
                                "--state-dir", str(tmp_path / "state")])
    s = streamer.Streamer(args, KEY)
    calls = []
    original = s.run_once

    def run_once_and_stop_after_two():
        calls.append(1)
        code = original()
        if len(calls) >= 2:
            s.stop_event.set()
        return code

    s.run_once = run_once_and_stop_after_two
    monkeypatch.setattr(streamer, "BACKOFF", [0, 0, 0])
    caplog.set_level("INFO")
    assert s.run() == 0
    assert len(calls) == 2
    status_text = (tmp_path / "state" / "status.json").read_text(encoding="utf-8")
    assert KEY not in caplog.text and KEY not in status_text
    status = json.loads(status_text)
    assert status["restarts"] >= 1 and status["state"] == "stopped" and status["last_error"]

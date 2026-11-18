# 24/7 loop live stream

Stream calm, seamlessly looping videos (a cat lapping water, a marble flicked
round a track, endless dominoes...) to YouTube Live around the clock.

```
pixelle_video/loops        render loops  ->  clip.mp4 + clip.wav (one period, seam-checked)
                           build programs -> programs/<scene>.mp4 (30 min, stream-ready)
deploy/livestream          streamer.py: ffmpeg -c copy -> RTMPS -> YouTube (restarts, key redaction)
```

Rendering needs the Pixelle-Video Python environment (numpy, OpenCV, loguru).
The streaming host only needs `python3` (stdlib) and `ffmpeg`: programs can be
rendered anywhere and copied over.

## 1. Render the loops

```bash
python -m pixelle_video.loops list
python -m pixelle_video.loops program all --out output/loops --minutes 30
```

| Scene | Period | Type |
|---|---|---|
| `cat_drinking` 貓咪喝水 | 20 s | soothing |
| `marble_orbit` 手指推彈珠 | 30 s | soothing |
| `rain_ripples` 雨落池塘 | 40 s | soothing |
| `pendulum_wave` 擺錘波 | 60 s | soothing |
| `domino_forever` 永遠倒不完的骨牌 | 30 s | suspense |
| `cup_on_edge` 杯子到底會不會掉 | 16 s | suspense (ends in a blink before the answer) |

Every scene is a periodic function of time, so frame `t` equals frame
`t + period`; `meta.json` records a seam check (change across the loop point
vs. ordinary frame-to-frame change). Sound is procedural (no copyright claims):
frequency-domain noise that is exactly periodic plus synthesised events wrapped
around the period.

Defaults: 1920x1080, 30 fps, H.264 CRF 23 capped at 3.5 Mbit/s, keyframe every
2 s (YouTube requires 4 s or less), AAC 128 kbit/s 48 kHz. Measured bitrates are
0.5-1.7 Mbit/s because the scenes are simple. Use `--size 1280x720` to halve
bandwidth again.

## 2. YouTube setup (once)

1. YouTube Studio -> **Go live**. A channel's first live stream needs phone
   verification and can take up to 24 hours to be enabled.
2. **Stream** tab: create a **persistent stream key**, set latency to *Normal*,
   enable **Auto-start** and disable **Auto-stop** (so a short reconnect resumes the same broadcast).
3. Mark the stream "Not made for kids" or as appropriate, and add the
   "altered or synthetic content" disclosure if you later add realistic AI footage.
4. Test with visibility **Unlisted** first.

Note: YouTube does not keep an archive of streams longer than 12 hours, so a
24/7 stream has no replay.

## 3. Install the streamer

```bash
sudo bash deploy/livestream/install.sh --library /srv/loop-streamer/programs
sudo cp output/loops/programs/*.mp4 /srv/loop-streamer/programs/
sudo chmod 644 /srv/loop-streamer/programs/*.mp4
# put the key in the env file without echoing it or saving it in shell history:
read -rsp "Stream key: " K && sudo sed -i "s#^YOUTUBE_STREAM_KEY=.*#YOUTUBE_STREAM_KEY=$K#" /etc/loop-streamer/stream.env; unset K; echo
sudo bash -c 'set -a; . /etc/loop-streamer/stream.env; python3 /opt/loop-streamer/streamer.py --check'
sudo systemctl start loop-streamer
journalctl -u loop-streamer -f
```

Local test without a key: `streamer.py --library DIR --output /tmp/test.flv --duration 60`.

Behaviour:

- Programs are shuffled and looped forever with `-re -stream_loop -1 -c copy`
  (CPU use is a few percent of one core).
- Files whose codec settings differ from the first program are skipped.
- On exit ffmpeg restarts after 5-60 s (exponential, reset after 5 healthy minutes);
  each restart rescans the library. `--rotate-hours N` restarts on purpose to pick up new programs.
- The key never appears in logs or in `status.json` (`/var/lib/loop-streamer/`).
  It is still visible in the ffmpeg process arguments to local users of the host,
  so use a host you do not share.
- Known artefact: about 20 ms of audio overlap where one program file ends and the next
  begins (AAC encoder padding), i.e. once every 30 minutes with the default program length.

## 4. Choosing a host

Bandwidth: at ~1 Mbit/s a stream sends about 0.45 GB per hour, roughly
330 GB per month (1080p at the 3.5 Mbit/s cap: up to ~1.2 TB/month).

| Host | Fits 24/7? | Notes |
|---|---|---|
| instabox **Free** | No | 100 GiB egress/month; a run is paused after at most 480 min, then a 5-20 min cooldown. Fine for rendering programs. |
| instabox Pro | Yes | $20/month plus usage (a 2 vCPU / 4 GiB machine running all month is roughly $12 of CPU+RAM); 1 TiB egress included. |
| Oracle Cloud Always Free (Arm VM) | Yes | Free tier with generous egress; needs card verification. Check current limits before relying on it. |
| Home PC / mini PC / Raspberry Pi 4+ | Yes | Free if always on; needs ~2 Mbit/s spare upload. |

## Policy notes

- 24/7 looping streams are common (lo-fi, ambience); keep the content original
  and copyright-free (these scenes and sounds are generated from code).
- Monetisation is not a goal here; if that changes, YouTube's "reused /
  repetitive content" policy applies to looping channels.

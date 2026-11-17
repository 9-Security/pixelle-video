# Animation Generator

The **Animation** pipeline turns a story idea (or an existing story) into a narrated animated short.

## How it works

```
Story idea ──► LLM: structured animation script
                 ├─ character sheet (fixed English appearance per character)
                 └─ scenes: narration / visual / characters / motion / camera
             ──► per-scene prompt = style preset + scene visual + character appearances
             ──► per scene:
                   TTS narration
                   camera mode : text-to-image ─► Ken Burns camera move (zoom / pan)
                   ai_video    : text-to-video (prompt + motion + camera hints)
                   subtitle template overlay
             ──► concatenate + optional BGM ──► final.mp4 + animation_script.json
```

Character consistency comes from repeating the full character description and a fixed style
prefix in every scene prompt, so it works with any image model without LoRA or reference images.

## Modes

| Option | Values | Notes |
|---|---|---|
| `mode` | `generate` / `adapt` | `generate`: write a new story from an idea. `adapt`: split your story into scenes and keep your wording. |
| `motion_mode` | `camera` / `ai_video` | `camera` works with any image workflow and is cheapest. `ai_video` needs a video workflow and a `video_*` template. |
| `art_style` | `anime`, `3d_cartoon`, `watercolor`, `chinese_ink`, `pixel_art`, `clay`, `flat_vector`, `comic` | `custom_style` overrides it. |
| `camera_motion` | `auto`, `zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `pan_up`, `pan_down`, `static` | `auto` uses the camera chosen by the LLM per scene. |
| `camera_intensity` | 0.0 – 0.5 | Default 0.15. Halved automatically for `image_*` templates because their subtitles move with the camera. |

`video_*` templates are recommended: the image is animated underneath and subtitles stay fixed.
The default `video_fullscreen.html` (1080x1920, also 1920x1080) fills the whole frame with the animation
and adds top/bottom scrims for readable title and subtitles (`title_size`, `subtitle_size`, `accent_color` are adjustable).

Each character has a `name` (ASCII, used in English image prompts) and a `display_name` (used in narration,
in the story's language). Any ASCII name that leaks into narration is replaced with the display name.

## Free setup: Gemini + Cloudflare Workers AI

1. **LLM** – Settings → LLM preset **Gemini**, paste a Google AI Studio API key
   (base URL `https://generativelanguage.googleapis.com/v1beta/openai/`, model `gemini-flash-latest`).
2. **Images** – Settings → API Media Models → **Cloudflare Workers AI**: fill in the Account ID and an
   API token with *Workers AI* permission. Then choose source **API models** and
   `flux-1-schnell - API Cloudflare` in the Animation tab (workflow key `api/cloudflare/flux-1-schnell`).
3. **Voice** – local Edge-TTS needs no key.

| Workflow key | Model | Notes |
|---|---|---|
| `api/cloudflare/flux-1-schnell` | `@cf/black-forest-labs/flux-1-schnell` | Best quality; fixed square output, center-cropped to the template size |
| `api/cloudflare/sdxl-lightning` | `@cf/bytedance/stable-diffusion-xl-lightning` | Fast; native width/height |
| `api/cloudflare/sdxl-base-1.0` | `@cf/stabilityai/stable-diffusion-xl-base-1.0` | Slower; native width/height |
| `api/cloudflare/dreamshaper-8-lcm` | `@cf/lykon/dreamshaper-8-lcm` | Native width/height |

Rate limits (HTTP 429) and 5xx errors are retried with exponential backoff; 401/403 fail fast.
Check Cloudflare's current free allowance before large batches.

## Web UI

Open the **🎞️ Animation** tab, enter a story idea, choose an art style, an animation mode, an
image (or video) workflow and a voice, then click **Generate Animation**. The generated script
(characters and scenes) is shown under the video and can be downloaded as JSON.

## API

```bash
curl -X POST http://localhost:8000/api/animation/generate/async \
  -H "Content-Type: application/json" \
  -d '{
        "text": "A little fox afraid of the dark sets out to bring the moon back",
        "n_scenes": 5,
        "art_style": "watercolor",
        "motion_mode": "camera",
        "media_workflow": "api/cloudflare/flux-1-schnell",
        "frame_template": "1080x1920/video_fullscreen.html",
        "tts_voice": "en-US-AriaNeural"
      }'
```

Poll `GET /api/tasks/{task_id}`; the result contains `video_url` and `script_url`.
`GET /api/animation/styles` lists the style presets.

## Python

```python
from pixelle_video.service import PixelleVideoCore

core = PixelleVideoCore()
await core.initialize()
result = await core.generate_video(
    text="A little fox afraid of the dark sets out to bring the moon back",
    pipeline="animation",
    n_scenes=5,
    art_style="watercolor",
    media_workflow="runninghub/image_flux.json",
    frame_template="1080x1920/video_fullscreen.html",
)
print(result.video_path)
```

## Output

`output/<task_id>/` contains `final.mp4`, `frames/`, `storyboard.json`, `metadata.json` and
`animation_script.json` (script, style prompt, camera moves and final per-scene prompts) for
reproducibility.

# Deploy Pixelle-Video on instabox

Runs the Web UI and API as systemd services on an instabox machine
(Debian 13, 4 vCPU / 8 GiB, 20 GiB `/data`), next to other local services
such as freellmapi and llama-server.

| Component | Where | Exposure |
|---|---|---|
| Web UI (Streamlit) | `127.0.0.1:8501` | instabox **private** port (token link) |
| REST API (FastAPI, no auth) | `127.0.0.1:8000` | localhost only (e.g. for `webui_bridge.py`) |
| LLM | Google Gemini (default) or freellmapi `127.0.0.1:3001/v1` | outbound HTTPS / localhost |
| Images | Cloudflare Workers AI | outbound HTTPS |
| Narration | Edge-TTS | outbound |

## Install

```bash
cd /data/workspace/pixelle-video
bash deploy/instabox/install.sh
```

The script installs `ffmpeg` / `chromium` / CJK fonts only when missing, runs
`uv sync --frozen --no-dev`, creates `config.yaml` and `secrets.env` (chmod 600)
if absent, downloads the segmentation model (skip with `--no-prefetch-model`), and installs:

- `pixelle-web.service`, `pixelle-api.service` (memory-capped, `Nice=5`)
- `pixelle-cleanup.timer`: deletes `output/<task_id>` folders older than 14 days

It uses the system Chromium (`PIXELLE_CHROMIUM_PATH=/usr/bin/chromium`), so no
Playwright browser download is needed.

## Secrets

`config.yaml` only holds references such as `${PIXELLE_LLM_API_KEY}`. Provide
values in one of two ways:

1. `secrets.env` next to `config.yaml` (chmod 600), or
2. instabox Org Secrets scoped to their host. The proxy swaps the placeholder for
   the real value, so the key never lands on disk. The units read
   `/run/instabox/env/env`.

Saving settings in the Web UI keeps the `${...}` references rather than writing
secret values into `config.yaml`.

## Publish the UI (private)

```bash
instabox-port publish 8501 --private --name pixelle
```

Open the printed link once; it sets a cookie (WebSockets included). Rotate it
with `--private --rotate-token` if it leaks.

## Cloudflare quota

The free plan allows roughly 130 FLUX images (~26 five-scene videos) per UTC day (estimate).
Check today's estimated usage with `cat ~/.cache/pixelle-video/usage/cloudflare.json`.

## Operate

```bash
systemctl status pixelle-web pixelle-api
journalctl -u pixelle-web -f
sudo systemctl restart pixelle-web pixelle-api
```

Upgrade: replace the code, then re-run `install.sh` (config and secrets are kept).

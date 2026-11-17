#!/usr/bin/env bash
# Install / upgrade Pixelle-Video on an instabox machine (Debian 13, systemd).
#
# Idempotent: safe to re-run after updating the code. It never overwrites an
# existing config.yaml or secrets.env and never deletes user data.
#
# Usage (from the project directory on the machine):
#   bash deploy/instabox/install.sh [--no-services] [--prefetch-model]
#
# Environment overrides:
#   PIXELLE_MODEL_DIR   segmentation model cache (default /data/workspace/models/pixelle)
#   MIN_FREE_MB         abort when /data has less free space (default 2500)

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$INSTALL_DIR/deploy/instabox"
RUN_USER="$(id -un)"
MODEL_DIR="${PIXELLE_MODEL_DIR:-/data/workspace/models/pixelle}"
MIN_FREE_MB="${MIN_FREE_MB:-2500}"
WITH_SERVICES=1
PREFETCH_MODEL=0

for arg in "$@"; do
    case "$arg" in
        --no-services) WITH_SERVICES=0 ;;
        --prefetch-model) PREFETCH_MODEL=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '[pixelle-install] %s\n' "$*"; }
die() { printf '[pixelle-install] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
[ "$(id -u)" -ne 0 ] || die "run as the normal user (sudo is used only where needed)"
command -v uv >/dev/null || die "uv not found (the instabox agent-browser template ships it)"
command -v sudo >/dev/null || die "sudo not found"

free_mb=$(df -Pm "$INSTALL_DIR" | awk 'NR==2 {print $4}')
log "install dir: $INSTALL_DIR (free: ${free_mb} MiB)"
[ "$free_mb" -ge "$MIN_FREE_MB" ] || die "need at least ${MIN_FREE_MB} MiB free, have ${free_mb} MiB"

# ---------------------------------------------------------------- system packages
missing=()
command -v ffmpeg >/dev/null || missing+=(ffmpeg)
command -v chromium >/dev/null || missing+=(chromium)
fc-list :lang=zh 2>/dev/null | grep -q . || missing+=(fonts-noto-cjk)
if [ "${#missing[@]}" -gt 0 ]; then
    log "installing system packages: ${missing[*]}"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}"
else
    log "system packages present (ffmpeg, chromium, CJK fonts)"
fi

# ---------------------------------------------------------------- python env
log "syncing Python environment (no dev extras)"
cd "$INSTALL_DIR"
uv sync --frozen --no-dev

# ---------------------------------------------------------------- config & secrets
if [ ! -f config.yaml ]; then
    cp "$DEPLOY_DIR/config.instabox.yaml" config.yaml
    log "created config.yaml from deploy/instabox/config.instabox.yaml"
else
    log "keeping existing config.yaml"
fi
if [ ! -f secrets.env ]; then
    install -m 600 "$DEPLOY_DIR/secrets.env.example" secrets.env
    log "created secrets.env (chmod 600) - fill in the keys"
fi
chmod 600 secrets.env config.yaml

mkdir -p "$MODEL_DIR" output
if [ "$PREFETCH_MODEL" -eq 1 ]; then
    log "prefetching segmentation model into $MODEL_DIR"
    PIXELLE_MODEL_DIR="$MODEL_DIR" .venv/bin/python -c \
        "from pixelle_video.services.animation_fx.segmentation import ensure_model; print(ensure_model())"
fi

# ---------------------------------------------------------------- services
if [ "$WITH_SERVICES" -eq 1 ]; then
    for unit in pixelle-web.service pixelle-api.service pixelle-cleanup.service pixelle-cleanup.timer; do
        sed -e "s#__INSTALL_DIR__#$INSTALL_DIR#g" -e "s#__USER__#$RUN_USER#g" -e "s#__MODEL_DIR__#$MODEL_DIR#g" \
            "$DEPLOY_DIR/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
    done
    sudo systemctl daemon-reload
    sudo systemctl enable --now pixelle-cleanup.timer
    sudo systemctl enable pixelle-web.service pixelle-api.service
    sudo systemctl restart pixelle-web.service pixelle-api.service
    log "services restarted: pixelle-web (127.0.0.1:8501), pixelle-api (127.0.0.1:8000)"
    log "publish the UI privately with: instabox-port publish 8501 --private --name pixelle"
fi

log "done"

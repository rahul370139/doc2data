#!/bin/bash
# Deploy and run doc2data on DGX2 as two persistent services:
#   1) doc2data-server   — FastAPI + Streamlit + Florence-2 + Ollama (GPU container)
#   2) doc2data-frontend — Next.js app served on port 3000 (proxies to :8000)
#
# Usage: ./deploy_and_run.sh
#
# Access URLs after success (via Tailscale):
#   Next.js frontend : http://100.126.216.92:3000
#   FastAPI docs     : http://100.126.216.92:8000/docs
#   Streamlit (legacy): http://100.126.216.92:8501

set -euo pipefail

DGX_HOST="radiant-dgx2@100.126.216.92"
DGX_KEY="../../dgx-spark/tailscale_spark2"
REMOTE_DIR="/home/radiant-dgx2/doc2data"
BACKEND_CONTAINER="doc2data-server"
FRONTEND_CONTAINER="doc2data-frontend"
NETWORK_NAME="doc2data-net"

# Allow non-interactive deploys by feeding the SSH key passphrase via
# ``SSHPASS`` when available.  When the user is running locally with a
# cached ssh-agent this is a no-op, so the legacy behaviour is preserved.
SSH_BASE=(ssh -i "$DGX_KEY")
RSYNC_SSH="ssh -i $DGX_KEY"
if [ -n "${SSHPASS:-}" ] && command -v sshpass > /dev/null 2>&1; then
    SSH_BASE=(sshpass -P passphrase -e ssh -i "$DGX_KEY")
    RSYNC_SSH="sshpass -P passphrase -e ssh -i $DGX_KEY"
fi

# Pick up an HF access token from the local .env (or the shell env) so
# gated / rate-limited model downloads authenticate on DGX.  We fall
# back to an empty string so ``huggingface_hub`` just retries
# anonymously when the token isn't set.
HF_ACCESS_TOKEN_VALUE=""
if [ -z "${HF_ACCESS_TOKEN:-}" ] && [ -f .env ]; then
    # Grep the assignment so we never source the whole .env (avoids
    # surprises like ``ENABLE_VLM=false`` globally disabling things).
    HF_ACCESS_TOKEN_VALUE="$(
        grep -E '^[[:space:]]*HF_ACCESS_TOKEN[[:space:]]*=' .env 2>/dev/null \
        | head -1 \
        | sed -E 's/^[^=]+=[[:space:]]*//' \
        | sed -E 's/^[[:space:]]+|[[:space:]]+$//g' \
        | sed -E "s/^['\"]//; s/['\"]\$//"
    )"
else
    HF_ACCESS_TOKEN_VALUE="${HF_ACCESS_TOKEN:-}"
fi
if [ -n "$HF_ACCESS_TOKEN_VALUE" ]; then
    echo "🔑 HF token detected (…$(echo "$HF_ACCESS_TOKEN_VALUE" | tail -c 5)); will pass to DGX."
else
    echo "ℹ️  No HF token found; model downloads will proceed anonymously."
fi

echo "📦 Syncing code to DGX..."
# Keep image assets (templates/debug references) for alignment reproducibility.
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'cache' \
    --exclude 'models_cache' \
    --exclude 'data/results' \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    --exclude 'frontend/.next' \
    --exclude 'frontend/node_modules' \
    --omit-dir-times \
    --no-times \
    -e "$RSYNC_SSH" \
    . "$DGX_HOST:$REMOTE_DIR"

echo "🐳 Building and deploying on DGX..."
"${SSH_BASE[@]}" "$DGX_HOST" bash -s -- \
    "$REMOTE_DIR" "$BACKEND_CONTAINER" "$FRONTEND_CONTAINER" "$NETWORK_NAME" \
    "$HF_ACCESS_TOKEN_VALUE" << 'ENDSSH'
set -euo pipefail

REMOTE_DIR="$1"
BACKEND_CONTAINER="$2"
FRONTEND_CONTAINER="$3"
NETWORK_NAME="$4"
HF_ACCESS_TOKEN="${5:-}"
export HF_ACCESS_TOKEN
# huggingface_hub reads these; set both so any library path (old or
# new) picks up the token without requiring a ``login`` call.
export HUGGING_FACE_HUB_TOKEN="${HF_ACCESS_TOKEN}"
export HF_TOKEN="${HF_ACCESS_TOKEN}"

cd "$REMOTE_DIR"

# Ensure a shared Docker network so frontend can reach the backend by name
if ! docker network inspect "$NETWORK_NAME" > /dev/null 2>&1; then
    echo "🌐 Creating Docker network '$NETWORK_NAME'..."
    docker network create "$NETWORK_NAME"
fi

# Ensure HuggingFace cache directory exists (for Florence-2 and other models)
mkdir -p /home/radiant-dgx2/.cache/huggingface

echo "🏗️ Building backend image..."
docker build -f docker/Dockerfile -t doc2data-gpu .

# Pre-download Florence-2-large if not already cached
FLORENCE_CACHE="/home/radiant-dgx2/.cache/huggingface"
FLORENCE_MODEL_DIR="$FLORENCE_CACHE/hub/models--florence-community--Florence-2-large"
# Common docker-run flags for the short-lived pre-download containers.
# We pass the HF token if available so gated repos and rate limits are
# avoided.  We never print the token.
HF_TOKEN_FLAGS=()
if [ -n "${HF_ACCESS_TOKEN:-}" ]; then
    HF_TOKEN_FLAGS+=(
        -e "HUGGING_FACE_HUB_TOKEN=$HF_ACCESS_TOKEN"
        -e "HF_TOKEN=$HF_ACCESS_TOKEN"
        -e "HF_ACCESS_TOKEN=$HF_ACCESS_TOKEN"
    )
fi

if [ -d "$FLORENCE_MODEL_DIR" ]; then
    echo "✅ Florence-2-large (community) already cached at $FLORENCE_MODEL_DIR"
else
    echo "📥 Pre-downloading florence-community/Florence-2-large (~1.5GB)..."
    if docker run --rm --gpus all \
        -v "$FLORENCE_CACHE:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        -e TRANSFORMERS_OFFLINE=0 \
        -e HF_HUB_OFFLINE=0 \
        -e USE_TF=0 \
        "${HF_TOKEN_FLAGS[@]}" \
        doc2data-gpu \
        python3 -c "from transformers import Florence2ForConditionalGeneration, CLIPImageProcessor, RobertaTokenizerFast; m='florence-community/Florence-2-large'; Florence2ForConditionalGeneration.from_pretrained(m); CLIPImageProcessor.from_pretrained(m); RobertaTokenizerFast.from_pretrained(m); print('OK')"; then
        echo "✅ Florence-2-large downloaded successfully"
    else
        echo "⚠️ Florence-2 pre-download failed (will attempt at runtime)"
    fi
fi

# Pre-download GOT-OCR 2.0 (stepfun-ai/GOT-OCR-2.0-hf, ~1.2GB).  We
# cache it under the same HF hub so the first-ever container run
# doesn't hang on a network download the instant a rescue fires.
# Failing this step is non-fatal — the agent falls back gracefully.
GOT_OCR_MODEL_DIR="$FLORENCE_CACHE/hub/models--stepfun-ai--GOT-OCR-2.0-hf"
if [ -d "$GOT_OCR_MODEL_DIR" ]; then
    echo "✅ GOT-OCR 2.0 already cached at $GOT_OCR_MODEL_DIR"
else
    echo "📥 Pre-downloading stepfun-ai/GOT-OCR-2.0-hf (~1.2GB)..."
    if docker run --rm --gpus all \
        -v "$FLORENCE_CACHE:/root/.cache/huggingface" \
        -e HF_HOME=/root/.cache/huggingface \
        -e TRANSFORMERS_OFFLINE=0 \
        -e HF_HUB_OFFLINE=0 \
        -e USE_TF=0 \
        "${HF_TOKEN_FLAGS[@]}" \
        doc2data-gpu \
        python3 -c "from transformers import AutoProcessor, AutoModelForImageTextToText; m='stepfun-ai/GOT-OCR-2.0-hf'; AutoProcessor.from_pretrained(m, use_fast=True); AutoModelForImageTextToText.from_pretrained(m); print('OK')"; then
        echo "✅ GOT-OCR 2.0 downloaded successfully"
    else
        echo "⚠️ GOT-OCR 2.0 pre-download failed (rescue ladder will skip it)"
    fi
fi

# Quick sanity check — confirm GOT-OCR actually LOADS with the new
# non-accelerate code path.  If this passes we know ``device_map``
# wasn't silently needed.  A failure here will surface the real root
# cause in the deploy log rather than 3 minutes later mid-benchmark.
echo "🔬 Smoke-testing GOT-OCR 2.0 load inside the image..."
docker run --rm --gpus all \
    -v "$FLORENCE_CACHE:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_OFFLINE=0 \
    -e USE_TF=0 \
    "${HF_TOKEN_FLAGS[@]}" \
    doc2data-gpu \
    python3 -c "import torch; from transformers import AutoModelForImageTextToText, AutoProcessor; m='stepfun-ai/GOT-OCR-2.0-hf'; p = AutoProcessor.from_pretrained(m, use_fast=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'; mdl = AutoModelForImageTextToText.from_pretrained(m, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(dev); mdl.eval(); print(f'GOT-OCR 2.0 loaded on {dev}, params={sum(p.numel() for p in mdl.parameters()) / 1e6:.1f}M')" \
    && echo "✅ GOT-OCR 2.0 smoke test passed" \
    || echo "⚠️ GOT-OCR 2.0 smoke test failed — see above; rescue will skip it"

# Smoke-test PARSeq — we need to verify (a) torch.hub can reach
# GitHub, (b) pytorch_lightning + timm are in the image, and (c) the
# model actually loads on the GPU before the benchmark fires it for
# real.  We also leave the downloaded weights in the torch.hub cache
# so the live server doesn't pay the first-load cost on the first
# rescue call.  This is non-fatal — if PARSeq can't load, the agent
# logs a warning and the rescue ladder skips the ``parseq`` strategy.
echo "🔬 Smoke-testing PARSeq load inside the image..."
docker run --rm --gpus all \
    -v "$FLORENCE_CACHE:/root/.cache/huggingface" \
    -v /home/radiant-dgx2/.cache/torch:/root/.cache/torch \
    -e HF_HOME=/root/.cache/huggingface \
    -e TORCH_HOME=/root/.cache/torch \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_OFFLINE=0 \
    -e USE_TF=0 \
    "${HF_TOKEN_FLAGS[@]}" \
    doc2data-gpu \
    python3 -c "import torch; import pytorch_lightning as pl; import timm; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; m = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True); m = m.to(dev).eval(); print(f'PARSeq loaded on {dev}, params={sum(p.numel() for p in m.parameters()) / 1e6:.1f}M, pl={pl.__version__}, timm={timm.__version__}')" \
    && echo "✅ PARSeq smoke test passed" \
    || echo "⚠️ PARSeq smoke test failed — see above; rescue ladder will skip the parseq strategy"

echo "🛑 Stopping old backend container (if running)..."
docker stop "$BACKEND_CONTAINER" 2>/dev/null || true
docker rm   "$BACKEND_CONTAINER" 2>/dev/null || true

echo "🚀 Starting backend..."
docker run --gpus all \
    -d \
    --restart unless-stopped \
    --network "$NETWORK_NAME" \
    --network-alias "doc2data-api" \
    -p 8501:8501 \
    -p 8000:8000 \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --add-host=host.docker.internal:host-gateway \
    -v "$(pwd)"/models_cache:/root/.paddlex \
    -v "$(pwd)"/data:/app/data \
    -v "$(pwd)"/cache:/app/cache \
    -v /home/radiant-dgx2/.cache/huggingface:/root/.cache/huggingface \
    -v /home/radiant-dgx2/.cache/torch:/root/.cache/torch \
    -v /home/radiant-dgx2/.ollama/models:/root/.ollama/models \
    -e CMS1500_TEMPLATE_PATH=/app/data/raw/cms1500_template.pdf \
    -e CMS1500_RED_S_MIN=0.34 \
    -e CMS1500_RED_V_MIN=0.45 \
    -e CMS1500_RED_RATIO_SWITCH=0.0012 \
    -e CMS1500_RANSAC_REPROJ_DEFAULT=3.5 \
    -e CMS1500_RANSAC_REPROJ_HANDWRITTEN=5.5 \
    -e CMS1500_QUAD_MIN_SCORE_DEFAULT=0.52 \
    -e CMS1500_QUAD_MIN_SCORE_HANDWRITTEN=0.34 \
    -e CMS1500_MIN_FEATURE_QUALITY=0.33 \
    -e DISABLE_MODEL_SOURCE_CHECK=true \
    -e USE_TF=0 \
    -e TF_CPP_MIN_LOG_LEVEL=3 \
    -e TRANSFORMERS_OFFLINE=0 \
    -e HF_HUB_OFFLINE=0 \
    -e TORCH_HOME=/root/.cache/torch \
    -e OLLAMA_HOST=localhost:11434 \
    -e OLLAMA_MODEL_VLM=minicpm-v \
    -e OLLAMA_MODEL_VLM_OCR=minicpm-v \
    -e VLM_MODEL_SECTION="${VLM_MODEL_SECTION:-openbmb/minicpm-o4.5:latest}" \
    -e VLM_MODEL_SECTION_FALLBACK="${VLM_MODEL_SECTION_FALLBACK:-minicpm-v}" \
    -e OLLAMA_NUM_PARALLEL=4 \
    -e OLLAMA_MAX_LOADED_MODELS=2 \
    -e OLLAMA_KEEP_ALIVE=30m \
    -e ENABLE_SLM=true \
    -e ENABLE_VLM=true \
    -e HF_HOME=/root/.cache/huggingface \
    ${HF_ACCESS_TOKEN:+-e HUGGING_FACE_HUB_TOKEN=$HF_ACCESS_TOKEN} \
    ${HF_ACCESS_TOKEN:+-e HF_TOKEN=$HF_ACCESS_TOKEN} \
    ${HF_ACCESS_TOKEN:+-e HF_ACCESS_TOKEN=$HF_ACCESS_TOKEN} \
    --name "$BACKEND_CONTAINER" \
    doc2data-gpu

echo "🏗️ Building frontend image..."
docker build -f frontend/Dockerfile -t doc2data-frontend:latest frontend

echo "🛑 Stopping old frontend container (if running)..."
docker stop "$FRONTEND_CONTAINER" 2>/dev/null || true
docker rm   "$FRONTEND_CONTAINER" 2>/dev/null || true

echo "🚀 Starting frontend..."
docker run \
    -d \
    --restart unless-stopped \
    --network "$NETWORK_NAME" \
    -p 3000:3000 \
    -e API_BASE_URL="http://doc2data-api:8000" \
    --name "$FRONTEND_CONTAINER" \
    doc2data-frontend:latest

echo "⏳ Waiting for services to start..."
sleep 10

echo "📊 Service status:"
echo "-- backend --"
docker logs "$BACKEND_CONTAINER"  2>&1 | tail -20
echo "-- frontend --"
docker logs "$FRONTEND_CONTAINER" 2>&1 | tail -10

# Health probe
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "✅ Backend /health responded"
else
    echo "⚠️ Backend /health did NOT respond (first boot may take longer)"
fi
if curl -sf http://localhost:3000 > /dev/null; then
    echo "✅ Frontend responded"
else
    echo "⚠️ Frontend did NOT respond yet"
fi

echo ""
echo "=============================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "📡 URLs (via Tailscale):"
echo "   Frontend        : http://100.126.216.92:3000"
echo "   FastAPI         : http://100.126.216.92:8000"
echo "   API Docs        : http://100.126.216.92:8000/docs"
echo "   Streamlit (legacy): http://100.126.216.92:8501"
echo ""
echo "📝 Useful commands:"
echo "   Backend logs    : docker logs -f $BACKEND_CONTAINER"
echo "   Frontend logs   : docker logs -f $FRONTEND_CONTAINER"
echo "   Restart both    : docker restart $BACKEND_CONTAINER $FRONTEND_CONTAINER"
echo "   Stop both       : docker stop $BACKEND_CONTAINER $FRONTEND_CONTAINER"
echo "=============================================="
ENDSSH

echo ""
echo "🎉 Deployment complete! Check the DGX for service status."

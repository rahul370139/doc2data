#!/bin/bash
# Deploy and run doc2data on DGX server as a persistent service
# Usage: ./deploy_and_run.sh

set -e

DGX_HOST="radiant-dgx2@100.126.216.92"
DGX_KEY="../../dgx-spark/tailscale_spark2"
REMOTE_DIR="/home/radiant-dgx2/doc2data"
CONTAINER_NAME="doc2data-server"

echo "📦 Syncing code to DGX..."
# Keep image assets (templates/debug references) for alignment reproducibility.
# We intentionally do NOT exclude *.png/*.jpg globally.
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'cache' \
    --exclude 'models_cache' \
    --exclude 'data/results' \
    --omit-dir-times \
    --no-times \
    --exclude '.venv' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    -e "ssh -i $DGX_KEY" \
    . "$DGX_HOST:$REMOTE_DIR"

echo "🐳 Building and deploying on DGX..."
ssh -i "$DGX_KEY" "$DGX_HOST" << 'ENDSSH'
cd /home/radiant-dgx2/doc2data
CONTAINER_NAME="doc2data-server"

# Ensure HuggingFace cache directory exists (for Florence-2 and other models)
mkdir -p /home/radiant-dgx2/.cache/huggingface

echo "🏗️ Building Docker image..."
docker build -f docker/Dockerfile -t doc2data-gpu .

# Pre-download Florence-2 if not already cached (uses florence-community fork with native weights)
FLORENCE_CACHE="/home/radiant-dgx2/.cache/huggingface"
FLORENCE_MODEL_DIR="$FLORENCE_CACHE/hub/models--florence-community--Florence-2-large"
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
        doc2data-gpu \
        python3 -c "from transformers import Florence2ForConditionalGeneration, CLIPImageProcessor, RobertaTokenizerFast; m='florence-community/Florence-2-large'; Florence2ForConditionalGeneration.from_pretrained(m); CLIPImageProcessor.from_pretrained(m); RobertaTokenizerFast.from_pretrained(m); print('OK')"; then
        echo "✅ Florence-2-large downloaded successfully"
    else
        echo "⚠️ Florence-2 pre-download failed (will attempt at runtime)"
    fi
fi

echo "🛑 Stopping old container (if running)..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "🚀 Starting persistent server..."
docker run --gpus all \
    -d \
    --restart unless-stopped \
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
    -e OLLAMA_HOST=localhost:11434 \
    -e OLLAMA_MODEL_VLM=minicpm-v \
    -e OLLAMA_MODEL_VLM_OCR=minicpm-v \
    -e ENABLE_SLM=true \
    -e ENABLE_VLM=true \
    -e HF_HOME=/root/.cache/huggingface \
    --name $CONTAINER_NAME \
    doc2data-gpu

echo "⏳ Waiting for services to start..."
sleep 10

echo "📊 Checking service status..."
docker logs $CONTAINER_NAME 2>&1 | tail -20

echo ""
echo "=============================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "📡 API Endpoints (via Tailscale):"
echo "   FastAPI:   http://100.126.216.92:8000"
echo "   API Docs:  http://100.126.216.92:8000/docs"
echo "   Streamlit: http://100.126.216.92:8501"
echo ""
echo "🔗 Share with team members:"
echo "   1. They need to be on the same Tailscale network"
echo "   2. Share the URL: http://100.126.216.92:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "   View logs:    docker logs -f $CONTAINER_NAME"
echo "   Stop server:  docker stop $CONTAINER_NAME"
echo "   Restart:      docker restart $CONTAINER_NAME"
echo "=============================================="
ENDSSH

echo ""
echo "🎉 Deployment complete! Check the DGX for service status."

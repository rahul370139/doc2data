#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting End-to-End GPU Pipeline Run..."

# 1. Check for NVIDIA Drivers
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi
echo "✅ NVIDIA Drivers detected"

# 2. Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found."
    exit 1
fi
echo "✅ Docker detected"

# 3. Build Container
echo "🏗️  Building Docker container (doc2data-gpu)..."
docker build -t doc2data-gpu .

# 4. Run Verification Script inside Container
echo "🧪 Running verification script inside container..."
docker run --gpus all \
    --rm \
    -v $(pwd):/app \
    -v $(pwd)/models_cache:/root/.paddlex \
    doc2data-gpu \
    python3 verify_gpu_pipeline.py

echo "🎉 End-to-End Run Complete!"
echo "To start the server persistently, run: ./run_docker_gpu.sh"


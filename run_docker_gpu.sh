#!/bin/bash

# Build the Docker image
echo "🏗️ Building Docker image (this may take a while)..."
docker build -t doc2data-gpu .

# Run the container
echo "🚀 Starting container with GPU support..."
echo "   - Streamlit: http://localhost:8501"
echo "   - FastAPI: http://localhost:8000"

# Check if NVIDIA Runtime is available
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "⚠️  Warning: NVIDIA Runtime not found in Docker. GPU might not work."
    echo "   Please install nvidia-container-toolkit."
fi

# Run with GPU support
# --gpus all: Enable all GPUs
# --add-host: Allow container to access host services (like Ollama)
# -v: Mount models cache to avoid re-downloading
docker run --gpus all \
    -it \
    --rm \
    -p 8501:8501 \
    -p 8000:8000 \
    --add-host=host.docker.internal:host-gateway \
    -v $(pwd)/models_cache:/root/.paddlex \
    -v $(pwd)/data:/app/data \
    --name doc2data-gpu-container \
    doc2data-gpu


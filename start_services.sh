#!/bin/bash
# Startup script for doc2data services
# This script starts FastAPI and Streamlit IMMEDIATELY, then downloads models in background

set -e

echo "🚀 Starting Doc2Data Services..."
echo "   FastAPI will be available at: http://0.0.0.0:8000"
echo "   Streamlit will be available at: http://0.0.0.0:8501"

# Create log directory
mkdir -p /tmp/logs

# 1. Start FastAPI server FIRST (most important for API access)
echo "📡 Starting FastAPI server on port 8000..."
cd /app
nohup uvicorn app.api_main:app --host 0.0.0.0 --port 8000 --workers 2 > /tmp/logs/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "   FastAPI PID: $FASTAPI_PID"

# Wait briefly for FastAPI to start
sleep 3

# Check if FastAPI started
if kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "✅ FastAPI started successfully"
else
    echo "❌ FastAPI failed to start. Check /tmp/logs/fastapi.log"
    cat /tmp/logs/fastapi.log | tail -20
fi

# 2. Start Ollama server in background (if available)
if command -v ollama > /dev/null 2>&1; then
    echo "🤖 Starting Ollama server..."
    nohup ollama serve > /tmp/logs/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "   Ollama PID: $OLLAMA_PID"
    
    # Optional model pull on start (disabled by default to avoid re-downloads)
    if [ "${OLLAMA_PULL_ON_START:-false}" = "true" ]; then
        (
            sleep 10  # Wait for Ollama to fully start
            echo "📥 Downloading SLM model in background..."
            ollama pull qwen2.5:3b 2>&1 | tee /tmp/logs/ollama_pull.log || echo "SLM model pull failed"
            echo "✅ SLM model ready"
        ) &
    else
        echo "ℹ️ Skipping Ollama model pull (OLLAMA_PULL_ON_START=false)"
    fi
else
    echo "⚠️ Ollama not installed - SLM/VLM features disabled"
fi

# 3. Start Streamlit (this runs in foreground to keep container alive)
echo "🎨 Starting Streamlit on port 8501..."
exec streamlit run app/streamlit_main.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false


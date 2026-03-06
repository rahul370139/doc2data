#!/bin/bash
# Startup script for doc2data services

set -e

echo "🚀 Starting Doc2Data Services..."
echo "   FastAPI will be available at: http://0.0.0.0:8000"
echo "   Streamlit will be available at: http://0.0.0.0:8501"

# Create log directory
mkdir -p /tmp/logs

# 1. Start Ollama FIRST (VLM/SLM depends on it)
if [ -x /usr/local/bin/ollama ]; then
    echo "🤖 Starting Ollama server..."
    # Ollama uses OLLAMA_HOST for its LISTEN address.
    # Force 0.0.0.0 so it accepts connections from within the container.
    # Python code connects to localhost:11434 via Config.OLLAMA_HOST.
    OLLAMA_HOST="0.0.0.0:11434" nohup /usr/local/bin/ollama serve > /tmp/logs/ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo "   Ollama PID: $OLLAMA_PID"

    # Wait until Ollama is ready (up to 15 seconds)
    for i in $(seq 1 15); do
        if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Show available models
    MODEL_COUNT=$(curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "0")
    echo "   Ollama models available: $MODEL_COUNT"
    curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for m in d.get('models',[]):
        print(f\"   - {m['name']}\")
except: pass
" 2>/dev/null || true

    # Pull required VLM model if not already available
    VLM_MODEL="${OLLAMA_MODEL_VLM:-minicpm-v}"
    HAS_VLM=$(curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    names=[m['name'] for m in d.get('models',[])]
    print('yes' if any('$VLM_MODEL' in n for n in names) else 'no')
except: print('no')
" 2>/dev/null || echo "no")

    if [ "$HAS_VLM" != "yes" ]; then
        echo "📥 Pulling VLM model: $VLM_MODEL (required for table extraction + OCR rescue)..."
        /usr/local/bin/ollama pull "$VLM_MODEL" 2>&1 | tail -5 || echo "⚠️ Failed to pull $VLM_MODEL"
    else
        echo "✅ VLM model $VLM_MODEL already available"
    fi

    # Pre-warm only the models we use: minicpm-v (rescue + table) and llava (table fallback)
    for m in "minicpm-v" "llava"; do
        echo "🔥 Pre-warming $m ..."
        curl -sf http://localhost:11434/api/generate \
            -d "{\"model\": \"$m\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
            --max-time 120 > /dev/null 2>&1 && echo "✅ $m warm" || echo "⚠️ $m warm-up failed"
    done
else
    echo "⚠️ Ollama binary not found at /usr/local/bin/ollama - VLM/SLM features disabled"
fi

# 2. Start FastAPI server
echo "📡 Starting FastAPI server on port 8000..."
cd /app
nohup uvicorn app.api_main:app --host 0.0.0.0 --port 8000 --workers 2 > /tmp/logs/fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "   FastAPI PID: $FASTAPI_PID"

sleep 3

if kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "✅ FastAPI started successfully"
else
    echo "❌ FastAPI failed to start. Check /tmp/logs/fastapi.log"
    tail -20 /tmp/logs/fastapi.log
fi

# 3. Start Streamlit (foreground - keeps container alive)
echo "🎨 Starting Streamlit on port 8501..."
exec streamlit run app/streamlit_main.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false

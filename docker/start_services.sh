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

    # Required models:
    #   - $OLLAMA_MODEL_VLM              (minicpm-v)       → rescue + table fallback
    #   - $VLM_MODEL_SECTION             (minicpm-o4.5)    → Tier-1 section reads
    #   - $VLM_MODEL_SECTION_FALLBACK    (minicpm-v)       → Tier-1 fallback
    REQUIRED_MODELS=(
        "${OLLAMA_MODEL_VLM:-minicpm-v}"
        "${VLM_MODEL_SECTION:-openbmb/minicpm-o4.5:latest}"
        "${VLM_MODEL_SECTION_FALLBACK:-minicpm-v}"
    )
    # Deduplicate while preserving order
    declare -A SEEN_MODELS
    UNIQUE_REQUIRED=()
    for m in "${REQUIRED_MODELS[@]}"; do
        if [ -n "$m" ] && [ -z "${SEEN_MODELS[$m]:-}" ]; then
            SEEN_MODELS[$m]=1
            UNIQUE_REQUIRED+=("$m")
        fi
    done

    for MODEL in "${UNIQUE_REQUIRED[@]}"; do
        HAS_MODEL=$(curl -sf http://localhost:11434/api/tags 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    names=[m['name'] for m in d.get('models',[])]
    print('yes' if any('$MODEL' in n or n.split(':')[0] == '$MODEL'.split(':')[0] for n in names) else 'no')
except: print('no')
" 2>/dev/null || echo "no")

        if [ "$HAS_MODEL" != "yes" ]; then
            echo "📥 Pulling $MODEL ..."
            /usr/local/bin/ollama pull "$MODEL" 2>&1 | tail -5 || echo "⚠️ Failed to pull $MODEL"
        else
            echo "✅ Model $MODEL already available"
        fi
    done

    # Pre-warm only the models we actually use as primaries.
    WARM_MODELS=(
        "${OLLAMA_MODEL_VLM:-minicpm-v}"
        "${VLM_MODEL_SECTION:-openbmb/minicpm-o4.5:latest}"
    )
    for m in "${WARM_MODELS[@]}"; do
        echo "🔥 Pre-warming $m ..."
        curl -sf http://localhost:11434/api/generate \
            -d "{\"model\": \"$m\", \"prompt\": \"hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
            --max-time 180 > /dev/null 2>&1 && echo "✅ $m warm" || echo "⚠️ $m warm-up failed"
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

#!/bin/bash
#
# Start Doc2Data API Server
# Usage: ./scripts/start_api_server.sh [--port 8000] [--host 0.0.0.0] [--workers 4]
#
# For DGX deployment:
#   nohup ./scripts/start_api_server.sh --port 8000 --workers 4 > logs/api.log 2>&1 &
#
# The API will be accessible at: http://<DGX_IP>:8000
# Share this URL with others to connect from any machine.
#

set -e

# Default values
PORT=8000
HOST="0.0.0.0"
WORKERS=1
RELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create logs directory
mkdir -p logs

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "uvicorn not found. Installing..."
    pip install uvicorn[standard] fastapi python-multipart
fi

echo "========================================"
echo "Doc2Data API Server"
echo "========================================"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"
echo ""
echo "API Endpoints:"
echo "  Health:   http://$HOST:$PORT/health"
echo "  Extract:  http://$HOST:$PORT/extract/v2"
echo "  CMS-1500: http://$HOST:$PORT/extract/cms1500"
echo "  UB-04:    http://$HOST:$PORT/extract/ub04"
echo "  Reducto:  http://$HOST:$PORT/extract/reducto"
echo "  Docs:     http://$HOST:$PORT/docs"
echo ""
echo "To share with others, use your machine's IP:"
echo "  http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo '<YOUR_IP>'):$PORT"
echo "========================================"

# Run the server
if [ "$WORKERS" -gt 1 ]; then
    # Production mode with multiple workers
    exec uvicorn app.api_main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info
else
    # Development mode
    exec uvicorn app.api_main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        $RELOAD
fi

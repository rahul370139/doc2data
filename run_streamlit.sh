#!/bin/bash
# Launch Streamlit demo
# Usage: ./run_streamlit.sh

cd "$(dirname "$0")"
source venv/bin/activate

echo "=========================================="
echo "Starting Streamlit Demo"
echo "=========================================="
echo ""
echo "The app will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

streamlit run app/streamlit_main.py \
    --server.address localhost \
    --server.port 8501 \
    --server.headless false


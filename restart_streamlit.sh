#!/bin/bash
# Streamlit Restart Script
# Usage: ./restart_streamlit.sh

cd "/Users/rahul/Downloads/Code scripts/doc2data"

echo "🛑 Stopping Streamlit..."
pkill -9 -f "streamlit" 2>/dev/null
lsof -ti:8501 | xargs kill -9 2>/dev/null
sleep 2

echo "🔄 Starting Streamlit..."
source venv/bin/activate
streamlit run app/streamlit_main.py --server.address localhost --server.port 8501 2>&1 &

sleep 6
if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Streamlit restarted successfully at http://localhost:8501"
else
    echo "⏳ Streamlit is starting... (check http://localhost:8501)"
fi

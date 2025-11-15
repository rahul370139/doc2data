#!/bin/bash
# Quick script to check if Ollama is running

echo "🔍 Checking Ollama status..."
echo ""

# Check if ollama command exists
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed or not in PATH"
    echo "   Install from: https://ollama.ai"
    exit 1
fi

echo "✅ Ollama is installed: $(which ollama)"
echo ""

# Check if Ollama server is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server is running on http://localhost:11434"
    echo ""
    echo "📦 Available models:"
    ollama list
    echo ""
    echo "💡 To start Ollama server: ollama serve"
    echo "💡 To pull a model: ollama pull qwen2.5:7b-instruct"
else
    echo "❌ Ollama server is NOT running"
    echo ""
    echo "💡 To start Ollama server, run: ollama serve"
    echo "   (This will start the server in the foreground)"
    echo "   Or run it in background: nohup ollama serve > /dev/null 2>&1 &"
fi


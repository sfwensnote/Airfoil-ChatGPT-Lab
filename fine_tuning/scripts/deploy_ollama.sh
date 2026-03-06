#!/bin/bash
cd /Users/wensifan/bot-remote-windows
# Ensure Ollama server is running (will fail silently if already running)
OLLAMA_HOST="127.0.0.1:11434" ./Ollama.app/Contents/Resources/ollama serve > ollama.log 2>&1 &
sleep 5

echo "Creating model 'airfoil-tutor'..."
OLLAMA_HOST="127.0.0.1:11434" ./Ollama.app/Contents/Resources/ollama create airfoil-tutor -f fine_tuning/outputs/Modelfile

echo "Testing model..."
OLLAMA_HOST="127.0.0.1:11434" ./Ollama.app/Contents/Resources/ollama run airfoil-tutor "什么是雷诺数？"

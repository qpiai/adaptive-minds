#!/bin/bash
# Setup script for Adaptive Minds using uv

set -e  # Exit on any error

echo "🚀 Setting up Adaptive Minds environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    if [ ! -f .env ]; then
        echo "❌ HF_TOKEN not found. Please:"
        echo "   1. Get a token from: https://huggingface.co/settings/tokens"
        echo "   2. Accept Llama 3.1 license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
        echo "   3. Create .env file: echo 'HF_TOKEN=your_token_here' > .env"
        exit 1
    fi
fi

echo "✅ Installing dependencies with uv..."
uv sync

echo "✅ Downloading models..."
cd playground
uv run python download_models.py
cd ..

echo "🎉 Setup complete! You can now:"
echo "   • Start server: cd playground && uv run python server.py"
echo "   • Test API: curl http://localhost:8765/status"
echo "   • Run examples: uv run python examples/basic_usage.py"
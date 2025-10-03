#!/bin/bash

# Adaptive Minds - cURL Examples
# 
# This script demonstrates how to interact with Adaptive Minds using cURL commands.
# Make sure the server is running: docker compose up

API_BASE_URL="http://localhost:8765"

echo "🧠 Adaptive Minds - cURL Examples"
echo "=================================="

# Function to make API calls with better formatting
make_request() {
    local query="$1"
    local description="$2"
    
    echo ""
    echo "📝 $description"
    echo "Query: $query"
    echo "---"
    
    curl -s -X POST "$API_BASE_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\"}" | \
        python3 -m json.tool 2>/dev/null || echo "❌ Failed to parse response"
    
    echo ""
    sleep 2
}

# Check server status first
echo "🔍 Checking server status..."
curl -s "$API_BASE_URL/status" | python3 -m json.tool 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Server not responding. Make sure to run: docker compose up"
    exit 1
fi

echo "✅ Server is running!"
echo ""

# Example queries for each domain expert
make_request "Hello! How are you?" "General conversation"

make_request "What is the molecular formula of water?" "Chemistry query"

make_request "How do I calculate compound interest?" "Finance query"

make_request "What is the difference between AI and machine learning?" "AI/Technology query"

make_request "What are the symptoms of flu?" "Medical query"

make_request "Tell me a fun fact about space" "General knowledge"

# Show how to clear history
echo "🧹 Clearing conversation history..."
curl -s -X DELETE "$API_BASE_URL/chat/history"
echo "History cleared!"

echo ""
echo "✅ All examples completed!"
echo ""
echo "💡 Try your own queries:"
echo "curl -X POST $API_BASE_URL/chat \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"query\": \"Your question here\"}'"

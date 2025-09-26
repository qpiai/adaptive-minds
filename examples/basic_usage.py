#!/usr/bin/env python3
"""
Basic Usage Examples for Adaptive Minds API

This script demonstrates how to interact with the Adaptive Minds API
using simple Python requests.

Prerequisites:
- Adaptive Minds server running (docker compose up)
- Server accessible at http://localhost:8765
"""

import requests
import json
import time

# API Configuration
API_BASE_URL = "http://localhost:8765"

def check_server_status():
    """Check if the Adaptive Minds server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running!")
            print(f"📊 Loaded adapters: {', '.join(data['loaded_adapters'])}")
            print(f"🖥️  GPU memory: {data['gpu_memory_allocated']:.1f}GB")
            return True
        else:
            print(f"❌ Server error: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to run: docker compose up")
        return False

def send_query(query: str):
    """Send a query to Adaptive Minds and display the response."""
    try:
        print(f"\n🤔 Query: {query}")
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"🎯 Selected Expert: {data['selected_adapter']}")
            print(f"💭 Response: {data['response']}")
            print(f"🔍 Reasoning: {data['reasoning']}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def clear_history():
    """Clear the conversation history."""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat/history")
        if response.status_code == 200:
            print("🧹 Conversation history cleared!")
        else:
            print(f"❌ Failed to clear history: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def main():
    """Run example queries demonstrating different domain experts."""
    
    print("🧠 Adaptive Minds - Basic Usage Examples")
    print("=" * 50)
    
    # Check server status
    if not check_server_status():
        return
    
    # Example queries for each domain
    example_queries = [
        # General conversation
        "Hello! How are you today?",
        
        # Chemistry expert
        "What is the molecular formula of caffeine?",
        
        # Finance expert  
        "How does compound interest work?",
        
        # AI/Technology expert
        "Explain what machine learning is",
        
        # Medical expert
        "What are the symptoms of common cold?",
        
        # General knowledge
        "What is the capital of France?"
    ]
    
    print(f"\n🚀 Running {len(example_queries)} example queries...")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}/{len(example_queries)}")
        print('='*60)
        
        send_query(query)
        
        # Small delay between queries
        if i < len(example_queries):
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print("✅ All examples completed!")
    print("\n💡 Try your own queries:")
    print(f"   curl -X POST {API_BASE_URL}/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "Your question here"}\'')

if __name__ == "__main__":
    main()

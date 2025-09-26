#!/usr/bin/env python3
"""
Interactive Demo for Adaptive Minds

A simple interactive command-line interface to chat with Adaptive Minds.
Perfect for testing and demonstrating the system's capabilities.

Usage:
    python examples/interactive_demo.py

Prerequisites:
- Adaptive Minds server running (docker compose up)
- Python requests library
"""

import requests
import json
import sys

# API Configuration
API_BASE_URL = "http://localhost:8765"

def check_server():
    """Check if server is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_query(query: str):
    """Send query and return response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def clear_history():
    """Clear conversation history."""
    try:
        requests.delete(f"{API_BASE_URL}/chat/history")
        return True
    except:
        return False

def print_response(data):
    """Pretty print the response."""
    if "error" in data:
        print(f"❌ Error: {data['error']}")
        return
    
    print(f"🎯 Expert: {data.get('selected_adapter', 'Unknown')}")
    print(f"💭 Response: {data.get('response', 'No response')}")
    
    # Show reasoning if available
    reasoning = data.get('reasoning', '')
    if reasoning and reasoning.strip():
        print(f"🔍 Why this expert: {reasoning}")

def show_help():
    """Show available commands."""
    print("\n📋 Available commands:")
    print("  /help     - Show this help")
    print("  /clear    - Clear conversation history") 
    print("  /status   - Show server status")
    print("  /examples - Show example queries")
    print("  /quit     - Exit the demo")
    print("  Or just type any question!\n")

def show_examples():
    """Show example queries."""
    examples = [
        ("General", "Hello, how are you?"),
        ("Chemistry", "What is the molecular formula of caffeine?"),
        ("Finance", "How does compound interest work?"),
        ("AI/Tech", "Explain machine learning"),
        ("Medical", "What are symptoms of flu?"),
        ("Knowledge", "What is the capital of Japan?")
    ]
    
    print("\n💡 Example queries to try:")
    for domain, query in examples:
        print(f"  {domain:10} → {query}")
    print()

def main():
    """Main interactive loop."""
    print("🧠 Adaptive Minds - Interactive Demo")
    print("====================================")
    
    # Check server
    if not check_server():
        print("❌ Cannot connect to Adaptive Minds server!")
        print("💡 Make sure to run: docker compose up")
        sys.exit(1)
    
    print("✅ Connected to Adaptive Minds server!")
    print("Type /help for commands or just ask any question.")
    print("Press Ctrl+C to exit anytime.\n")
    
    try:
        while True:
            # Get user input
            user_input = input("🤔 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/help':
                    show_help()
                elif command == '/clear':
                    if clear_history():
                        print("🧹 Conversation history cleared!")
                    else:
                        print("❌ Failed to clear history")
                elif command == '/status':
                    try:
                        response = requests.get(f"{API_BASE_URL}/status")
                        data = response.json()
                        print(f"📊 Status: {'✅ Running' if data.get('is_initialized') else '❌ Not ready'}")
                        print(f"🧠 Loaded experts: {', '.join(data.get('loaded_adapters', []))}")
                        print(f"🖥️  GPU memory: {data.get('gpu_memory_allocated', 0):.1f}GB")
                    except:
                        print("❌ Failed to get status")
                elif command == '/examples':
                    show_examples()
                elif command in ['/quit', '/exit']:
                    print("👋 Goodbye!")
                    break
                else:
                    print(f"❓ Unknown command: {user_input}")
                    print("Type /help for available commands")
                
                continue
            
            # Send query to Adaptive Minds
            print("🤖 Thinking...")
            response = send_query(user_input)
            print_response(response)
            print()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()

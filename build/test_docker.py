#!/usr/bin/env python3
"""
Docker test script for Adaptive Minds
This script tests the basic functionality of the system in a Docker environment.
"""

import os
import sys
import time
import requests
from typing import Dict, Any

def test_server_health() -> bool:
    """Test if the FastAPI server is responding."""
    try:
        response = requests.get("http://localhost:8765/status", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

def test_chat_endpoint() -> bool:
    """Test the chat endpoint with a simple query."""
    try:
        test_query = {
            "query": "Hello, how are you?"
        }
        response = requests.post(
            "http://localhost:8765/chat",
            json=test_query,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat test successful!")
            print(f"   Selected adapter: {data.get('selected_adapter', 'Unknown')}")
            print(f"   Response preview: {data.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Chat endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint test failed: {e}")
        return False

def main():
    """Run all Docker tests."""
    print("🧪 Running Adaptive Minds Docker Tests")
    print("=" * 50)
    
    # Wait a bit for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    tests = [
        ("Server Health", test_server_health),
        ("Chat Endpoint", test_chat_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

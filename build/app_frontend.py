import streamlit as st
import requests
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="Adaptive Minds",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Adaptive Minds")
st.markdown("*LangGraph-powered multi-agent system with intelligent routing*")

# Configuration
API_BASE_URL = "http://localhost:8765"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_status' not in st.session_state:
    st.session_state.api_status = None

def check_api_status() -> Dict[str, Any]:
    """Check if the FastAPI server is running and get status."""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return {"connected": True, "data": response.json()}
        else:
            return {"connected": False, "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"connected": False, "error": str(e)}

def send_chat_message(query: str) -> Dict[str, Any]:
    """Send a chat message to the FastAPI server."""
    try:
        payload = {"query": query}
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def clear_chat_history():
    """Clear the chat history on the server."""
    try:
        response = requests.delete(f"{API_BASE_URL}/chat/history", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Sidebar for system status
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check API status
    if st.button("🔄 Refresh Status"):
        st.session_state.api_status = check_api_status()
        st.rerun()
    
    if st.session_state.api_status is None:
        st.session_state.api_status = check_api_status()
    
    status = st.session_state.api_status
    
    if status["connected"]:
        st.success("✅ Connected to FastAPI server")
        
        # Display system info
        data = status["data"]
        st.subheader("📊 System Info")
        st.write(f"**Initialized**: {'✅' if data['is_initialized'] else '❌'}")
        st.write(f"**Loaded Adapters**: {len(data['loaded_adapters'])}")
        
        if data['loaded_adapters']:
            st.write("**Adapters**:")
            for adapter in data['loaded_adapters']:
                st.write(f"  • {adapter}")
        
        if data.get('gpu_memory_allocated'):
            st.write(f"**GPU Memory**: {data['gpu_memory_allocated']:.1f}GB")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            if clear_chat_history():
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
            else:
                st.error("Failed to clear chat history")
    else:
        st.error("❌ Cannot connect to FastAPI server")
        st.write(f"**Error**: {status['error']}")
        st.write(f"**Expected URL**: {API_BASE_URL}")
        st.write("Make sure the FastAPI server is running on port 8765")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(f"🧠 **{message['adapter']} via Adaptive Minds**")
            st.write(message["content"])
    elif message["role"] == "reasoning":
        with st.expander("🔍 Router Agent Decision", expanded=False):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Check if API is connected
    if not st.session_state.api_status or not st.session_state.api_status["connected"]:
        st.error("❌ Cannot connect to FastAPI server. Please check the server status in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Send to API and get response
    with st.spinner("🤔 Processing your query..."):
        result = send_chat_message(prompt)
        
        if result["success"]:
            data = result["data"]
            
            # Display reasoning
            with st.expander("🔍 Router Agent Decision", expanded=False):
                st.markdown(data["reasoning"])
            
            # Add reasoning to chat history
            st.session_state.chat_history.append({
                "role": "reasoning", 
                "content": data["reasoning"]
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(f"🧠 **{data['selected_adapter']} via Adaptive Minds**")
                st.write(data["response"])
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": data["response"],
                "adapter": data["selected_adapter"]
            })
        else:
            st.error(f"❌ Error: {result['error']}")

# Footer
st.markdown("---")
st.markdown(f"*Connected to FastAPI server at {API_BASE_URL}*")

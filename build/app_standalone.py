import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import gc
from typing import TypedDict, Annotated, Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator

# Page configuration
st.set_page_config(
    page_title="Adaptive Minds",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Adaptive Minds")
st.markdown("*LangGraph-powered multi-agent system with intelligent routing*")

# Define the state that will be passed between agents
class AgentState(TypedDict):
    query: str
    selected_adapter: str
    response: str
    reasoning: str
    messages: Annotated[List[dict], operator.add]
    conversation_history: List[dict]
    
# Initialize session state for Streamlit
if 'base_model' not in st.session_state:
    st.session_state.base_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'loaded_adapters' not in st.session_state:
    st.session_state.loaded_adapters = {}
if 'workflow' not in st.session_state:
    st.session_state.workflow = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Enhanced LoRA adapter configurations with General adapter
LORA_ADAPTERS = {
    "General": {
        "path": "/app/loras/llama-8B-alpaca-2k",
        "description": "General purpose assistant for everyday conversations and casual chat",
        "system_prompt": "You are a helpful, friendly assistant. Provide clear, conversational responses to any topic while being engaging and informative."
    },
    "Chemistry": {
        "path": "/app/loras/llama-8B-chemistry",
        "description": "Chemistry expert for compounds, reactions, and laboratory procedures",
        "system_prompt": "You are a chemistry expert. Provide clear, accurate answers about chemistry topics including formulas, reactions, and molecular structures. Always prioritize safety information when relevant."
    },
    "Finance": {
        "path": "/app/loras/llama-8B-finance-alpaca", 
        "description": "Finance expert for investments, banking, and economic topics",
        "system_prompt": "You are a finance expert. Help with financial concepts, investments, market analysis, and economic topics. Provide practical, actionable financial advice.",
    },
    "AI": {
        "path": "/app/loras/llama-8B-gpt-ai",
        "description": "AI and technology expert for machine learning and programming",
        "system_prompt": "You are an AI and machine learning expert. Explain technical concepts, programming, and AI topics clearly. Provide comprehensive explanations with examples when helpful.",
    },
    "Medical": {
        "path": "/app/loras/llama-8B-medical-alpaca",
        "description": "Medical expert for healthcare and health information",
        "system_prompt": "You are a medical knowledge expert. Provide informative responses about health and medical topics. Always recommend consulting healthcare professionals for medical advice."
    }
}

@st.cache_resource
def load_base_model():
    """Load the base model and tokenizer."""
    base_model_path = "/app/base_model/llama-3.1-8B-instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        return base_model, tokenizer
    except Exception as e:
        st.error(f"Error loading base model: {e}")
        return None, None

def load_adapter(adapter_name):
    """Load a specific LoRA adapter."""
    if adapter_name in st.session_state.loaded_adapters:
        return st.session_state.loaded_adapters[adapter_name]
    
    try:
        adapter_path = LORA_ADAPTERS[adapter_name]["path"]
        model = PeftModel.from_pretrained(st.session_state.base_model, adapter_path)
        model.eval()
        st.session_state.loaded_adapters[adapter_name] = model
        return model
    except Exception as e:
        st.error(f"Error loading {adapter_name} adapter: {e}")
        return None

def auto_load_all_adapters():
    """Automatically load all available adapters."""
    success_count = 0
    total_adapters = len(LORA_ADAPTERS)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (adapter_name, adapter_info) in enumerate(LORA_ADAPTERS.items()):
        status_text.text(f"Loading {adapter_name} adapter...")
        progress_bar.progress((i + 1) / total_adapters)
        
        if adapter_name not in st.session_state.loaded_adapters:
            model = load_adapter(adapter_name)
            if model:
                success_count += 1
        else:
            success_count += 1
    
    progress_bar.empty()
    status_text.empty()
    
    return success_count, total_adapters

def get_conversation_history():
    """Get the last 2 user-assistant exchanges from chat history."""
    history = []
    
    # Get all user and assistant messages in order
    user_messages = []
    assistant_messages = []
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            user_messages.append(message["content"])
        elif message["role"] == "assistant":
            assistant_messages.append(message["content"])
    
    # Pair them up and take the last 2
    pairs = list(zip(user_messages, assistant_messages))
    recent_pairs = pairs[-2:] if len(pairs) >= 2 else pairs
    
    # Convert to the expected format
    for user_msg, assistant_msg in recent_pairs:
        history.append({
            "user": user_msg,
            "assistant": assistant_msg
        })
    
    return history

# AI-Powered Router Agent
def enhanced_router_agent(state: AgentState) -> AgentState:
    """
    AI-Powered Router Agent: Uses the base model to intelligently select the best domain expert.
    """
    query = state["query"]
    
    # Initialize reasoning
    reasoning_parts = [f"🔍 **Query**: '{query}'"]
    
    try:
        # AI selection prompt
        ai_prompt = f"""Analyze this user query and select the most appropriate domain expert to handle it.

Query: "{query}"

Available Domain Experts:
1. General - For casual conversation, greetings, general knowledge, everyday questions, and topics that don't fit other categories
2. Chemistry - For chemical compounds, reactions, molecules, laboratory procedures, chemical safety, and chemistry-related questions
3. Finance - For money, investments, stocks, banking, economics, trading, companies, financial planning, and business topics
4. AI - For artificial intelligence, machine learning, programming, algorithms, reinforcement learning, neural networks, and technology topics
5. Medical - For health, diseases, treatments, symptoms, anatomy, medicine, medical advice, and healthcare topics

Instructions: 
- Analyze the query carefully
- Consider the main topic and intent
- Choose the domain expert that best matches the query
- If unsure or the query is general/casual, choose General
- Respond with ONLY the domain name (General, Chemistry, Finance, AI, or Medical)

Selected Domain:"""

        # Generate AI selection
        inputs = st.session_state.tokenizer(ai_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(st.session_state.base_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = st.session_state.base_model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
            )
        
        response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        ai_selection = response[len(ai_prompt):].strip()
        
        # Extract the first word/domain name from AI response
        ai_domain = ai_selection.split()[0] if ai_selection.split() else ""
        
        # Map AI response to valid domains
        selected_adapter = None
        for adapter_name in LORA_ADAPTERS.keys():
            if adapter_name.lower() in ai_domain.lower():
                selected_adapter = adapter_name
                break
        
        if selected_adapter:
            reasoning_parts.extend([
                f"🤖 **AI Selection**: {selected_adapter}",
                f"🎯 **AI Response**: '{ai_selection.strip()}'",
                f"✅ **Decision**: {selected_adapter} via AI semantic analysis"
            ])
        else:
            # Fallback to General if AI response is unclear
            selected_adapter = "General"
            reasoning_parts.extend([
                f"⚠️ **AI Response Unclear**: '{ai_selection}'",
                f"🔄 **Fallback**: Using General for safety"
            ])
        
        return {
            **state,
            "selected_adapter": selected_adapter,
            "reasoning": "\n".join(reasoning_parts),
            "conversation_history": get_conversation_history()
        }
        
    except Exception as e:
        # Emergency fallback
        error_reasoning = f"❌ **Router Error**: {str(e)}\n🔄 **Emergency Fallback**: Using General"
        return {
            **state,
            "selected_adapter": "General",
            "reasoning": error_reasoning,
            "conversation_history": get_conversation_history()
        }

# Enhanced Expert Agent with conversation history
def enhanced_expert_agent(state: AgentState) -> AgentState:
    """
    Enhanced Expert Agent: Uses the selected adapter to generate domain-specific responses
    with conversation history and improved prompt formatting.
    """
    query = state["query"]
    adapter_name = state["selected_adapter"]
    conversation_history = state.get("conversation_history", [])
    
    if adapter_name not in st.session_state.loaded_adapters:
        return {
            **state,
            "response": f"Error: {adapter_name} adapter not loaded. Please initialize the system first."
        }
    
    model = st.session_state.loaded_adapters[adapter_name]
    system_prompt = LORA_ADAPTERS[adapter_name]["system_prompt"]
    
    try:
        # Build conversation history context
        history_context = ""
        if conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for i, exchange in enumerate(conversation_history, 1):
                history_context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        
        # Enhanced prompt formatting based on adapter type
        if adapter_name == "General":
            # More conversational format for general chat
            response_prompt = f"""<|system|>
{system_prompt}{history_context}
<|user|>
{query}
<|assistant|>
"""
        else:
            # Professional format for specialized domains
            response_prompt = f"""System: {system_prompt}{history_context}

Query: {query}
Response:"""
        
        # Tokenize and prepare inputs
        inputs = st.session_state.tokenizer(response_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(st.session_state.base_model.device) for k, v in inputs.items()}
        
        # Generate response with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=st.session_state.tokenizer.eos_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        full_response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if adapter_name == "General":
            if "<|assistant|>" in full_response:
                response = full_response.split("<|assistant|>")[-1].strip()
            else:
                response = full_response[len(response_prompt):].strip()
        else:
            if "Response:" in full_response:
                response = full_response.split("Response:")[-1].strip()
            else:
                response = full_response[len(response_prompt):].strip()
        
        # Clean up response
        response = response.replace("<|user|>", "").replace("<|system|>", "").strip()
        
        # Remove any remaining conversation history from response
        if "Previous conversation:" in response:
            response = response.split("Previous conversation:")[0].strip()
        
        # Ensure response is not empty
        if not response:
            response = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
        
        return {
            **state,
            "response": response
        }
        
    except Exception as e:
        return {
            **state,
            "response": f"Error generating response: {str(e)}. Please try again."
        }

# Create the workflow
def create_workflow() -> StateGraph:
    """Create the LangGraph workflow with enhanced agents."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", enhanced_router_agent)
    workflow.add_node("expert", enhanced_expert_agent)
    
    # Add edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "expert")
    workflow.add_edge("expert", END)
    
    return workflow.compile()

# Sidebar for system initialization
with st.sidebar:
    st.header("🔧 System Control")
    
    # System status
    if st.session_state.base_model is None:
        st.error("❌ System not initialized")
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Loading base model and adapters..."):
                # Load base model first
                base_model, tokenizer = load_base_model()
                if base_model and tokenizer:
                    st.session_state.base_model = base_model
                    st.session_state.tokenizer = tokenizer
                    st.success("✅ Base model loaded!")
                    
                    # Then load all adapters
                    success_count, total_adapters = auto_load_all_adapters()
                    if success_count == total_adapters:
                        st.success(f"✅ All {total_adapters} adapters loaded!")
                    else:
                        st.warning(f"⚠️ {success_count}/{total_adapters} adapters loaded")
                    
                    # Initialize workflow
                    st.session_state.workflow = create_workflow()
                    st.success("🔗 Workflow initialized!")
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to load base model")
    else:
        st.success("✅ Base model loaded")
        
        # Adapter loading section
        st.subheader("🧩 Adapters")
        
        # Show loaded adapters
        if st.session_state.loaded_adapters:
            st.write("**Loaded Adapters:**")
            for adapter_name in st.session_state.loaded_adapters.keys():
                st.write(f"✅ {adapter_name}")
        
        # Initialize workflow
        if st.session_state.workflow is None and st.session_state.loaded_adapters:
            st.session_state.workflow = create_workflow()
            st.success("🔗 Workflow initialized!")
        
        # System stats
        st.subheader("📊 System Stats")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            st.write(f"🔥 GPU Memory: {memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
st.header("💬 Chat Interface")

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
    # Check if system is ready
    if not st.session_state.workflow or not st.session_state.loaded_adapters:
        st.error("❌ Please initialize the system first using the sidebar controls.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Process query through workflow
    with st.spinner("🤔 Processing your query..."):
        try:
            # Create initial state
            initial_state = {
                "query": prompt,
                "selected_adapter": "",
                "response": "",
                "reasoning": "",
                "messages": [],
                "conversation_history": []
            }
            
            # Run workflow
            result = st.session_state.workflow.invoke(initial_state)
            
            # Display reasoning
            with st.expander("🔍 Router Agent Decision", expanded=False):
                st.markdown(result["reasoning"])
            
            # Add reasoning to chat history
            st.session_state.chat_history.append({
                "role": "reasoning", 
                "content": result["reasoning"]
            })
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(f"🧠 **{result['selected_adapter']} via Adaptive Minds**")
                st.write(result["response"])
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "adapter": result["selected_adapter"]
            })
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Powered by LangGraph, LoRA adapters, and Streamlit*")

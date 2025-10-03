from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import gc
import json
import re
from typing import TypedDict, Annotated, Literal, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import operator
import uvicorn

app = FastAPI(
    title="Adaptive Minds API",
    description="LangGraph-powered multi-agent system with intelligent routing",
    version="1.0.0"
)

# Define the state that will be passed between agents
class AgentState(TypedDict):
    query: str
    selected_adapter: str
    response: str
    reasoning: str
    messages: Annotated[List[dict], operator.add]
    conversation_history: List[dict]

# Global state management
class GlobalState:
    base_model = None
    tokenizer = None
    loaded_adapters = {}
    workflow = None
    chat_history = []  # Global chat history
    adapter_config = None  # Loaded from JSON
    router_prompt_template = None  # Dynamic router prompt

global_state = GlobalState()

# ============================================================================
# LOAD ADAPTER CONFIGURATION FROM JSON
# ============================================================================
def load_adapter_config(config_path="/app/build/metadata.json"):
    """Load LoRA adapter configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Adapter configuration file not found: {config_path}. Please create metadata.json"
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid JSON in adapter configuration: {str(e)}"
        )

def build_router_prompt(query: str, config: dict) -> str:
    """Build dynamic router prompt from configuration."""
    adapters = config['adapters']
    template = config['router_prompt_template']
    
    # Build domain list with descriptions (auto-generated from keywords)
    domain_list_items = []
    for idx, (domain_name, adapter_info) in enumerate(adapters.items(), 1):
        # Auto-generate routing description from keywords
        keywords = adapter_info.get('keywords', [])
        routing_desc = f"For {', '.join(keywords[:8])}" + (" and related topics" if len(keywords) > 8 else "")
        
        domain_list_items.append(
            f"{idx}. {domain_name} - {routing_desc}"
        )
    domain_list = "\n".join(domain_list_items)
    
    # Get domain names for instruction
    domain_names = ", ".join(adapters.keys())
    
    # Fill in template
    prompt = template.format(
        query=query,
        domain_list=domain_list,
        domain_names=domain_names
    )
    
    return prompt

# ============================================================================
# LoRA ADAPTER CONFIGURATION - Now loaded from metadata.json
# ============================================================================
# To add your own custom LoRA adapter:
# 1. Edit metadata.json
# 2. Add a new entry to the "adapters" object with:
#    - Unique domain name (key)
#    - path: Path to your adapter directory
#    - description: Brief description
#    - system_prompt: Defines the expert's role and behavior
#    - keywords: List of relevant keywords (routing description auto-generated from these)
# 3. Restart the server - the AI router will automatically recognize it!
# ============================================================================

LORA_ADAPTERS = {}  # Will be loaded from JSON at startup

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
        raise HTTPException(status_code=500, detail=f"Error loading base model: {str(e)}")

def load_adapter(adapter_name):
    """Load a specific LoRA adapter."""
    if adapter_name in global_state.loaded_adapters:
        return global_state.loaded_adapters[adapter_name]
    
    try:
        adapter_path = LORA_ADAPTERS[adapter_name]["path"]
        model = PeftModel.from_pretrained(global_state.base_model, adapter_path)
        model.eval()
        global_state.loaded_adapters[adapter_name] = model
        return model
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_name}' not found in configuration")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading {adapter_name} adapter: {str(e)}")

def get_conversation_history():
    """Get the last 2 user-assistant exchanges from chat history."""
    history = []
    
    # Get all user and assistant messages in order
    user_messages = []
    assistant_messages = []
    
    for message in global_state.chat_history:
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

def enhanced_router_agent(state: AgentState) -> AgentState:
    """
    AI-Powered Router Agent: Uses the base model to intelligently select the best domain expert.
    Router prompt is dynamically generated from metadata.json configuration.
    """
    query = state["query"]
    
    # Initialize reasoning
    reasoning_parts = [f"🔍 **Query**: '{query}'"]
    
    try:
        # Build dynamic AI selection prompt from configuration
        ai_prompt = build_router_prompt(query, global_state.adapter_config)

        # Generate AI selection
        inputs = global_state.tokenizer(ai_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(global_state.base_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = global_state.base_model.generate(
                **inputs,
                max_new_tokens=15,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=global_state.tokenizer.eos_token_id,
            )
        
        response = global_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            "conversation_history": state["conversation_history"]
        }
        
    except Exception as e:
        # Emergency fallback
        error_reasoning = f"❌ **Router Error**: {str(e)}\n🔄 **Emergency Fallback**: Using General"
        return {
            **state,
            "selected_adapter": "General",
            "reasoning": error_reasoning,
            "conversation_history": state["conversation_history"]
        }

def enhanced_expert_agent(state: AgentState) -> AgentState:
    """
    Enhanced Expert Agent: Uses the selected adapter to generate domain-specific responses
    with conversation history and improved prompt formatting.
    """
    query = state["query"]
    adapter_name = state["selected_adapter"]
    conversation_history = state["conversation_history"]
    
    if adapter_name not in global_state.loaded_adapters:
        return {
            **state,
            "response": f"Error: {adapter_name} adapter not loaded. Please initialize the system first."
        }
    
    model = global_state.loaded_adapters[adapter_name]
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
        inputs = global_state.tokenizer(response_prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(global_state.base_model.device) for k, v in inputs.items()}
        
        # Generate response with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=global_state.tokenizer.eos_token_id,
                eos_token_id=global_state.tokenizer.eos_token_id
            )
        
        # Decode and clean response
        full_response = global_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
        
        # Remove hallucinated links and references
        # Remove <link>, [link], or generic link mentions
        response = re.sub(r'<link>|<\/link>|\[link\]|\(link\)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'<url>|<\/url>|\[url\]|\(url\)', '', response, flags=re.IGNORECASE)
        # Remove phrases suggesting external lookup
        response = re.sub(r'you can find .{0,50}(on github|in the .{0,20} repo|on the website)', '', response, flags=re.IGNORECASE)
        response = re.sub(r'check out .{0,50}(github|repository|website|documentation)', '', response, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and punctuation
        response = re.sub(r'\s+', ' ', response).strip()
        response = re.sub(r'\.{2,}', '.', response)
        
        # Ensure response is not empty
        if not response or len(response) < 10:
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

# API Models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    selected_adapter: str
    reasoning: str

class SystemStatus(BaseModel):
    is_initialized: bool
    loaded_adapters: List[str]
    gpu_memory_allocated: Optional[float]
    gpu_memory_reserved: Optional[float]

# Add startup event handler
@app.on_event("startup")
async def startup_event():
    """Initialize the system when the server starts."""
    global LORA_ADAPTERS
    try:
        # Load adapter configuration from JSON
        print("📖 Loading adapter configuration from metadata.json...")
        config = load_adapter_config()
        global_state.adapter_config = config
        LORA_ADAPTERS = config['adapters']
        global_state.router_prompt_template = config['router_prompt_template']
        print(f"✅ Loaded {len(LORA_ADAPTERS)} adapter configurations")
        
        # Load base model
        print("🔄 Loading base model...")
        base_model, tokenizer = load_base_model()
        global_state.base_model = base_model
        global_state.tokenizer = tokenizer
        print("✅ Base model loaded successfully")
        
        # Load all adapters
        print("🔄 Loading LoRA adapters...")
        for adapter_name in LORA_ADAPTERS:
            load_adapter(adapter_name)
            print(f"  ✅ Loaded {adapter_name}")
        
        # Initialize workflow
        global_state.workflow = create_workflow()
        
        print("\n🎉 ALL SYSTEMS READY!")
        print(f"🧠 Available domains: {', '.join(list(global_state.loaded_adapters.keys()))}")
        print(f"📊 Configuration version: {config.get('_metadata', {}).get('version', 'N/A')}\n")
        
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")
        import traceback
        traceback.print_exc()

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not global_state.workflow or not global_state.loaded_adapters:
        raise HTTPException(
            status_code=503, 
            detail="System initialization failed. Please check server logs and restart the server."
        )
    
    try:
        # Create initial state
        initial_state = {
            "query": request.query,
            "selected_adapter": "",
            "response": "",
            "reasoning": "",
            "messages": [],
            "conversation_history": get_conversation_history()
        }
        
        # Run workflow
        result = global_state.workflow.invoke(initial_state)
        
        # Update chat history
        global_state.chat_history.extend([
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": result["response"], "adapter": result["selected_adapter"]},
            {"role": "reasoning", "content": result["reasoning"]}
        ])
        
        return ChatResponse(
            response=result["response"],
            selected_adapter=result["selected_adapter"],
            reasoning=result["reasoning"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=SystemStatus)
async def get_status():
    gpu_memory_allocated = None
    gpu_memory_reserved = None
    
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
    
    return SystemStatus(
        is_initialized=global_state.workflow is not None,
        loaded_adapters=list(global_state.loaded_adapters.keys()),
        gpu_memory_allocated=gpu_memory_allocated,
        gpu_memory_reserved=gpu_memory_reserved
    )

@app.delete("/chat/history")
async def clear_chat_history():
    """Clear the global chat history."""
    global_state.chat_history = []
    return {"message": "Chat history cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
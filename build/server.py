from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import gc
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

global_state = GlobalState()

# Enhanced LoRA adapter configurations with General adapter
LORA_ADAPTERS = {
    "General": {
        "path": "/app/loras/llama-8B-alpaca-2k",
        "description": "General purpose assistant for everyday conversations, casual chat, general knowledge, greetings, and miscellaneous tasks",
        "system_prompt": "You are a helpful, friendly assistant. Provide clear, conversational responses to any topic while being engaging and informative. Be natural and personable in your responses.",
        "keywords": "general, chat, conversation, help, question, talk, discuss, explain, tell me, how are you, hello, hi, greetings, thanks, please, what, how, why, when, where, who"
    },
    "Chemistry": {
        "path": "/app/loras/llama-8B-chemistry",
        "description": "Chemistry expert specializing in chemical compounds, reactions, molecular structures, periodic table, acids, bases, laboratory procedures",
        "system_prompt": "You are a chemistry expert. Provide clear, accurate answers about chemistry topics including chemical formulas, reactions, molecular structures, and laboratory procedures. Always prioritize safety information when relevant.",
        "keywords": "chemistry, chemical, molecule, compound, formula, acid, base, element, reaction, benzene, H2O, periodic table, organic, inorganic, crocin, hexen, ol, skin, acne, safety"
    },
    "Finance": {
        "path": "/app/loras/llama-8B-finance-alpaca", 
        "description": "Finance expert specializing in investments, banking, trading, economics, financial planning, compound interest, stocks, bonds",
        "system_prompt": "You are a finance expert. Help with financial concepts, investments, market analysis, and economic topics. Provide practical, actionable financial advice.",
        "keywords": "finance, financial, money, investment, stock, bond, trading, banking, loan, interest, economics, budget, cryptocurrency, general electric, ge, company, business, investing, strategies"
    },
    "AI": {
        "path": "/app/loras/llama-8B-gpt-ai",
        "description": "AI and technology expert specializing in machine learning, neural networks, programming, algorithms, data science, reinforcement learning",
        "system_prompt": "You are an AI and machine learning expert. Explain technical concepts, programming, and AI topics clearly. Provide comprehensive explanations with examples when helpful.",
        "keywords": "AI, artificial intelligence, machine learning, neural network, algorithm, programming, model, training, reinforcement learning, RL, deep learning, NLP, llama, code, python, javascript"
    },
    "Medical": {
        "path": "/app/loras/llama-8B-medical-alpaca",
        "description": "Medical expert specializing in healthcare, diseases, treatments, anatomy, symptoms, medical procedures, health information",
        "system_prompt": "You are a medical knowledge expert. Provide informative responses about health, medical topics, and healthcare information. Always recommend consulting healthcare professionals for medical advice.",
        "keywords": "medical, medicine, health, disease, treatment, doctor, patient, symptom, diagnosis, healthcare, anatomy, physiology, skin, acne, safety, ingredient, drug, depressed, depression"
    }
}

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
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
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
    try:
        # Load base model
        base_model, tokenizer = load_base_model()
        global_state.base_model = base_model
        global_state.tokenizer = tokenizer
        
        # Load all adapters
        for adapter_name in LORA_ADAPTERS:
            load_adapter(adapter_name)
        
        # Initialize workflow
        global_state.workflow = create_workflow()
        
        print("✅ System initialized successfully!")
        print(f"🧠 Loaded adapters: {list(global_state.loaded_adapters.keys())}")
        
    except Exception as e:
        print(f"❌ Error initializing system: {str(e)}")

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
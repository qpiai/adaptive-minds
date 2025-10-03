# 🧠 Adaptive Minds Playground

**The complete toolkit for training, managing, and deploying specialized AI experts.**

Train custom LoRA adapters, manage multiple models, and deploy intelligent routing systems with ease.

## 🚀 Quick Start

### Option 1: Use Pre-trained Models (5 minutes)
```bash
# 1. Configure models (use example or customize)
cp models_config.yaml models_config.yaml.backup  # Backup if exists
# Edit models_config.yaml to your needs

# 2. Download all models
python manage_models.py sync

# 3. Start the server
python server.py
```

### Option 2: Train Your Own Expert (30 minutes)
```bash
# 1. Train a LoRA on your dataset
python train/train.py --dataset your-dataset --lora-name my-expert

# 2. Register it automatically
python manage_models.py add-lora \
  --name "MyExpert" \
  --path "./loras/my-expert" \
  --description "Expert in my domain" \
  --system-prompt "You are an expert in..." \
  --keywords domain specific keywords

# 3. Rebuild and start
python manage_models.py rebuild
python server.py
```

### Option 3: Mix & Match (Flexible)
```bash
# Use some pre-trained + some custom + some from other users
# Edit models_config.yaml:
# - Enable/disable any adapter
# - Add your HF repos
# - Point to local LoRAs
# - Mix sources freely
```

---

## 📋 What You Get

| Feature | Description |
|---------|-------------|
| **🎯 Smart Routing** | AI automatically selects the best expert for each query |
| **🔧 Easy Training** | Train LoRAs with any HuggingFace dataset |
| **📦 Model Management** | Download, sync, and manage models with CLI |
| **🌐 Web Interface** | Beautiful Streamlit frontend |
| **⚡ Fast API** | RESTful API for integration |
| **🔍 Auto-Discovery** | Find and register local LoRAs automatically |

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for 8B models)

### Install Dependencies
```bash
# Basic server dependencies
uv sync

# Add training dependencies (optional)
uv sync --extra training
```

### Environment Setup
```bash
# For HuggingFace models (optional)
export HF_TOKEN="your_huggingface_token_here"

# For CUDA (if available)
export CUDA_VISIBLE_DEVICES=0
```

---

## 📖 Detailed Usage

### 1. Model Configuration

Edit `models_config.yaml` to define your models:

```yaml
# Base model (required)
base_model:
  name: "llama-3.1-8B-instruct"
  source: "huggingface"  # or "local"
  huggingface_id: "meta-llama/Llama-3.1-8B-Instruct"
  local_path: "./base_model/llama-3.1-8B-instruct"

# LoRA adapters (as many as you want)
lora_adapters:
  - name: "General"
    source: "huggingface"  # or "local"
    huggingface_id: "pavan01729/llama-8B-alpaca-2k"
    local_path: "./loras/llama-8B-alpaca-2k"
    description: "General purpose assistant"
    system_prompt: "You are a helpful assistant..."
    keywords: [general, chat, conversation]
    enabled: true  # Enable/disable as needed
```

### 2. Model Management CLI

```bash
# List all configured models
python manage_models.py list

# Download/sync all models
python manage_models.py sync

# Validate configuration
python manage_models.py validate

# Discover local LoRAs
python manage_models.py discover

# Rebuild metadata.json
python manage_models.py rebuild
```

### 3. Training Custom LoRAs

```bash
# Train on any HuggingFace dataset
python train/train.py \
  --dataset "medalpaca/medical_meadow_medical_flashcards" \
  --lora-name "medical-expert" \
  --steps 500

# Train with custom parameters
python train/train.py \
  --dataset "squad" \
  --lora-name "qa-expert" \
  --steps 1000 \
  --batch-size 4 \
  --lora-r 32
```

### 4. Registering New LoRAs

```bash
# Add a local LoRA
python manage_models.py add-lora \
  --name "MyExpert" \
  --path "./loras/my-expert" \
  --description "Expert in my domain" \
  --system-prompt "You are an expert in..." \
  --keywords domain specific keywords

# Enable/disable adapters
python manage_models.py enable --name "MyExpert"
python manage_models.py disable --name "OldExpert"
```

### 5. Starting the Server

```bash
# Start with all configured models
python server.py

# Access the web interface
# http://localhost:8501

# Or use the API
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question here"}'
```

---

## 🎯 Popular Use Cases

### 1. **Domain-Specific Chatbots**
```bash
# Train a legal expert
python train/train.py --dataset "pile-of-law/pile-of-law" --lora-name "legal-expert"

# Train a coding assistant
python train/train.py --dataset "codeparrot/github-code" --lora-name "coding-expert"
```

### 2. **Multi-Domain Support System**
```yaml
# In models_config.yaml - mix different sources
lora_adapters:
  - name: "CustomerSupport"
    source: "huggingface"
    huggingface_id: "company/support-lora"
    # ...
  - name: "TechnicalExpert"
    source: "local"
    local_path: "./loras/technical-expert"
    # ...
  - name: "SalesExpert"
    source: "huggingface"
    huggingface_id: "sales-ai/sales-lora"
    # ...
```

### 3. **Research Assistant**
```bash
# Train on scientific papers
python train/train.py --dataset "scientific_papers" --lora-name "research-expert"

# Train on medical data
python train/train.py --dataset "medalpaca/medical_meadow" --lora-name "medical-expert"
```

---

## 📊 Available Datasets

| Domain | Dataset | Description |
|--------|---------|-------------|
| **Medical** | `medalpaca/medical_meadow_medical_flashcards` | Medical knowledge |
| **Legal** | `pile-of-law/pile-of-law` | Legal documents |
| **Code** | `codeparrot/github-code` | Programming |
| **Math** | `microsoft/orca-math-word-problems-200k` | Mathematics |
| **Science** | `sciq` | Science questions |
| **General** | `squad`, `squad_v2` | General Q&A |

---

## 🔧 Advanced Configuration

### Custom Base Models
```yaml
base_model:
  name: "custom-model"
  source: "local"
  local_path: "/path/to/your/model"
```

### Custom Router Prompts
```yaml
router:
  prompt_template: |
    Your custom routing logic here...
    Query: "{query}"
    Experts: {domain_list}
    Choose: {domain_names}
```

### Training Parameters
```bash
# Advanced training
python train/train.py \
  --dataset "your-dataset" \
  --lora-name "expert" \
  --steps 2000 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --lora-r 64
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "No HF_TOKEN provided"**
```bash
export HF_TOKEN="your_token_here"
python manage_models.py sync
```

**2. "CUDA out of memory"**
```bash
# Reduce batch size
python train/train.py --batch-size 1 --dataset your-dataset --lora-name expert
```

**3. "Model not found"**
```bash
# Check if model exists
python manage_models.py validate
python manage_models.py list
```

**4. "LoRA not working"**
```bash
# Rebuild metadata
python manage_models.py rebuild
# Restart server
python server.py
```

### Performance Tips

- **Memory**: Use smaller batch sizes for training
- **Speed**: Enable mixed precision (automatic)
- **Quality**: More training steps ≠ always better
- **Storage**: Clean up old checkpoints regularly

---

## 📁 Directory Structure

```
playground/
├── models_config.yaml      # Your model configuration
├── manage_models.py        # Model management CLI
├── server.py              # Main server
├── metadata.json          # Generated from config
├── train/
│   ├── train.py           # Training script
│   └── README.md          # Training documentation
├── loras/                 # Your LoRA adapters
│   ├── llama-8B-alpaca-2k/
│   ├── my-custom-expert/
│   └── ...
└── base_model/            # Base model
    └── llama-3.1-8B-instruct/
```

---

## 🤝 Contributing

1. **Add new datasets**: Update the training documentation
2. **Improve routing**: Enhance the router logic
3. **Add features**: Extend the CLI tools
4. **Fix bugs**: Report and fix issues

---

## 📚 Additional Resources

- **Training Guide**: See `train/README.md`
- **API Documentation**: Check `server.py` endpoints
- **Model Management**: Use `python manage_models.py --help`
- **Web Interface**: Access `http://localhost:8501`

---

**Ready to build your AI expert system? Start with `python manage_models.py sync`!** 🚀

# 🚀 Quick Start Guide

**Get Adaptive Minds running in 5 minutes with Docker**

## Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU with CUDA support
- 16GB+ VRAM (recommended)
- Internet connection for model downloads

## Step 1: Clone Repository

```bash
git clone https://github.com/qpiai/adaptive-minds.git
cd adaptive-minds
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your HuggingFace token
nano .env
```

Add this line:
```
HF_TOKEN=your_huggingface_token_here
```

> **💡 Get your token**: https://huggingface.co/settings/tokens  
> **💡 Accept Llama license**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

## Step 3: Start the System

```bash
docker compose up
```

This will:
1. 📥 Download base model (~17 files, ~16GB)
2. 📥 Download 5 LoRA adapters (~42 files each)
3. 🚀 Start FastAPI server (port 8765)
4. 🎨 Start Streamlit UI (port 8501)

**⏱️ First run takes ~15-20 minutes** (downloads models)  
**⚡ Subsequent runs**: Instant (uses cached models)

## Step 4: Access the System

### Web Interface
Open your browser: **http://localhost:8501**

![Streamlit UI](./assets/adaptive_minds_v2_lite.gif)

### API Endpoint
```bash
# Check status
curl http://localhost:8765/status

# Send a query
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the chemical formula for water?"
  }'
```

## What You Get

### 🧠 5 Domain Experts

| Expert | Handles | Example Queries |
|--------|---------|----------------|
| **General** | Everyday conversation | "Hello", "How are you?", "Tell me about..." |
| **Chemistry** | Compounds & reactions | "What is H2O?", "Explain photosynthesis" |
| **Finance** | Money & investments | "How do I invest in stocks?", "What is a bond?" |
| **AI** | Tech & programming | "How to train a neural network?", "Explain transformers" |
| **Medical** | Health information | "What causes a headache?", "Symptoms of flu?" |

### 🎯 Smart Routing

The system automatically picks the best expert:
```
Query: "How do I train a neural network?" → AI Expert ✅
Query: "What is H2O?" → Chemistry Expert ✅
Query: "Hello, how are you?" → General Expert ✅
```

## Verify Everything Works

### 1. Check Service Status
```bash
# Check if containers are running
docker compose ps

# Should show:
# adaptive-minds-server   running   0.0.0.0:8765->8765/tcp, 0.0.0.0:8501->8501/tcp
```

### 2. Check API Health
```bash
curl http://localhost:8765/status | jq
```

Expected output:
```json
{
  "is_initialized": true,
  "loaded_adapters": ["General", "Chemistry", "Finance", "AI", "Medical"],
  "gpu_memory_allocated": 15.11,
  "gpu_memory_reserved": 16.52
}
```

### 3. Test Each Expert

```bash
# Test General
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'

# Test Chemistry
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the chemical formula for water?"}'

# Test AI
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I train a neural network?"}'
```

## Common Commands

```bash
# Start system
docker compose up

# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Stop system
docker compose down

# Restart
docker compose restart

# Rebuild after changes
docker compose up --build
```

## Troubleshooting

### Issue: Port already in use

```bash
# Check what's using the port
lsof -i:8765
lsof -i:8501

# Kill the process
kill -9 <PID>
```

### Issue: CUDA out of memory

```bash
# Edit docker-compose.yml to use smaller batch sizes
# Or reduce number of loaded experts in build/metadata.json
```

### Issue: Model download fails

```bash
# Check HF_TOKEN in .env
cat .env | grep HF_TOKEN

# Check if you accepted the Llama license
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# Try manual download
docker compose exec adaptive-minds-server python /app/build/download_models.py
```

### Issue: Container won't start

```bash
# Check logs
docker compose logs

# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Rebuild
docker compose down
docker compose build --no-cache
docker compose up
```

## What's Next?

### Use the Web Interface
1. Open http://localhost:8501
2. Type your questions
3. See which expert handles each query
4. View conversation history

### Explore the API
- **POST /chat** - Send queries
- **GET /status** - System status
- **GET /available_experts** - List experts

### Customize & Train Your Own
Want to add custom experts or train on your own data?  
**👉 See [playground/README.md](playground/README.md) for development guide**

### Production Deployment
- Use reverse proxy (nginx/traefik)
- Add authentication
- Scale with Kubernetes
- Monitor with Prometheus/Grafana

## Performance Tips

### Optimize GPU Usage
```bash
# Set specific GPU
CUDA_VISIBLE_DEVICES=0 docker compose up

# Use multiple GPUs (if available)
# Edit docker-compose.yml:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           device_ids: ['0', '1']
```

### Speed Up First Start
```bash
# Pre-download models before Docker
python build/download_models.py

# Then start Docker
docker compose up
```

## System Requirements

### Minimum
- 16GB VRAM (NVIDIA GPU)
- 32GB RAM
- 50GB disk space
- Ubuntu 20.04+

### Recommended
- 24GB+ VRAM (RTX 3090, A5000, etc.)
- 64GB RAM
- 100GB SSD
- Ubuntu 22.04

### Cloud Deployment
Works great on:
- AWS p3/p4 instances
- GCP GPU instances
- Azure NC series
- Lambda Labs
- RunPod

## Support

- 📖 **Full Documentation**: [README.md](README.md)
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/yourusername/adaptive-minds/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/adaptive-minds/discussions)
- 🤗 **Models**: [HuggingFace Collection](https://huggingface.co/collections/pavan01729/adaptive-minds-68cbab3565664604be49a462)

---

**🎉 Congratulations! You now have a working multi-agent AI system with specialized domain experts!**


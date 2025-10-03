# 🎯 Adding New LoRA Experts

**Complete guide to training and deploying custom domain experts**

This guide covers two approaches:
1. **Quick Deploy** - Use existing LoRAs from HuggingFace
2. **Train Custom** - Train your own domain-specific experts

## 🚀 Quick Deploy (Use Existing LoRAs)

### From HuggingFace

Add any compatible LoRA from HuggingFace:

```yaml
# In playground/models_config.yaml:
lora_adapters:
  - name: "LegalExpert"
    source: "huggingface"
    huggingface_id: "someuser/legal-llama-lora"  # Any compatible LoRA
    local_path: "./loras/legal-expert"
    description: "Legal expert for law-related questions"
    system_prompt: "You are a legal expert. Provide clear, accurate answers about legal topics."
    keywords:
      - legal
      - law
      - contract
      - regulation
      - attorney
    enabled: true
```

Then sync and deploy:
```bash
cd playground
python manage_models.py sync      # Download the model
python manage_models.py rebuild   # Update configuration
python server.py                  # Start server
```

### From Local Path

If you have a LoRA locally:

```yaml
lora_adapters:
  - name: "CustomExpert"
    source: "local"
    local_path: "./loras/my-custom-expert"  # Your local LoRA
    description: "My custom trained expert"
    system_prompt: "You are an expert in..."
    keywords: [custom, domain, specific]
    enabled: true
```

Then:
```bash
python manage_models.py rebuild
python server.py
```

---

## 🧪 Train Custom Expert

### Prerequisites

```bash
# Install training dependencies
uv sync --extra training

# Or with pip
pip install unsloth datasets trl transformers
```

### Step 1: Choose Your Dataset

Find a dataset on HuggingFace that matches your domain:

| Domain | Example Datasets |
|--------|------------------|
| **Medical** | `medalpaca/medical_meadow_medical_flashcards` |
| **Legal** | `pile-of-law/pile-of-law` |
| **Code** | `codeparrot/github-code` |
| **Math** | `microsoft/orca-math-word-problems-200k` |
| **Science** | `sciq` |
| **General QA** | `squad`, `squad_v2` |

**Browse datasets**: https://huggingface.co/datasets

### Step 2: Train the LoRA

```bash
cd playground/train

# Basic training
python train.py \
  --dataset "medalpaca/medical_meadow_medical_flashcards" \
  --lora-name "medical-expert" \
  --steps 500

# Advanced training
python train.py \
  --dataset "your-dataset" \
  --lora-name "custom-expert" \
  --steps 1000 \
  --batch-size 4 \
  --learning-rate 2e-4 \
  --lora-r 32
```

**Training parameters:**
- `--dataset`: HuggingFace dataset name
- `--lora-name`: Name for your LoRA (creates `./loras/{lora-name}`)
- `--steps`: Training steps (500-2000 recommended)
- `--batch-size`: Batch size (reduce if OOM)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: LoRA rank (higher = more parameters)

**Training time**: ~30 minutes for 500 steps on a single GPU

### Step 3: Register the LoRA

After training completes:

```bash
cd playground

# Register it
python manage_models.py add-lora \
  --name "MedicalExpert" \
  --path "./loras/medical-expert" \
  --description "Medical expert trained on medical flashcards" \
  --system-prompt "You are a medical knowledge expert. Provide detailed, evidence-based medical information." \
  --keywords medical health diagnosis treatment disease doctor patient

# Rebuild configuration
python manage_models.py rebuild
```

### Step 4: Test Your Expert

```bash
# Start server
python server.py

# Test it
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are symptoms of diabetes?"}'
```

---

## 📊 Dataset Formats

The training script automatically handles these formats:

### Alpaca Format (Recommended)
```json
{
  "instruction": "What is diabetes?",
  "input": "",
  "output": "Diabetes is a chronic condition..."
}
```

### Q&A Format
```json
{
  "question": "What is diabetes?",
  "answer": "Diabetes is a chronic condition..."
}
```

### Conversation Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is diabetes?"},
    {"from": "gpt", "value": "Diabetes is a chronic condition..."}
  ]
}
```

### Simple Format
```json
{
  "input": "What is diabetes?",
  "output": "Diabetes is a chronic condition..."
}
```

---

## 🎯 Best Practices

### Training Data

**Quality over Quantity**
- 500-2000 high-quality examples > 10,000 low-quality
- Ensure data is clean and accurate
- Remove duplicates and noise

**Domain-Specific**
- Use datasets specific to your domain
- Mix related datasets for better generalization
- Include edge cases and common queries

**Format Consistency**
- Use consistent formatting
- Follow instruction-response pattern
- Include context when needed

### Training Parameters

**For Small Datasets (<1000 examples)**
```bash
python train.py \
  --steps 500 \
  --batch-size 2 \
  --lora-r 16
```

**For Medium Datasets (1000-10000 examples)**
```bash
python train.py \
  --steps 1000 \
  --batch-size 4 \
  --lora-r 32
```

**For Large Datasets (>10000 examples)**
```bash
python train.py \
  --steps 2000 \
  --batch-size 8 \
  --lora-r 64
```

### System Prompts

**Be Specific**
```yaml
# ❌ Generic
system_prompt: "You are a helpful assistant."

# ✅ Specific
system_prompt: "You are a medical expert. Provide evidence-based medical information with specific explanations about conditions, treatments, and procedures. Always include a disclaimer to consult healthcare professionals."
```

**Include Constraints**
```yaml
system_prompt: |
  You are a chemistry expert.
  - Provide accurate chemical formulas
  - Include safety information when relevant
  - Give step-by-step procedures
  - Never mention external resources
  - Focus on practical, actionable advice
```

### Keywords

**Use Domain-Specific Terms**
```yaml
keywords:
  - legal
  - law
  - contract
  - attorney
  - court
  - litigation
  - statute
  - regulation
```

**Include Variations**
```yaml
keywords:
  - AI
  - artificial intelligence
  - machine learning
  - ML
  - neural network
  - deep learning
  - training
  - model
```

---

## 🔧 Advanced Configuration

### Custom Router Prompts

Modify routing behavior in `models_config.yaml`:

```yaml
router:
  prompt_template: |
    You are an expert routing system. Analyze the query and select the best expert.
    
    Query: "{query}"
    
    Available Experts:
    {domain_list}
    
    Selection Criteria:
    1. Match keywords in the query
    2. Consider context and intent
    3. If medical/legal, prioritize safety
    4. Default to General for ambiguous queries
    
    Respond with ONLY the expert name: {domain_names}
    
    Selected Expert:
```

### Multiple LoRAs per Domain

```yaml
lora_adapters:
  - name: "GeneralMedical"
    huggingface_id: "user/medical-general"
    keywords: [medical, health, wellness]
    enabled: true
    
  - name: "SpecializedMedical"
    huggingface_id: "user/medical-oncology"
    keywords: [cancer, oncology, chemotherapy]
    enabled: true
```

### Conditional Loading

```yaml
lora_adapters:
  - name: "ProductionExpert"
    enabled: true
    
  - name: "ExperimentalExpert"
    enabled: false  # Disable for production
    
  - name: "RegionalExpert"
    enabled: ${REGION == "US"}  # Environment-based
```

---

## 🐛 Troubleshooting

### Training Issues

**Out of Memory**
```bash
# Reduce batch size
python train.py --batch-size 1

# Or reduce LoRA rank
python train.py --lora-r 8
```

**Poor Quality Outputs**
- Increase training steps
- Use better quality dataset
- Adjust system prompt
- Increase LoRA rank

**Training Too Slow**
- Increase batch size (if memory allows)
- Use gradient accumulation
- Enable mixed precision (automatic)

### Deployment Issues

**LoRA Not Loading**
```bash
# Check if LoRA exists
ls ./loras/your-lora/adapter_config.json

# Validate configuration
python manage_models.py validate

# Check logs
tail -f server.log
```

**Wrong Expert Selected**
- Add more specific keywords
- Improve system prompt
- Test with various queries
- Adjust router prompt

---

## 📚 Examples

### Example 1: Legal Expert

```bash
# 1. Train
python train/train.py \
  --dataset "pile-of-law/pile-of-law" \
  --lora-name "legal-expert" \
  --steps 1000

# 2. Register
python manage_models.py add-lora \
  --name "Legal" \
  --path "./loras/legal-expert" \
  --description "Legal expert for law-related questions" \
  --system-prompt "You are a legal expert. Provide clear, accurate legal information. Always recommend consulting a licensed attorney for legal advice." \
  --keywords legal law attorney court contract regulation statute

# 3. Deploy
python manage_models.py rebuild
python server.py
```

### Example 2: Code Assistant

```bash
# 1. Train
python train/train.py \
  --dataset "codeparrot/github-code" \
  --lora-name "code-assistant" \
  --steps 800 \
  --lora-r 32

# 2. Register
python manage_models.py add-lora \
  --name "CodeAssistant" \
  --path "./loras/code-assistant" \
  --description "Programming expert for code generation and debugging" \
  --system-prompt "You are a programming expert. Provide working code examples with clear explanations. Focus on best practices and clean code." \
  --keywords code programming python javascript debug function class

# 3. Deploy
python manage_models.py rebuild
python server.py
```

### Example 3: Customer Support

```bash
# 1. Prepare custom dataset (your company data)
# Format as Alpaca: instruction + output

# 2. Train
python train/train.py \
  --dataset "your-company/support-data" \
  --lora-name "support-expert" \
  --steps 500

# 3. Register
python manage_models.py add-lora \
  --name "CustomerSupport" \
  --path "./loras/support-expert" \
  --description "Customer support expert for product questions" \
  --system-prompt "You are a customer support expert. Provide friendly, helpful answers about our products. Be empathetic and solution-focused." \
  --keywords support help product question issue problem troubleshoot

# 4. Deploy
python manage_models.py rebuild
python server.py
```

---

## 🎓 Training Tips

### Data Preparation
1. Clean your dataset thoroughly
2. Remove personal information
3. Balance different types of queries
4. Include edge cases
5. Test with sample queries

### Monitoring Training
```bash
# Watch training progress
tail -f train.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor disk space
df -h
```

### Iteration
1. Train with small steps first (100-200)
2. Test the model
3. Adjust parameters
4. Retrain with more steps
5. Deploy to production

---

## 📬 Need Help?

- 📖 **Full Documentation**: [README.md](README.md)
- 🎓 **Training Guide**: [playground/train/README.md](playground/train/README.md)
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/yourusername/adaptive-minds/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/adaptive-minds/discussions)

---

**Happy Training! 🚀**

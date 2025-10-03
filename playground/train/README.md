# 🧠 LoRA Training for Adaptive Minds

Train new domain experts using Unsloth for 2-5x faster training with 70% less memory usage.

## 🚀 Quick Start

### Install Training Dependencies
```bash
# Install all dependencies including training
uv sync --extra training
```

### Simple Training
```bash
# Train a medical expert
uv run python simple_train.py --dataset medalpaca/medical_meadow_medical_flashcards --lora-name medical-expert

# Train a QA expert
uv run python simple_train.py --dataset squad --lora-name qa-expert --steps 300

# Train with custom parameters
uv run python simple_train.py --dataset your-dataset --lora-name custom-expert --steps 1000 --batch-size 4 --lora-r 32
```

## 📋 Parameters

- `--dataset`: Hugging Face dataset name (required)
- `--lora-name`: Name for your LoRA adapter (required)
- `--steps`: Training steps (default: 500)
- `--batch-size`: Batch size (default: 2)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: LoRA rank (default: 16)

## 📊 Dataset Format

The script automatically handles these formats:
- `instruction` + `output`
- `question` + `answer`
- `input` + `output`
- Pre-formatted `text` field

## 🔧 Using Your Trained LoRA

1. **After training completes**, add to `server.py`:
```python
LORA_ADAPTERS = {
    # ... existing adapters ...
    "YourDomain": {
        "path": "./loras/your-lora-name",
        "description": "Your domain expert description",
        "system_prompt": "You are a helpful assistant specializing in..."
    },
}
```

> **Note:** No keywords needed! The AI router uses semantic understanding to select the right expert.

2. **Restart the server**:
```bash
python ../server.py
```

3. **Test your new expert**:
```bash
curl -X POST http://localhost:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Your domain-specific question"}'
```

## 💡 Popular Datasets

- **Medical**: `medalpaca/medical_meadow_medical_flashcards`
- **Legal**: `pile-of-law/pile-of-law`
- **Science**: `sciq`
- **Code**: `codeparrot/github-code`
- **Math**: `microsoft/orca-math-word-problems-200k`
- **General QA**: `squad`, `squad_v2`

## 🎯 Training Tips

- **Start small**: Use 500 steps for testing, increase for production
- **Memory**: Reduce `batch-size` if you get OOM errors
- **Quality**: More training steps ≠ always better, monitor for overfitting
- **LoRA rank**: Higher `lora-r` = more parameters but better adaptation

## 🔍 Directory Structure

```
playground/train/
├── simple_train.py         # Main training script
├── prepare_data.py         # Data preparation utilities
├── train_lora.py          # Advanced training script
├── configs/               # Training configurations
├── data/                  # Local training data
└── datasets/              # Downloaded datasets
```

## ⚡ Requirements

Training dependencies are in optional extras. Install with:
```bash
# Basic server dependencies
uv sync

# Add training dependencies
uv sync --extra training
```

This installs Unsloth, Datasets, TRL, and other training dependencies.
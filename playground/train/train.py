#!/usr/bin/env python3
"""
Simple LoRA Training Script for Adaptive Minds

Train a new domain expert with just a Hugging Face dataset and LoRA name.

Usage:
    python simple_train.py --dataset medalpaca/medical_meadow_medical_flashcards --lora-name medical-v2
    python simple_train.py --dataset squad --lora-name qa-expert --steps 300
"""

import argparse
import logging
import os
import sys
from typing import Optional

# Import unsloth first for optimizations
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_prompt(example):
    """Format training example into Llama chat format with Alpaca format as default."""
    instruction = None
    response = None
    input_text = ""

    # 1. Try Alpaca format first (instruction, input, output)
    if "instruction" in example and "output" in example:
        instruction = example["instruction"]
        response = example["output"]
        input_text = example.get("input", "")  # Optional input field

        # Combine instruction and input if both exist
        if input_text and input_text.strip():
            instruction = f"{instruction}\n\n{input_text}"

    # 2. Try other common formats
    elif "question" in example and "answer" in example:
        instruction = example["question"]
        response = example["answer"]
    elif "prompt" in example and "response" in example:
        instruction = example["prompt"]
        response = example["response"]
    elif "input" in example and "output" in example:
        instruction = example["input"]
        response = example["output"]
    elif "text" in example:
        # For pre-formatted datasets
        return example["text"]
    elif "conversations" in example:
        # Handle conversation format
        convs = example["conversations"]
        if len(convs) >= 2:
            # Find human and assistant messages
            for i in range(len(convs)-1):
                if convs[i].get("from") in ["human", "user"] and convs[i+1].get("from") in ["gpt", "assistant"]:
                    instruction = convs[i]["value"]
                    response = convs[i+1]["value"]
                    break
    else:
        # Try to find any text-like fields
        text_fields = [k for k in example.keys() if isinstance(example[k], str) and len(example[k]) > 10]
        if len(text_fields) >= 2:
            instruction = example[text_fields[0]]
            response = example[text_fields[1]]
        else:
            # Skip examples we can't format
            return None

    if not instruction or not response:
        return None

    # Clean up text
    instruction = str(instruction).strip()
    response = str(response).strip()

    if not instruction or not response:
        return None

    # Format in Llama 3.1 chat template
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

def inspect_dataset(dataset, sample_size=3):
    """Inspect dataset structure to understand the format."""
    logger.info(f"📊 Dataset inspection:")
    logger.info(f"   Features: {list(dataset.features.keys())}")
    logger.info(f"   Size: {len(dataset)}")

    for i in range(min(sample_size, len(dataset))):
        logger.info(f"\n   Example {i+1}:")
        example = dataset[i]
        for key, value in example.items():
            value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            logger.info(f"     {key}: {value_str}")

def train_lora(
    dataset_name: str,
    lora_name: str,
    steps: int = 500,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16
):
    """Train a LoRA adapter."""

    # Paths
    base_model_path = "../base_model/llama-3.1-8B-instruct"
    output_dir = f"../loras/{lora_name}"

    logger.info(f"🚀 Training LoRA: {lora_name}")
    logger.info(f"📊 Dataset: {dataset_name}")
    logger.info(f"💾 Output: {output_dir}")

    # Load model
    logger.info("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except:
        # Try with different splits
        try:
            dataset = load_dataset(dataset_name)["train"]
        except:
            dataset = load_dataset(dataset_name)
            # Take the first available split
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            logger.info(f"Using split: {split_name}")

    logger.info(f"Dataset size: {len(dataset)}")

    # Inspect dataset structure
    inspect_dataset(dataset)

    # Format dataset
    def formatting_func(examples):
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            example = {key: examples[key][i] for key in examples.keys()}
            formatted = format_prompt(example)
            if formatted:
                texts.append(formatted)
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

    # Filter out None values
    dataset = dataset.filter(lambda x: x["text"] is not None)
    logger.info(f"Formatted dataset size: {len(dataset)}")

    if len(dataset) == 0:
        logger.error("No valid examples found in dataset!")
        return

    # Split dataset into train/eval for best checkpoint selection
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        # Checkpoint strategy: keep only best and last checkpoints
        save_strategy="steps",
        save_steps=steps // 4,  # Evaluate and save 4 times during training
        save_total_limit=2,  # Keep only 2 checkpoints (best + last)
        eval_strategy="steps",
        eval_steps=steps // 4,  # Evaluate at same frequency as saving
        load_best_model_at_end=True,  # Load best checkpoint at end
        metric_for_best_model="loss",  # Use loss to determine best model
        greater_is_better=False,  # Lower loss is better
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train
    logger.info("🎯 Starting training...")
    logger.info("💾 Checkpoint strategy: Keeping only best (based on eval loss) and most recent checkpoints")
    trainer.train()

    # Save
    logger.info("💾 Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("✅ Training completed!")
    logger.info(f"📁 LoRA saved to: {output_dir}")
    
    # NEW: Auto-register option
    logger.info("\n💡 Next steps:")
    logger.info("   1. Register your LoRA:")
    logger.info(f'      python manage_models.py add-lora \\')
    logger.info(f'        --name "{lora_name}" \\')
    logger.info(f'        --path "{output_dir.replace("../", "./")}" \\')
    logger.info(f'        --description "Your custom trained expert" \\')
    logger.info(f'        --system-prompt "You are an expert in your domain..." \\')
    logger.info(f'        --keywords domain specific keywords')
    logger.info("   2. Rebuild metadata: python manage_models.py rebuild")
    logger.info("   3. Start server: python server.py")
    
    # Check if manage_models.py exists and offer auto-registration
    if os.path.exists("../manage_models.py"):
        logger.info("\n🤖 Auto-registration available!")
        logger.info("   Run this command to auto-register:")
        logger.info(f'   python ../manage_models.py add-lora --name "{lora_name}" --path "{output_dir.replace("../", "./")}" --description "Custom trained expert" --system-prompt "You are an expert in your domain" --keywords custom domain expert')

def main():
    parser = argparse.ArgumentParser(description="Simple LoRA training for Adaptive Minds")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset name (e.g., squad, medalpaca/medical_meadow)")
    parser.add_argument("--lora-name", type=str, required=True, help="Name for the LoRA (e.g., medical-expert)")
    parser.add_argument("--steps", type=int, default=500, help="Training steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")

    args = parser.parse_args()

    train_lora(
        dataset_name=args.dataset,
        lora_name=args.lora_name,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r
    )

if __name__ == "__main__":
    main()
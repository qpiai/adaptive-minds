"""Shared SFT recipe — the one training script used for every paper
specialist in Table 2.

The recipe (r=32, α=64, 3 epochs, batch 8×2, LR 2e-4 cosine, seed 42) is
identical across the 9 benchmarks. Only the dataset, the base model, and
the output adapter name change. The dataset is expected to be a parquet
or JSONL file with `prompt` and `completion` columns; each row is rendered
as `{prompt} {completion}` and trained on causally.

Example — train the SQL specialist:

    python training/train_sft.py \\
        --dataset hf://pavan01729/spider-sft \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --lora-name qwen25_sql_v1 \\
        --output-dir ./loras/

GRPO and 3-stage curriculum specialists (legal, chemistry, quantum) use
this script for stage 1 (SFT warm-start), then the GRPO step is run
separately — see the per-benchmark sections in `training/README.md`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

RECIPE = dict(
    max_seq_length=1024,
    lora_r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    learning_rate=2e-4,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    seed=42,
)


def format_example(row: dict) -> dict:
    """Render `prompt` + `completion` columns into a single training string."""
    return {"text": f"{row['prompt']} {row['completion']}"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="HF dataset id, parquet path, or jsonl path")
    p.add_argument("--base-model", required=True,
                   help="HF id or local path to the base model")
    p.add_argument("--lora-name", required=True,
                   help="Output adapter directory name (under --output-dir)")
    p.add_argument("--output-dir", type=Path, default=Path("./loras"))
    p.add_argument("--epochs", type=int, default=RECIPE["num_train_epochs"])
    p.add_argument("--max-seq-length", type=int,
                   default=RECIPE["max_seq_length"])
    args = p.parse_args()

    # Heavy imports done lazily so --help works without GPU deps.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = args.output_dir / args.lora_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] base={args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )

    lora_cfg = LoraConfig(
        r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"],
        lora_dropout=RECIPE["lora_dropout"], bias="none",
        target_modules=RECIPE["target_modules"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable={trainable / 1e6:.1f}M")

    # Load dataset — supports hf://, parquet, jsonl.
    src = args.dataset
    if src.startswith("hf://"):
        raw = load_dataset(src[len("hf://"):], split="train")
    elif src.endswith(".parquet"):
        raw = load_dataset("parquet", data_files=src)["train"]
    elif src.endswith(".jsonl") or src.endswith(".json"):
        raw = load_dataset("json", data_files=src)["train"]
    else:
        raw = load_dataset(src, split="train")
    print(f"[data] {len(raw):,} (prompt, completion) pairs")
    ds = raw.map(format_example, remove_columns=raw.column_names)
    print(f"[data] sample text[0]: {ds[0]['text'][:200]!r}")

    cfg = SFTConfig(
        output_dir=str(out_dir),
        learning_rate=RECIPE["learning_rate"],
        warmup_ratio=RECIPE["warmup_ratio"],
        lr_scheduler_type=RECIPE["lr_scheduler_type"],
        weight_decay=RECIPE["weight_decay"],
        per_device_train_batch_size=RECIPE["per_device_train_batch_size"],
        gradient_accumulation_steps=RECIPE["gradient_accumulation_steps"],
        num_train_epochs=args.epochs,
        bf16=True, fp16=False,
        max_length=args.max_seq_length,
        save_strategy="epoch", save_total_limit=2,
        logging_steps=5, report_to="none", seed=RECIPE["seed"],
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds,
        processing_class=tokenizer,
    )
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    trainer.save_model(str(out_dir))
    try:
        tokenizer.save_pretrained(str(out_dir))
    except Exception as e:
        print(f"[warn] tokenizer save: {e}")

    metrics = stats.metrics if hasattr(stats, "metrics") else {}
    log = {
        "adapter": args.lora_name,
        "base_model": args.base_model,
        "data": {"source": args.dataset, "rows": len(raw)},
        "recipe": RECIPE,
        "epochs_actual": args.epochs,
        "final_train_loss": metrics.get("train_loss"),
        "train_runtime_seconds": metrics.get("train_runtime", elapsed),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "trainable_params_million": round(trainable / 1e6, 2),
    }
    (out_dir / "train_log.json").write_text(
        json.dumps(log, indent=2, default=str))
    print(f"[log] {out_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()

"""Download the LoRA subdirs listed in $LORA_SUBDIRS into /loras.

Used by the `prefetch` sidecar in docker-compose.yml so the `vllm`
service can mount /loras read-only and pass local paths to
`--lora-modules`. Idempotent — re-running over an already-populated
cache is a no-op.
"""
import os
import sys

from huggingface_hub import snapshot_download


def main() -> int:
    repo = os.environ.get("LORA_REPO", "pavan01729/adaptive-minds-loras")
    subdirs = [s.strip() for s in os.environ.get("LORA_SUBDIRS", "").split(",") if s.strip()]
    out_dir = os.environ.get("LORA_OUT_DIR", "/loras")

    if not subdirs:
        print("prefetch: $LORA_SUBDIRS is empty; nothing to do", flush=True)
        return 0

    # Only the actual LoRA weights — skip tokenizer files so they don't
    # override the base model's chat template at serving time.
    for sub in subdirs:
        print(f"prefetch: {repo}/{sub} → {out_dir}/{sub}", flush=True)
        snapshot_download(
            repo,
            allow_patterns=[
                f"{sub}/adapter_config.json",
                f"{sub}/adapter_model.safetensors",
            ],
            local_dir=out_dir,
        )

    print(f"prefetch: done ({len(subdirs)} adapter{'s' if len(subdirs) != 1 else ''})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

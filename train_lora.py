#!/usr/bin/env python3
"""
train_lora.py — AgentHub LoRA adapter training launcher

Trains a single LoRA adapter on a cloud GPU using unsloth + Qwen3-8B as base.
Each Wave 1 adapter uses this same script with a different --adapter argument.

Prerequisites (on cloud GPU, e.g. RunPod A100 or Lambda Labs):
  pip install unsloth transformers datasets trl peft bitsandbytes

Usage examples:
  python3 train_lora.py --adapter dr01   # Document Reviewer
  python3 train_lora.py --adapter cr01   # Code Reviewer
  python3 train_lora.py --adapter ar01   # Architecture Reviewer
  python3 train_lora.py --adapter tr01   # Test Reviewer

Output:
  ./output/{adapter_id}/            HuggingFace SafeTensors checkpoint
  ./output/{adapter_id}.gguf        GGUF-converted adapter for llama.cpp
  (deploy to: /home/erol/models/lora/adapters/ on ub02)

Hot-swap after deployment:
  curl -s -X POST http://ub02:8080/lora \
    -H 'Content-Type: application/json' \
    -d '[{"id": 0, "path": "/home/erol/models/lora/adapters/<adapter>.gguf", "scale": 1.0}]'
"""

import argparse, os, json
from pathlib import Path

# ── Adapter registry ──────────────────────────────────────────────────────────
ADAPTERS = {
    "rt01": {
        "name":         "LRA-RT01",
        "description":  "Router Adapter",
        "dataset":      "router_adapter_dataset.jsonl",
        "output_name":  "router_adapter",
        "rank":         8,      # classifier only — no deep reasoning required
        "alpha":        16,
        "epochs":       5,      # small dataset, needs more epochs
        "max_seq_len":  1024,   # task envelopes are short
    },
    "dr01": {
        "name":         "LRA-DR01",
        "description":  "Document Reviewer",
        "dataset":      "doc_reviewer_dataset.jsonl",
        "output_name":  "doc_reviewer",
        "rank":         16,
        "alpha":        32,
        "epochs":       3,
        "max_seq_len":  2048,
    },
    "cr01": {
        "name":         "LRA-CR01",
        "description":  "Code Reviewer",
        "dataset":      "code_reviewer_dataset.jsonl",
        "output_name":  "code_reviewer",
        "rank":         16,
        "alpha":        32,
        "epochs":       3,
        "max_seq_len":  4096,   # code review benefits from longer context
    },
    "ar01": {
        "name":         "LRA-AR01",
        "description":  "Architecture Reviewer",
        "dataset":      "arch_reviewer_dataset.jsonl",
        "output_name":  "arch_reviewer",
        "rank":         32,
        "alpha":        64,
        "epochs":       3,
        "max_seq_len":  2048,
    },
    "tr01": {
        "name":         "LRA-TR01",
        "description":  "Test Reviewer",
        "dataset":      "test_reviewer_dataset.jsonl",
        "output_name":  "test_reviewer",
        "rank":         16,
        "alpha":        32,
        "epochs":       3,
        "max_seq_len":  4096,
    },
    "ta01": {
        "name":         "LRA-TA01",
        "description":  "Telemetry Analyst",
        "dataset":      "telemetry_analyst_dataset.jsonl",
        "output_name":  "telemetry_analyst",
        "rank":         24,
        "alpha":        48,
        "epochs":       3,
        "max_seq_len":  2048,
    },
    "ld01": {
        "name":         "LRA-LD01",
        "description":  "LoRA Designer",
        "dataset":      "lora_designer_dataset.jsonl",
        "output_name":  "lora_designer",
        "rank":         32,
        "alpha":        64,
        "epochs":       3,
        "max_seq_len":  2048,
    },
    "la01": {
        "name":         "LRA-LA01",
        "description":  "LoRA Analyst",
        "dataset":      "lora_analyst_dataset.jsonl",
        "output_name":  "lora_analyst",
        "rank":         24,
        "alpha":        48,
        "epochs":       3,
        "max_seq_len":  2048,
    },
}

BASE_MODEL = "Qwen/Qwen3-8B"

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a single AgentHub LoRA adapter")
    p.add_argument("--adapter",    required=True, choices=list(ADAPTERS),
                   help="Adapter ID (rt01, dr01, cr01, ar01, tr01, ta01, ld01, la01)")
    p.add_argument("--dataset-dir", type=Path,
                   default=Path(__file__).parent,
                   help="Directory containing the *_dataset.jsonl files")
    p.add_argument("--output-dir",  type=Path,
                   default=Path(__file__).parent / "output",
                   help="Directory to write checkpoints and GGUF output")
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch-size",  type=int, default=2)
    p.add_argument("--grad-accum",  type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--dry-run", action="store_true",
                   help="Print config and exit without training")
    return p.parse_args()

# ── Training ──────────────────────────────────────────────────────────────────

def load_dataset_jsonl(path: Path):
    """Load ChatML-format JSONL into a HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def format_chatml(example):
    """Convert messages list to a single ChatML string for tokenization."""
    parts = []
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return {"text": "\n".join(parts)}


def train(cfg: dict, args):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n{'='*60}")
    print(f"Training {cfg['name']} — {cfg['description']}")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Rank       : {cfg['rank']}")
    print(f"  Alpha      : {cfg['alpha']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  Max seq len: {cfg['max_seq_len']}")
    print(f"  Dataset    : {args.dataset_dir / cfg['dataset']}")
    print(f"  Output     : {args.output_dir / cfg['output_name']}")
    print(f"{'='*60}\n")

    dataset_path = args.dataset_dir / cfg["dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Run build_{args.adapter}_dataset.py first."
        )

    output_path = args.output_dir / cfg["output_name"]
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model with unsloth 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=cfg["max_seq_len"],
        dtype=None,             # auto-detect bfloat16 on A100
        load_in_4bit=True,
    )

    # Attach LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Disable Qwen3 thinking mode (--reasoning-budget 0 equivalent at training time)
    if tokenizer.chat_template and "think" in tokenizer.chat_template.lower():
        print("INFO: Qwen3 thinking tokens detected — ensure training data does not "
              "contain <think> blocks. Add /no_think prefix to system prompts if needed.")

    # Load and format dataset
    raw_dataset = load_dataset_jsonl(dataset_path)
    dataset = raw_dataset.map(format_chatml, remove_columns=["messages"])

    # Split train / eval 95/5
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"Train: {len(train_dataset)} pairs | Eval: {len(eval_dataset)} pairs")

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nCheckpoint saved: {output_path}")

    # Convert to GGUF for llama.cpp deployment
    gguf_path = args.output_dir / f"{cfg['output_name']}.gguf"
    print(f"\nConverting to GGUF: {gguf_path}")
    print("Run the following after training completes:")
    print(f"  python3 ~/src/llama.cpp/convert_lora_to_gguf.py \\")
    print(f"    --base {BASE_MODEL} \\")
    print(f"    {output_path} \\")
    print(f"    --outfile {gguf_path}")
    print(f"\nThen deploy to ub02:")
    print(f"  scp {gguf_path} erol@ub02:/home/erol/models/lora/adapters/")
    print(f"\nHot-swap on ub02:")
    print(f"  curl -s -X POST http://ub02:8080/lora \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '[{{\"id\": 0, \"path\": \"/home/erol/models/lora/adapters/"
          f"{cfg['output_name']}.gguf\", \"scale\": 1.0}}]'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = ADAPTERS[args.adapter]

    if args.dry_run:
        print(f"Dry run — would train {cfg['name']} ({cfg['description']})")
        print(f"  Dataset : {args.dataset_dir / cfg['dataset']}")
        print(f"  Output  : {args.output_dir / cfg['output_name']}")
        print(f"  Rank    : {cfg['rank']}, Alpha: {cfg['alpha']}, Epochs: {cfg['epochs']}")
        return

    train(cfg, args)


if __name__ == "__main__":
    main()

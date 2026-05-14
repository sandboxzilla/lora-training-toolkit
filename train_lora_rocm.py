#!/usr/bin/env python3
"""
train_lora_rocm.py — LoRA adapter training for ub02 ROCm (no bitsandbytes)

Strategy: bf16, device_map="auto" splits layers across iGPU (~10GB) and CPU RAM.
Only LoRA adapter parameters are trained — tiny memory footprint.
Gradient checkpointing eliminates activation storage.

Prerequisites (~/lora_train_env):
  torch 2.6.0+rocm6.1, transformers 5.x, peft, trl, datasets, accelerate, numpy

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ~/lora_train_env/bin/python train_lora_rocm.py --adapter rt01

Note: First run downloads Qwen3-8B from HuggingFace (~16GB). Subsequent runs use cache.
"""
import argparse, json
from pathlib import Path

ADAPTERS = {
    "rt01": {"name":"LRA-RT01","description":"Router Adapter","dataset":"router_adapter_dataset.jsonl","output_name":"router_adapter","rank":8,"alpha":16,"epochs":5,"max_seq_len":1024},
    "dr01": {"name":"LRA-DR01","description":"Document Reviewer","dataset":"doc_reviewer_dataset.jsonl","output_name":"doc_reviewer","rank":16,"alpha":32,"epochs":3,"max_seq_len":4096},
    "cr01": {"name":"LRA-CR01","description":"Code Reviewer","dataset":"code_reviewer_dataset.jsonl","output_name":"code_reviewer","rank":16,"alpha":32,"epochs":3,"max_seq_len":4096},
    "ar01": {"name":"LRA-AR01","description":"Architecture Reviewer","dataset":"arch_reviewer_dataset.jsonl","output_name":"arch_reviewer","rank":32,"alpha":64,"epochs":3,"max_seq_len":2048},
    "tr01": {"name":"LRA-TR01","description":"Test Reviewer","dataset":"test_reviewer_dataset.jsonl","output_name":"test_reviewer","rank":16,"alpha":32,"epochs":3,"max_seq_len":4096},
    "r01":  {"name":"LRA-R01", "description":"RAG Synthesis Adapter","dataset":"rag_synthesis_dataset.jsonl","output_name":"rag_synthesis","rank":16,"alpha":32,"epochs":3,"max_seq_len":2048},
}

BASE_MODEL = "Qwen/Qwen3-8B"
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA (ROCm, bf16, CPU offload)")
    p.add_argument("--adapter",      required=True, choices=list(ADAPTERS))
    p.add_argument("--model-path",   type=str,  default=None,
                   help="Local path to base model (overrides BASE_MODEL repo ID)")
    p.add_argument("--dataset-dir",  type=Path, default=Path(__file__).parent)
    p.add_argument("--output-dir",   type=Path, default=Path(__file__).parent / "output")
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=16)
    p.add_argument("--warmup-steps", type=int,   default=20)
    # Max memory per device — tune based on actual free VRAM at start time
    p.add_argument("--gpu-mem",      default="9GiB",
                   help="Max GPU memory to use (default: 9GiB — leaves margin for KV/activations)")
    p.add_argument("--cpu-mem",      default="18GiB",
                   help="Max CPU RAM for model offload (default: 18GiB)")
    p.add_argument("--save-steps",   type=int,   default=10,
                   help="Save a checkpoint every N optimizer steps (default: 10)")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Explicit checkpoint path to resume from. Use 'auto' to pick the latest checkpoint in the output dir.")
    p.add_argument("--force-cpu",    action="store_true",
                   help="Force the CPU-only execution path even if Torch reports a GPU is available.")
    p.add_argument("--max-seq-len",   type=int,   default=None,
                   help="Override the adapter sequence length for this run.")
    p.add_argument("--dry-run",      action="store_true")
    return p.parse_args()

def load_dataset_jsonl(path):
    from datasets import Dataset
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)

def latest_checkpoint(output_path: Path) -> Path | None:
    checkpoints = sorted(
        (p for p in output_path.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None

def format_chatml(example):
    parts = []
    for msg in example["messages"]:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return {"text": "\n".join(parts)}

def train(cfg, args):
    if args.force_cpu:
        import os
        os.environ["ACCELERATE_USE_CPU"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        os.environ["ROCR_VISIBLE_DEVICES"] = ""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    model_id = args.model_path or BASE_MODEL

    gpu_ok = torch.cuda.is_available() and not args.force_cpu
    print(f"\n{'='*60}")
    print(f"Training {cfg['name']} — {cfg['description']}")
    if args.force_cpu:
        print("  Mode       : CPU-only forced by operator")
    elif gpu_ok:
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM total : {vram:.1f} GB  (limit: {args.gpu_mem})")
    else:
        print("  WARNING: No GPU detected — CPU only (very slow)")
    print(f"  Base model : {model_id}")
    print(f"  Rank/Alpha : {cfg['rank']}/{cfg['alpha']}")
    print(f"  Epochs     : {cfg['epochs']}")
    effective_seq_len = args.max_seq_len or cfg["max_seq_len"]
    print(f"  Seq len    : {effective_seq_len}")
    if args.force_cpu:
        print(f"  Mode       : CPU-only (no GPU offload)")
    else:
        print(f"  Mode       : bf16 + device_map=auto (CPU offload for overflow)")
    print(f"  Dataset    : {args.dataset_dir / cfg['dataset']}")
    print(f"  Output     : {args.output_dir / cfg['output_name']}")
    print(f"{'='*60}\n")

    dataset_path = args.dataset_dir / cfg["dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path = args.output_dir / cfg["output_name"]
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # memory budget: fill GPU first, overflow to CPU RAM
    max_memory = {0: args.gpu_mem, "cpu": args.cpu_mem} if gpu_ok else {"cpu": args.cpu_mem}

    print("Loading base model (first run downloads ~16GB from HuggingFace)...")
    model_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map="cpu" if args.force_cpu else "auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template and \
            "think" in str(tokenizer.chat_template).lower():
        print("INFO: Qwen3 thinking tokens present — training data must not contain <think> blocks.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["rank"],
        lora_alpha=cfg["alpha"],
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    raw = load_dataset_jsonl(dataset_path)
    ds  = raw.map(format_chatml, remove_columns=["messages"])
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(split['train'])} | Eval: {len(split['test'])}")

    use_gc = args.force_cpu or effective_seq_len > 2048

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        use_cpu=args.force_cpu,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=gpu_ok or args.force_cpu,
        fp16=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="epoch",
        # load_best_model_at_end=True would try to reload the base model using
        # model.config._name_or_path (the HF repo ID), which fails when running
        # from a local path without HF hub access. LoRA only saves adapters anyway —
        # pick the best checkpoint manually from output_dir/checkpoint-*/eval_loss.
        load_best_model_at_end=False,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        # disable pin_memory — CPU offload + pin_memory can deadlock on ROCm
        dataloader_pin_memory=False,
        # SFTConfig-specific
        dataset_text_field="text",
        max_length=effective_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
    )

    resume = args.resume_from_checkpoint
    if resume == "auto":
        latest = latest_checkpoint(output_path)
        resume = str(latest) if latest else None
        if resume:
            print(f"Resuming from latest checkpoint: {resume}")
        else:
            print("No checkpoint found in output dir; starting fresh")
    elif resume:
        print(f"Resuming from explicit checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"\n✓ Checkpoint: {output_path}")
    gguf = args.output_dir / f"{cfg['output_name']}.gguf"
    print(f"\nConvert to GGUF:")
    print(f"  python3 ~/src/llama.cpp/convert_lora_to_gguf.py \\")
    print(f"    --base {BASE_MODEL} {output_path} --outfile {gguf}")
    print(f"Deploy:")
    print(f"  cp {gguf} ~/models/lora/adapters/")
    print(f"Hot-swap:")
    print(f"  curl -X POST http://localhost:8080/lora -H 'Content-Type: application/json' \\")
    print(f"    -d '[{{\"id\":0,\"path\":\"/home/erol/models/lora/adapters/{cfg['output_name']}.gguf\",\"scale\":1.0}}]'")

def main():
    args = parse_args()
    cfg  = ADAPTERS[args.adapter]
    if args.dry_run:
        model_id = args.model_path or BASE_MODEL
        print(f"Dry run — {cfg['name']} | {args.dataset_dir/cfg['dataset']} | rank {cfg['rank']} | {cfg['epochs']} ep | model: {model_id}")
        return
    train(cfg, args)

if __name__ == "__main__":
    main()

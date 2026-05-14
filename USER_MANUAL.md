# LoRA Training — User Manual

**Script:** `lora_train.py`
**Location:** `backend/scripts/ub02/lora_training/`
**Version:** 1.0 — 2026-05-14

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Contents](#2-directory-contents)
3. [Prerequisites](#3-prerequisites)
4. [Quick Start](#4-quick-start)
5. [Command-Line Reference](#5-command-line-reference)
6. [Config File Reference](#6-config-file-reference)
   - 6.1. [job](#61-job)
   - 6.2. [hardware](#62-hardware)
   - 6.3. [model](#63-model)
   - 6.4. [lora](#64-lora)
   - 6.5. [training](#65-training)
   - 6.6. [data](#66-data)
   - 6.7. [output](#67-output)
7. [Dataset Format](#7-dataset-format)
8. [Artifact Layout](#8-artifact-layout)
9. [Hardware Drivers](#9-hardware-drivers)
   - 9.1. [AMD ROCm (amd_rocm)](#91-amd-rocm-amd_rocm)
   - 9.2. [NVIDIA CUDA (nvidia_cuda)](#92-nvidia-cuda-nvidia_cuda)
   - 9.3. [CPU (cpu)](#93-cpu-cpu)
10. [Checkpoint and Resume](#10-checkpoint-and-resume)
11. [Smoke Testing](#11-smoke-testing)
12. [Monitoring Training Progress](#12-monitoring-training-progress)
13. [Post-Training Steps](#13-post-training-steps)
    - 13.1. [Gate 7 — Inference Validation](#131-gate-7--inference-validation)
    - 13.2. [GGUF Conversion](#132-gguf-conversion)
    - 13.3. [Adapter Deployment](#133-adapter-deployment)
14. [Guardian Integration](#14-guardian-integration)
15. [Exit Codes](#15-exit-codes)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Overview

`lora_train.py` is a config-driven script for fine-tuning causal language models using
[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) via the HuggingFace PEFT
library. All training parameters, hardware settings, paths, and metadata are defined in a
YAML job config file. The script has no hardcoded paths or defaults for required fields.

Key capabilities:

- **Model-agnostic** — works with any HuggingFace causal language model that has a chat
  template and supports PEFT LoRA. Not specific to any single model family. The only
  architecture-specific setting is `lora.target_modules`, which must match the projection
  layer names of the target model (see [Section 6.4](#64-lora)).
- **Hardware abstraction** — AMD ROCm, NVIDIA CUDA, and CPU training via a `driver` field.
  AMD-specific environment variables are set before the PyTorch import, as required by ROCm.
- **Structured artifact tree** — every run gets its own isolated directory (`r001/`, `r002/`,
  …), auto-incremented. All artifacts (log, status JSON, PID file, adapter, checkpoints) use
  the job name as their filename prefix.
- **Checkpoint and resume** — checkpoints save adapter weights, optimizer state, epoch, and
  example index. Resume replays the same RNG shuffles so data order is exactly reproduced.
- **Smoke testing** — validates config, data loading, model loading, and one training step
  without writing any artifacts.
- **Guardian-friendly exit codes** — exit 42 signals GPU irrecoverable state to an external
  guardian process, which can health-check the GPU before restarting.
- **Job scaffolding** — `--build` creates a directory tree and a pre-filled template config
  so new jobs can be started without writing YAML from scratch.

---

## 2. Directory Contents

All files live under `backend/scripts/ub02/lora_training/`.

### 2.1. Training Scripts

| File | Description |
|------|-------------|
| `lora_train.py` | **Main training script (this tool).** Config-driven, hardware-abstracted, checkpoint-resumable. All new training jobs should use this script. |
| `train_lora.py` | Earlier launcher targeting cloud GPU environments with the unsloth library. Not config-driven. Retained for reference. |
| `train_lora_rocm.py` | Earlier AMD ROCm training script using `device_map="auto"` to split layers across iGPU and CPU RAM. Retained for reference. |

### 2.2. Dataset Builders

Each `build_*.py` script produces one or more JSONL dataset files and a companion manifest.
All v2 builders (`*_ar01_*`, `*_cr01_*`, `*_dr01_v2_*`, `*_tr01_*`, `*_rt01_*`) retrieve
governance rules from DocCore at build time so no project-specific constraints are
hardcoded in the training examples.

| File | Adapter | Output |
|------|---------|--------|
| `build_dataset.py` | Original monolithic corpus | `dataset.jsonl` |
| `build_wave1_datasets.py` | LRA-CR01, LRA-DR01 (synthetic) | Generates examples via local llama.cpp server |
| `build_ar01_dataset.py` | LRA-AR01 Architecture Reviewer | `arch_reviewer_dataset.jsonl` |
| `build_cr01_dataset.py` | LRA-CR01 Code Reviewer | `code_reviewer_dataset.jsonl` |
| `build_dr01_dataset.py` | LRA-DR01 Document Reviewer (v1) | `doc_reviewer_dataset.jsonl` |
| `build_dr01_dataset_v2.py` | LRA-DR01 Document Reviewer (v2) | Fully DocCore-driven; no hardcoded rules |
| `build_tr01_dataset.py` | LRA-TR01 Test Reviewer | `test_reviewer_dataset.jsonl` |
| `build_rt01_dataset.py` | LRA-RT01 Router Adapter | `router_adapter_dataset.jsonl` |
| `build_r01_dataset.py` | LRA-R01 RAG Synthesis Adapter | `rag_synthesis_dataset.jsonl` |
| `build_r01_dataset_live.py` | LRA-R01 (live builder) | Reads directly from live `doccore_section_embeddings` rows |

### 2.3. Supporting Scripts

| File | Description |
|------|-------------|
| `retriever_client.py` | Shared DocCore retrieval client imported by all v2 dataset builders. Abstracts the `/retrieve` API call so builders stay DRY. |
| `pipeline.sh` | End-to-end shell pipeline: GGUF conversion → adapter deployment → smoke test → Wave 1 adapter builds (when datasets are present). |
| `monitor_r01.sh` | Polling monitor for the R01 RAG Synthesis adapter training run. Logs progress and alerts on stalls. |
| `r01_monitor_pm2.config.cjs` | PM2 process manager config for running `monitor_r01.sh` as a managed background process. |

### 2.4. Job Configs

| File | Description |
|------|-------------|
| `jobs/wave3_formatter.yaml` | Wave 3 Qwen3-8B document formatter job config. Fully annotated reference config for AMD ROCm training. |

### 2.5. Datasets and Manifests

Pre-built JSONL files and their companion markdown manifests (source stats, sample pairs):

| Dataset | Manifest | Adapter |
|---------|----------|---------|
| `dataset.jsonl` | `manifest.md` | Original monolithic |
| `arch_reviewer_dataset.jsonl` | `arch_reviewer_manifest.md` | LRA-AR01 |
| `code_reviewer_dataset.jsonl` | `code_reviewer_manifest.md` | LRA-CR01 |
| `doc_reviewer_dataset.jsonl` | `doc_reviewer_manifest.md` | LRA-DR01 |
| `router_adapter_dataset.jsonl` | `router_adapter_manifest.md` | LRA-RT01 |
| `test_reviewer_dataset.jsonl` | `test_reviewer_manifest.md` | LRA-TR01 |

### 2.6. Documentation

| File | Description |
|------|-------------|
| `USER_MANUAL.md` | This manual. |
| `adapter_strategy.md` | LoRA adapter ecosystem strategy and roadmap: which adapters are planned, their roles, and the overall training wave plan. |

---

## 3. Prerequisites

### 3.1. Python Environment

The script requires Python 3.9+ and these packages:

```
torch          ≥ 2.0         (ROCm or CUDA build as appropriate)
transformers   ≥ 4.38
peft           ≥ 0.9
pyyaml         ≥ 6.0
```

On **AMD ROCm** (activate your ROCm-compatible virtualenv first):

```bash
source ~/your_rocm_env/bin/activate
# The venv must provide a ROCm-enabled PyTorch build that matches your GPU architecture.
# Example: torch-2.11.0a0 built against ROCm 7.2.2 for your target GPU architecture
```

On a standard **NVIDIA CUDA** machine:

```bash
pip install torch transformers peft pyyaml
```

### 3.2. Dataset

At least one JSONL file in the data directory (see [Section 6](#6-dataset-format)).

### 3.3. Base Model

A HuggingFace model downloaded locally. The script sets `TRANSFORMERS_OFFLINE=1` when
`hardware.amd_rocm.transformers_offline` is true, so no network access is attempted during
training.

---

## 4. Quick Start

**Step 1.** Scaffold a new job:

```bash
python3 lora_train.py --build my_adapter --base-dir ~/lora_training
```

This creates:
- `jobs/my_adapter.yaml` — template config with inline instructions
- `~/lora_training/my_adapter/data/` — place your JSONL files here
- `~/lora_training/my_adapter/runs/` — populated on first run

**Step 2.** Edit the config. Fill in the REQUIRED fields (marked in the template):

```bash
nano jobs/my_adapter.yaml
```

Minimum required fields:

```yaml
hardware:
  driver: amd_rocm                        # or: nvidia_cuda | cpu
  amd_rocm:
    rocblas_tensile_libpath: /opt/rocm-<version>/lib/rocblas/library
    # Find yours: find /opt/rocm* -path "*/rocblas/library" -type d
model:
  path: /path/to/your/model               # local HuggingFace model directory
output:
  base_dir: /path/to/your/output/root     # all job artifacts root under here
```

**Step 3.** Place your dataset files in the data directory:

```bash
cp my_dataset.jsonl {base_dir}/my_adapter/data/
```

**Step 4.** Validate with a smoke test:

```bash
python3 lora_train.py --config jobs/my_adapter.yaml --smoke
```

A successful smoke test loads the tokenizer, loads the model, runs one training step on
32 examples, and exits 0. No files are written.

**Step 5.** Run full training:

```bash
python3 lora_train.py --config jobs/my_adapter.yaml
```

Artifacts appear at `{base_dir}/my_adapter/runs/r001/`.

---

## 5. Command-Line Reference

```
python3 lora_train.py (--config YAML | --build JOB_NAME) [OPTIONS]
```

`--config` and `--build` are mutually exclusive; exactly one is required.

### 5.1. Modes

| Flag | Argument | Description |
|------|----------|-------------|
| `--config` | `YAML` | Path to a job YAML config file. Loads config and runs training (unless `--smoke` or `--resume` also specified). |
| `--build` | `JOB_NAME` | Create a new job scaffold — template config plus directory tree — then exit. Requires `--base-dir`. |

### 5.2. Options

| Flag | Argument | Description |
|------|----------|-------------|
| `--smoke` | — | Smoke-test mode. Runs 1 epoch over 32 examples with output to stdout only. No log file, status JSON, PID file, adapter, or checkpoints are written. Use to validate config and data before committing to a full run. |
| `--resume` | — | Resume from the most recent checkpoint in the latest run directory. Restores adapter weights, optimizer state, epoch number, and example index. If no checkpoint exists, starts a new run. |
| `--base-dir` | `PATH` | Root directory for job artifacts. Required with `--build`. In `--config` mode this is read from `output.base_dir` in the config file. |
| `--jobs-dir` | `PATH` | Directory where `--build` writes the template config. Default: `./jobs/` next to `lora_train.py`. |
| `-h`, `--help`, `-?` | — | Show help and exit. |

### 5.3. Typical Command Patterns

```bash
# Scaffold
python3 lora_train.py --build my_job --base-dir ~/lora_training

# Scaffold with custom jobs directory
python3 lora_train.py --build my_job --base-dir ~/lora_training --jobs-dir ~/configs

# Smoke test
python3 lora_train.py --config jobs/my_job.yaml --smoke

# Full training
python3 lora_train.py --config jobs/my_job.yaml

# Resume after crash or interrupt
python3 lora_train.py --config jobs/my_job.yaml --resume

# Run in background (nohup)
nohup python3 lora_train.py --config jobs/my_job.yaml >> {base_dir}/my_job/runs/r001/my_job.log 2>&1 &
```

---

## 6. Config File Reference

The config is a YAML file. All fields are top-level sections: `job`, `hardware`, `model`,
`lora`, `training`, `data`, `output`.

**Required** fields are marked `[REQUIRED]`. All other fields are optional and have defaults
shown in brackets.

---

### 6.1. `job`

Metadata and agent context for the training job.

```yaml
job:
  name: my_adapter          # [REQUIRED] Job identifier.
                            # Used as: directory name under base_dir,
                            #          prefix for all artifact filenames.
                            # Use only letters, digits, underscores, hyphens.

  description: ""           # Human-readable description.

  labels: []                # List of tags for categorisation.
                            # Example: [lora, qwen3-8b, document-formatter]

  agent_context: |          # Free-form text for AI agents managing this job.
                            # Describe the adapter purpose, dataset source,
                            # required post-training steps, and any constraints.
```

**`job.name`** is the single most important field. It becomes:
- The subdirectory under `output.base_dir`: `~/lora_training/my_adapter/`
- The prefix for every artifact file: `my_adapter.log`, `my_adapter_status.json`, etc.

---

### 6.2. `hardware`

Controls which GPU backend to use and backend-specific environment variables.

```yaml
hardware:
  driver: amd_rocm          # [REQUIRED] One of: amd_rocm | nvidia_cuda | cpu

  lm_head_cpu_bridge: false # [default: false]
                            # true = keep lm_head on CPU as FP32.
                            # Saves ~1.2 GB VRAM at the cost of a
                            # CPU↔GPU transfer per forward pass.
                            # Required on AMD gfx1103 (Radeon 780M)
                            # where FP16 lm_head causes stability issues.
                            # Leave false on NVIDIA or when VRAM allows.

  amd_rocm:                 # Only used when driver: amd_rocm
    no_hipblaslt: true      # [default: true] Sets PYTORCH_NO_HIPBLASLT=1.
                            # Disables hipBLASLt; required on gfx1103 where
                            # hipBLASLt produces silent wrong results.

    rocblas_tensile_libpath: ""   # [REQUIRED for amd_rocm]
                                  # Path to native rocBLAS Tensile kernel library.
                                  # Find yours: find /opt/rocm* -path "*/rocblas/library" -type d
                                  # Typically: /opt/rocm-<version>/lib/rocblas/library

    pytorch_cuda_alloc_conf: expandable_segments:True
                            # Sets PYTORCH_CUDA_ALLOC_CONF.
                            # expandable_segments:True reduces OOM fragmentation.

    serialize_kernel: 3     # Sets AMD_SERIALIZE_KERNEL=3.
                            # Value 3 = fully synchronous HIP kernel dispatch.
                            # MANDATORY on gfx1103: without it, GPU errors surface
                            # inside C++ tensor destructors (noexcept context) and
                            # call std::terminate() rather than raising a Python
                            # RuntimeError. With it, errors are catchable in Python.

    transformers_offline: true
                            # Sets TRANSFORMERS_OFFLINE=1.
                            # Prevents any network access during training.
                            # Required when the training machine has no internet.

  nvidia_cuda:              # Only used when driver: nvidia_cuda
    tf32: true              # [default: true] Enable TF32 on Ampere+ GPUs.
                            # Provides ~3× matmul speedup with minimal accuracy loss.
```

---

### 6.3. `model`

Specifies the base model to fine-tune.

```yaml
model:
  path: ~/qwen3-8b-hf       # [REQUIRED] Local directory containing the HuggingFace
                            # model (config.json, model.safetensors, tokenizer, etc.).
                            # ~ is expanded. The model is loaded offline; no download
                            # is attempted during training.

  attn_implementation: eager  # [default: eager]
                              # Attention implementation to use:
                              #   eager          — standard PyTorch (always works)
                              #   flash_attention_2 — requires flash-attn package;
                              #                       NOT supported on AMD gfx1103
                              #   sdpa           — scaled dot-product attention
                              #                   (torch 2.0+, limited ROCm support)
                              # Use eager on AMD gfx1103.
```

The model is always loaded to CPU first as `float16`, then parameters are moved to GPU
(or kept on CPU for `lm_head` if `lm_head_cpu_bridge` is true). This two-step load avoids
peak RAM spikes from mixed-precision conversion on GPU.

---

### 6.4. `lora`

LoRA adapter configuration. These values define the adapter architecture and are embedded
in `adapter_config.json` inside the saved adapter.

```yaml
lora:
  r: 8                      # [default: 8] LoRA rank.
                            # Higher rank = more trainable parameters = better fit
                            # but more VRAM and risk of overfitting.
                            # Common values: 4, 8, 16, 32.

  alpha: 16                 # [default: 16] LoRA scaling factor.
                            # The effective scale is alpha/r.
                            # Convention: set alpha = 2×r (keeps scale at 2.0).

  target_modules:           # [default: [q_proj, k_proj, v_proj, o_proj]]
    - q_proj                # Which projection layers to adapt.
    - k_proj                # These names are architecture-specific — they must
    - v_proj                # match the actual layer names in the target model.
    - o_proj                #
                            # Llama / Qwen / Mistral families use:
                            #   q_proj, k_proj, v_proj, o_proj (attention)
                            #   gate_proj, up_proj, down_proj  (MLP — optional)
                            # Falcon uses:
                            #   query_key_value, dense, dense_h_to_4h, dense_4h_to_h
                            # GPT-2 / GPT-NeoX use:
                            #   c_attn, c_proj
                            #
                            # To list a model's available modules:
                            #   from transformers import AutoModelForCausalLM
                            #   m = AutoModelForCausalLM.from_pretrained("path")
                            #   print([n for n,_ in m.named_modules()])

  dropout: 0.0              # [default: 0.0] Dropout applied to LoRA weights.
                            # 0.0 is recommended for small datasets;
                            # try 0.05–0.1 for larger datasets.

  bias: none                # [default: none] Whether to train bias terms.
                            # Options: none | lora_only | all
                            # none is standard for most use cases.

  task_type: CAUSAL_LM      # [default: CAUSAL_LM] PEFT task type.
                            # Do not change for autoregressive language models.
```

---

### 6.5. `training`

Controls the training loop.

```yaml
training:
  epochs: 3                 # [default: 3] Number of full passes over the dataset.

  grad_accum: 4             # [default: 4] Gradient accumulation steps.
                            # The batch size per GPU step is always 1 (memory constraint).
                            # Effective batch size = grad_accum × 1 = grad_accum examples.
                            # An optimizer step occurs every grad_accum forward passes.

  max_seq_len: 128          # [default: 128] Maximum token length per example.
                            # Examples longer than this are truncated.
                            # Increase if your data has longer sequences;
                            # VRAM usage grows roughly linearly with this value.

  seed: 42                  # [default: 42] Random seed for dataset shuffling.
                            # The same seed produces the same data order across runs,
                            # which is required for exact checkpoint resume.

  optimizer: adafactor      # [default: adafactor]
                            # adafactor — memory-efficient adaptive optimizer.
                            #   Uses relative_step=True and warmup_init=True.
                            #   Loss oscillates for many steps during warmup — this
                            #   is expected and not a sign of a problem.
                            # adam — not yet implemented.
```

**Total optimizer steps** = `(examples × epochs) ÷ grad_accum`.
Example: 1,286 examples × 3 epochs ÷ 4 = 964 optimizer steps.

---

### 6.6. `data`

Controls where the training data is read from.

```yaml
data:
  dir: /other/path/to/data  # [optional]
                            # Default: {output.base_dir}/{job.name}/data/
                            # Override only when data lives in a different location.

  pattern: "*.jsonl"        # [default: "*.jsonl"]
                            # Glob pattern for dataset files within the data directory.
                            # All matching files are read and concatenated in sorted order.
```

If no files match `{dir}/{pattern}`, the script exits with code 1 and prints an informative
error with the expected path.

---

### 6.7. `output`

Controls where artifacts are written.

```yaml
output:
  base_dir: /path/to/output/root  # [REQUIRED] Root directory for all job artifacts.
                                 # The job subdirectory is: {base_dir}/{job.name}/

  run_id: auto               # [default: auto]
                             # auto — scan {base_dir}/{job.name}/runs/r*/ directories,
                             #        find the highest numeric suffix, increment by 1.
                             #        First run becomes r001; next is r002; etc.
                             # explicit — specify a run ID directly, e.g. r003.
                             #   Warning: specifying an existing run ID will overwrite
                             #   artifacts in that run's directory.

  checkpoint_every_steps: 100  # [default: 100]
                               # Save a checkpoint every N optimizer steps.
                               # Also saves a checkpoint at the end of each epoch
                               # (tagged ep{N}_final) regardless of this setting.

  # Per-artifact path overrides — rarely needed.
  # By default all paths are derived from {base_dir}/{job_name}/runs/{run_id}/.
  # Use these only when you need artifacts on a different filesystem (e.g. fast SSD).

  log: /alt/path/{job_name}.log             # Override log file path
  status: /alt/path/{job_name}_status.json  # Override status JSON path
  pid: /alt/path/{job_name}.pid             # Override PID file path
  adapter_dir: /fast/storage/adapter        # Override final adapter directory
  checkpoint_dir: /fast/storage/checkpoints # Override checkpoints directory
```

**Default artifact paths** (when no overrides are set):

| Artifact | Default path |
|----------|-------------|
| Training log | `{base_dir}/{job}/runs/{run_id}/{job}.log` |
| Status JSON | `{base_dir}/{job}/runs/{run_id}/{job}_status.json` |
| PID file | `{base_dir}/{job}/runs/{run_id}/{job}.pid` |
| Config snapshot | `{base_dir}/{job}/runs/{run_id}/config.yaml` |
| Final adapter | `{base_dir}/{job}/runs/{run_id}/adapter/` |
| Checkpoints | `{base_dir}/{job}/runs/{run_id}/checkpoints/` |

---

## 7. Dataset Format

Dataset files are JSONL (JSON Lines): one JSON object per line, UTF-8 encoded.

The script supports two input formats. Both are automatically detected per example.

### 7.1. Messages Format (Preferred)

Maps directly to the model's chat template. Use this format when your training data is
conversational or when you have multi-turn dialogues.

```jsonl
{"messages": [{"role": "user", "content": "What is a LoRA adapter?"}, {"role": "assistant", "content": "A LoRA adapter is a small set of weight matrices..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Summarise this document."}, {"role": "assistant", "content": "The document describes..."}]}
```

Each object must have a `messages` key whose value is a list of `{role, content}` dicts.
Valid roles: `system`, `user`, `assistant`. The tokenizer's `apply_chat_template` is used
to convert messages to a token sequence.

### 7.2. Alpaca Format

Use for instruction-tuning datasets in the Alpaca style.

```jsonl
{"instruction": "Translate the following English text to French.", "input": "Hello, how are you?", "output": "Bonjour, comment allez-vous ?"}
{"instruction": "Write a Python function to compute factorial.", "input": "", "output": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)"}
```

The `instruction` and `input` fields are concatenated (with a newline) to form the user
turn. `output` becomes the assistant turn. `input` may be empty.

### 7.3. Labels

The script uses next-token prediction with causal masking. The last token in each example
gets label `-100` (ignored by cross-entropy loss). All other tokens predict the next token
in the sequence.

Examples that tokenize to a single token (after truncation) are silently dropped.

---

## 8. Artifact Layout

Every training run is fully isolated in its own directory under `runs/`. Run IDs are
auto-incremented integers padded to three digits.

```
{base_dir}/{job_name}/
│
├── data/                              ← Dataset JSONL files (input; not modified)
│
└── runs/
    ├── r001/                          ← First training run
    │   ├── {job_name}.log             ← Full training transcript (append on resume)
    │   ├── {job_name}_status.json     ← Machine-readable progress (updated each step)
    │   ├── {job_name}.pid             ← PID of the training process
    │   ├── config.yaml                ← Snapshot of config at run start (reproducibility)
    │   ├── adapter/                   ← Final trained adapter
    │   │   ├── adapter_config.json
    │   │   ├── adapter_model.safetensors
    │   │   └── tokenizer files
    │   └── checkpoints/
    │       ├── ep1_step100/           ← Checkpoint at step 100, epoch 1
    │       │   ├── adapter_config.json
    │       │   ├── adapter_model.safetensors
    │       │   ├── optimizer_state.pt ← Required for exact resume
    │       │   └── training_state.json ← epoch, example_idx, total_opt_steps
    │       ├── ep1_step200/
    │       ├── ep1_final/             ← End-of-epoch checkpoint
    │       ├── ep2_step100/
    │       └── ...
    │
    └── r002/                          ← Second run (new run or after --resume fails)
        └── ...
```

### 8.1. `{job_name}_status.json`

Written after every optimizer step. Contains:

```json
{
  "state":      "RUNNING",
  "step":       150,
  "epoch":      2,
  "loss":       1.2345,
  "run_id":     "r001",
  "total_steps": 964
}
```

Used by external monitors and guardian scripts to track progress.

### 8.2. `training_state.json` (inside each checkpoint)

The resume anchor. Contains:

```json
{
  "epoch":           1,
  "example_idx":     399,
  "total_opt_steps": 100,
  "total_skipped":   0,
  "run_id":          "r001",
  "config":          "/path/to/jobs/my_adapter.yaml",
  "tag":             "ep1_step100"
}
```

---

## 9. Hardware Drivers

### 9.1. AMD ROCm (`amd_rocm`)

Targeted at AMD GPUs using ROCm. All environment variables are set **before** the PyTorch
import, which is required by ROCm's HIP runtime — variables set after import have no effect.

**Required environment setup (handled automatically by the script):**

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTORCH_NO_HIPBLASLT` | `1` | Disable hipBLASLt (produces wrong results on some AMD iGPUs) |
| `ROCBLAS_TENSILE_LIBPATH` | library path | Native rocBLAS kernel path for the installed ROCm version |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduce OOM fragmentation |
| `AMD_SERIALIZE_KERNEL` | `3` | Make GPU errors catchable as Python exceptions |
| `TRANSFORMERS_OFFLINE` | `1` | Prevent network access during training |

**`AMD_SERIALIZE_KERNEL=3` is mandatory on integrated AMD GPUs.** Without it, GPU kernel
errors are raised asynchronously inside C++ tensor destructors, which run in a `noexcept`
context. Python cannot catch these — they call `std::terminate()` and kill the process
without a traceable Python exception. With `AMD_SERIALIZE_KERNEL=3`, dispatch is
synchronous, errors surface as `RuntimeError` inside the `try` block in the training loop,
and the guardian can react correctly.

**`lm_head_cpu_bridge: true` is recommended on AMD integrated GPUs.** This keeps the
language model head (`lm_head`) on CPU as FP32 rather than moving it to GPU.

- Saves approximately 1.16 GB of VRAM (vocabulary size × hidden dim × 2 bytes).
- The CPU computation is ~30× faster than FP16 on integrated GPU because the FP32 path is
  heavily optimised on the CPU while FP16 on iGPU is not.
- A small CPU↔GPU transfer happens on each forward pass; this is negligible relative to
  the GPU compute time for the attention layers.

**Note on `HSA_OVERRIDE_GFX_VERSION`:** This variable is **not** used by the training
script. It is only needed when running llama.cpp against an older ROCm stack that does not
natively support the target GPU architecture. If your ROCm version supports your GPU
natively, this variable should not be set during training.

### 9.2. NVIDIA CUDA (`nvidia_cuda`)

Sets TF32 after the PyTorch import for Ampere and later GPUs. No additional configuration
is typically required for standard CUDA training.

```yaml
hardware:
  driver: nvidia_cuda
  nvidia_cuda:
    tf32: true              # Enables torch.backends.cuda.matmul.allow_tf32
                            # and torch.backends.cudnn.allow_tf32
```

Flash Attention 2 is available on NVIDIA GPUs — set `model.attn_implementation: flash_attention_2`
and ensure the `flash-attn` package is installed.

### 9.3. CPU (`cpu`)

No GPU environment setup. Training will be very slow (hours per epoch on small models;
impractical for 7B+ models). Use only for debugging dataset loading and tokenization, or
for tiny models.

```yaml
hardware:
  driver: cpu
```

---

## 10. Checkpoint and Resume

### 10.1. How Checkpoints Work

A checkpoint is saved:
- Every `output.checkpoint_every_steps` optimizer steps (tagged `ep{N}_step{M}`).
- At the end of each epoch (tagged `ep{N}_final`).

Each checkpoint directory contains:
- **`adapter_model.safetensors`** — LoRA weight deltas at that point in training.
- **`optimizer_state.pt`** — Adafactor moment estimates. Required to resume with the same
  optimiser trajectory; without it the optimizer restarts from warmup.
- **`training_state.json`** — Epoch, example index within the epoch, and total optimizer
  step count. This is the resume anchor.

### 10.2. Resuming After a Crash

```bash
python3 lora_train.py --config jobs/my_adapter.yaml --resume
```

The script:
1. Scans `{run_dir}/checkpoints/` in reverse order to find the most recent directory that
   contains `training_state.json`.
2. Loads the LoRA adapter from that checkpoint with `is_trainable=True`.
3. Restores the optimizer state dict.
4. Reads `epoch` and `example_idx` from `training_state.json`.
5. Replays the same RNG shuffles for all already-completed epochs (using `training.seed`)
   so the dataset ordering is reproduced exactly.
6. Skips all examples in the current epoch up to and including `example_idx`.
7. Continues training from the next example.

The log file is opened in append mode so the original transcript is preserved.

### 10.3. What `example_idx` means

`example_idx` is always set at an accumulation boundary — after every `grad_accum`
examples and an `optimizer.step()`. The resume logic skips all examples `i <= example_idx`
in the resume epoch. This means at most `grad_accum - 1` examples of work are lost
(the partial accumulation window before the crash), which is acceptable.

### 10.4. Smoke Testing a Resume

Run with `--smoke` and `--resume` together to verify that a checkpoint can be loaded
without starting a full training run:

```bash
python3 lora_train.py --config jobs/my_adapter.yaml --smoke --resume
```

---

## 11. Smoke Testing

```bash
python3 lora_train.py --config jobs/my_adapter.yaml --smoke
```

Smoke test mode:
- Loads the tokenizer, model, and LoRA adapter exactly as a full run would.
- Reads the dataset and tokenizes it; uses only the first 32 examples.
- Runs one epoch of training on those 32 examples.
- Prints all output to stdout. **No files are written** — no log file, no status JSON,
  no PID file, no adapter, no checkpoints.
- Exits 0 on success, 1 on any error.

Use smoke tests to:
- Verify the config file is syntactically correct and all required fields are present.
- Confirm the data directory contains readable JSONL files in the expected format.
- Confirm the model path is valid and the model loads correctly.
- Catch GPU OOM errors on a short run before committing to hours of training.
- Validate that a new environment (new machine, new ROCm version) can train at all.

---

## 12. Monitoring Training Progress

### 12.1. Log File

The training log is at `{run_dir}/{job_name}.log`. Each optimizer step emits one line:

```
[step  150/964] ep=2 loss=1.2345 VRAM=14588MB skip_ep=0
```

| Field | Description |
|-------|-------------|
| `step N/M` | Current optimizer step N out of estimated total M |
| `ep=N` | Current epoch (1-indexed) |
| `loss=N.NNNN` | Average loss over the last `grad_accum` examples |
| `VRAM=NMB` | GPU memory allocated in MB |
| `skip_ep=N` | Number of examples skipped due to GPU errors in this epoch |

To watch the log live:

```bash
tail -f {run_dir}/{job_name}.log
```

### 12.2. Status JSON

The status file at `{run_dir}/{job_name}_status.json` is a machine-readable snapshot,
updated after every optimizer step:

```bash
cat {run_dir}/{job_name}_status.json
```

### 12.3. Process Liveness

Check that the training process is still running:

```bash
# Using PID file
kill -0 $(cat {run_dir}/{job_name}.pid) && echo ALIVE

# Using process list
pgrep -af lora_train.py
```

### 12.4. GPU Memory (AMD)

```bash
rocm-smi --showmeminfo vram
```

Note: `rocm-smi` cannot report utilisation for AMD integrated GPUs. A 0% utilisation
reading does not mean the GPU is idle — confirm training progress via log freshness and
process liveness instead.

---

## 13. Post-Training Steps

### 13.1. Gate 7 — Inference Validation

After training completes, validate the adapter before converting or deploying it.

Load the final adapter and run a representative inference task:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path   = "/path/to/base-model"        # same model.path from the job config
adapter_path = "{run_dir}/adapter"           # final adapter directory

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
base      = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                  trust_remote_code=True)
model     = PeftModel.from_pretrained(base, adapter_path)
model.eval()

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Your test prompt here"}],
    tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Validate that the output:
- Is grammatically coherent.
- Follows the expected format for your use case.
- Does not regress on baseline capability.

### 13.2. GGUF Conversion

Convert the adapter to GGUF format for use with llama.cpp:

```bash
python3 /path/to/llama.cpp/convert_lora_to_gguf.py \
    --base /path/to/base-model \
    --lora {run_dir}/adapter \
    --outfile /path/to/output/my_adapter.gguf
```

### 13.3. Adapter Deployment

Hot-swap the adapter into a running llama.cpp server:

```bash
# Copy GGUF to the server's adapter directory
cp /path/to/output/my_adapter.gguf /path/to/server/adapters/

# Hot-swap via API (does not require server restart)
curl -X POST http://<server-host>:<port>/lora \
  -H 'Content-Type: application/json' \
  -d '{"lora_adapters": [{"id": 1, "path": "/path/to/server/adapters/my_adapter.gguf", "scale": 1.0}]}'

# Restart inference services if hot-swap is not sufficient
systemctl restart <inference-service-name>
```

---

## 14. Guardian Integration

The training script is designed to work with an external guardian process that monitors
the training process and restarts it on failure.

### 14.1. Exit Code Contract

| Code | Meaning | Guardian Action |
|------|---------|-----------------|
| 0 | Training complete | Guardian exits successfully |
| 1 | Config or setup error | Human intervention required |
| 42 | GPU irrecoverable | Run GPU health check, then restart |

Exit code 42 is emitted when the training loop catches a GPU error **and** the cleanup
sequence (`optimizer.zero_grad()`, `torch.cuda.empty_cache()`) also fails. This double
failure indicates the GPU is in an unrecoverable state. The guardian should:
1. Run a GPU health check (e.g., a small CUDA tensor allocation).
2. Wait for the GPU to recover if needed.
3. Restart the training script with `--resume`.

### 14.2. What the Guardian Should Monitor

A well-designed guardian polls:
- **Process liveness** — `kill -0 $PID` to confirm the training process is running.
- **Log freshness** — `stat -c %Y {log_file}` to detect hangs (no log update for > N minutes).
- **Status JSON** — for human-readable progress display.
- **Exit code** — from the process on termination.

### 14.3. Artifacts the Guardian Needs

The guardian needs four paths, all derivable from the config:
- `output.base_dir` + `job.name` → job directory
- PID file: `{run_dir}/{job_name}.pid`
- Log file: `{run_dir}/{job_name}.log`
- Status JSON: `{run_dir}/{job_name}_status.json`

---

## 15. Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Training completed. Adapter saved to `{run_dir}/adapter/`. The log contains `[DONE]`. |
| 1 | Config error: missing required field, data directory empty, model path invalid, or PyYAML not installed. Error message printed to stderr. |
| 42 | GPU irrecoverable: a GPU error occurred during the forward/backward pass **and** the cleanup operations (`zero_grad`, `empty_cache`) also failed. This signals the guardian to health-check the GPU before attempting a restart. |

---

## 16. Troubleshooting

### "ERROR: config '...' is missing required fields"

One or more required config fields are empty. The error message lists each missing field
and its description. Fill them in the YAML file before retrying.

### "ERROR: no data files found at ..."

The data directory is empty or the `data.pattern` glob matches nothing. Check that:
- JSONL files exist in `{base_dir}/{job_name}/data/` (or your custom `data.dir`).
- The filenames match `data.pattern` (default: `*.jsonl`).

### "ERROR: config already exists: ..."

`--build` refuses to overwrite an existing config. Delete the existing file or choose a
different job name.

### PyYAML not installed

```bash
pip install pyyaml
# or, in a virtualenv:
source ~/your_rocm_env/bin/activate && pip install pyyaml
```

### Loss oscillates for hundreds of steps (Adafactor)

This is expected. Adafactor with `warmup_init=True` and `relative_step=True` has a long
warmup period during which loss is unstable and high. This is not a bug. Loss typically
stabilises after `grad_accum × 100` to `grad_accum × 200` optimizer steps.

### `std::terminate()` called during training (AMD)

`AMD_SERIALIZE_KERNEL=3` is not set. Ensure `hardware.amd_rocm.serialize_kernel: 3` is in
the config. Without it, GPU errors are raised asynchronously in C++ destructors and cannot
be caught by Python.

### `rocm-smi` shows 0% GPU utilisation

This is normal for AMD integrated GPUs. The `rocm-smi` tool cannot
track utilisation for iGPUs. Confirm training is active via log freshness and process
liveness instead.

### OOM (Out of Memory) on AMD

Try these in order:
1. Reduce `training.max_seq_len` (e.g., 128 → 64).
2. Ensure `hardware.lm_head_cpu_bridge: true` is set.
3. Ensure `hardware.amd_rocm.pytorch_cuda_alloc_conf: expandable_segments:True` is set.
4. Reduce `lora.r` (e.g., 8 → 4).

### Resume fails: "No resumable run found"

No checkpoint with a `training_state.json` exists in the latest run. If the run never
reached the first checkpoint interval (`output.checkpoint_every_steps`), there is nothing to
resume from. The script falls back to starting a new run. To avoid this, lower
`checkpoint_every_steps` for short training runs.

### Config snapshot already exists

The script skips writing `config.yaml` to the run directory if it already exists (to protect
the original snapshot on resume). If you intentionally changed the config and want to update
the snapshot, delete `{run_dir}/config.yaml` before resuming.

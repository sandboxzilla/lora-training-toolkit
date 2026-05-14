#!/usr/bin/env python3
"""
lora_train.py — Config-driven LoRA fine-tuning for causal LMs.

Usage:
    # Create a new job scaffold:
    python3 lora_train.py --build my_job_name --base-dir ~/lora_training

    # Run training:
    python3 lora_train.py --config jobs/my_job_name.yaml
    python3 lora_train.py --config jobs/my_job_name.yaml --smoke
    python3 lora_train.py --config jobs/my_job_name.yaml --resume

Artifact layout (all derived from required output.base_dir + job.name):

    {base_dir}/{job_name}/
    ├── data/                              ← dataset JSONL files
    └── runs/
        └── {run_id}/                      ← r001, r002, ... (auto-incremented)
            ├── {job_name}.log
            ├── {job_name}_status.json
            ├── {job_name}.pid
            ├── config.yaml                ← config snapshot (reproducibility)
            ├── adapter/                   ← final saved adapter
            └── checkpoints/
                ├── ep1_step100/
                │   ├── adapter_config.json
                │   ├── adapter_model.safetensors
                │   ├── optimizer_state.pt
                │   └── training_state.json
                └── ...

Hardware drivers:
    amd_rocm    — sets AMD env vars before torch import; optional FP32 CPU bridge
    nvidia_cuda — enables TF32 post-import
    cpu         — no GPU setup

Exit codes:
    0  — completed
    1  — config error or fatal error
    42 — GPU irrecoverable; guardian should health-check GPU before restart
"""
import os, sys, glob, shutil, argparse, textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# ── CLI ───────────────────────────────────────────────────────────────────────
_EPILOG = """\
EXAMPLES
--------
  Scaffold a new job (create directory tree + template config):
    python3 lora_train.py --build my_adapter --base-dir ~/lora_training

  Validate config and data without writing artifacts:
    python3 lora_train.py --config jobs/my_adapter.yaml --smoke

  Full training run:
    python3 lora_train.py --config jobs/my_adapter.yaml

  Resume after crash or keyboard interrupt:
    python3 lora_train.py --config jobs/my_adapter.yaml --resume

  Place config in a custom directory:
    python3 lora_train.py --build my_adapter --base-dir ~/lora_training \\
                          --jobs-dir ~/my_configs

ARTIFACT LAYOUT
---------------
  All paths are derived from output.base_dir and job.name — nothing is
  hardcoded.  Run IDs auto-increment (r001, r002, ...) so each training
  run is fully isolated.

  {base_dir}/{job_name}/
  ├── data/                          ← place JSONL dataset files here
  └── runs/
      └── r001/                      ← new directory per run
          ├── {job_name}.log         ← full training transcript
          ├── {job_name}_status.json ← machine-readable progress
          ├── {job_name}.pid         ← training process ID
          ├── config.yaml            ← config snapshot (reproducibility)
          ├── adapter/               ← final adapter (load with PeftModel)
          └── checkpoints/
              ├── ep1_step100/
              │   ├── adapter_config.json
              │   ├── adapter_model.safetensors
              │   ├── optimizer_state.pt
              │   └── training_state.json
              └── ep1_final/

DATASET FORMAT (JSONL — one example per line)
---------------------------------------------
  Messages format (preferred — maps to chat template directly):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

  Alpaca format (instruction-tuning style):
    {"instruction": "Task description", "input": "Optional context", "output": "Expected answer"}

CONFIG — REQUIRED FIELDS
-------------------------
  job.name                                  Job identifier; becomes directory and file prefix
  hardware.driver                           amd_rocm | nvidia_cuda | cpu
  model.path                                Local HuggingFace model directory (offline)
  output.base_dir                           Root directory for all job artifacts
  hardware.amd_rocm.rocblas_tensile_libpath Required when driver=amd_rocm
                                            find: find /opt/rocm* -path "*/rocblas/library" -type d

CONFIG — KEY OPTIONAL FIELDS
-----------------------------
  hardware.lm_head_cpu_bridge     true = keep lm_head on CPU as FP32; saves ~1.2 GB VRAM
                                  Required on AMD gfx1103 (Radeon 780M); off by default
  hardware.amd_rocm.serialize_kernel  3 = AMD_SERIALIZE_KERNEL=3; mandatory on gfx1103
  model.attn_implementation       eager | flash_attention_2 | sdpa  (default: eager)
                                  eager is required on AMD gfx1103 (no flash-attn support)
  lora.r                          LoRA rank (default: 8)
  lora.alpha                      LoRA scale, usually 2×r (default: 16)
  lora.target_modules             Modules to adapt (default: q_proj k_proj v_proj o_proj)
  lora.dropout                    Dropout on LoRA weights (default: 0.0)
  training.epochs                 Number of training epochs (default: 3)
  training.grad_accum             Gradient accumulation steps; effective batch size (default: 4)
  training.max_seq_len            Sequence truncation length in tokens (default: 128)
  training.seed                   RNG seed for reproducible data order (default: 42)
  output.run_id                   auto (default) or explicit e.g. r003
  output.checkpoint_every_steps   Checkpoint interval in optimizer steps (default: 100)
  data.dir                        Override data directory (default: {base_dir}/{job_name}/data/)
  data.pattern                    Glob for dataset files (default: *.jsonl)
  output.log / output.status      Override individual artifact paths (rarely needed)
  output.adapter_dir              Override final adapter save path
  output.checkpoint_dir           Override checkpoint directory path

EXIT CODES
----------
   0  Training completed successfully — adapter saved to {run_dir}/adapter/
   1  Config error, missing required fields, no data found, or fatal setup error
  42  GPU irrecoverable — exit code signals guardian to health-check GPU before restart

See USER_MANUAL.md in the same directory for full documentation."""

parser = argparse.ArgumentParser(
    description=(
        "Config-driven LoRA fine-tuning for causal language models.\n\n"
        "Two modes (mutually exclusive):\n"
        "  --build JOB_NAME   Scaffold a new job directory tree and template config, then exit\n"
        "  --config YAML      Load a job config and run training (or --smoke / --resume)"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=_EPILOG,
    add_help=False,
)
parser.add_argument(
    "-h", "--help", "-?",
    action="help",
    default=argparse.SUPPRESS,
    help="Show this help message and exit",
)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument(
    "--config", metavar="YAML",
    help="Path to a job YAML config file — loads config and runs training",
)
mode.add_argument(
    "--build", metavar="JOB_NAME",
    help=(
        "Create a new job scaffold: write a template config to --jobs-dir "
        "and create {base_dir}/{job_name}/data/ and runs/ directories, then exit. "
        "Requires --base-dir."
    ),
)
parser.add_argument(
    "--smoke", action="store_true",
    help=(
        "Smoke-test mode: run 1 epoch over the first 32 examples, write all output "
        "to stdout only (no log file, no status JSON, no artifacts). "
        "Use to validate that the config, data, and model load correctly."
    ),
)
parser.add_argument(
    "--resume", action="store_true",
    help=(
        "Resume training from the most recent checkpoint in the latest run directory. "
        "Restores adapter weights, optimizer state, epoch, and example index so that "
        "training continues from exactly where it left off with the same data order."
    ),
)
parser.add_argument(
    "--base-dir", metavar="PATH",
    help=(
        "Root directory for job artifacts. Required with --build. "
        "In --config mode the base_dir is taken from output.base_dir in the config file."
    ),
)
parser.add_argument(
    "--jobs-dir", metavar="PATH", default=None,
    help=(
        "Directory where template config files are written by --build. "
        "Default: ./jobs/ next to lora_train.py. "
        "Use to keep configs outside the script tree."
    ),
)
args = parser.parse_args()

# ── --build: scaffold a new job and exit ─────────────────────────────────────
if args.build:
    job_name = args.build
    if not args.base_dir:
        parser.error("--build requires --base-dir PATH  (root for job artifacts)")

    jobs_dir  = Path(args.jobs_dir).expanduser() if args.jobs_dir else SCRIPT_DIR / "jobs"
    base_dir  = Path(args.base_dir).expanduser()
    job_dir   = base_dir / job_name
    data_dir  = job_dir / "data"
    runs_dir  = job_dir / "runs"
    cfg_path  = jobs_dir / f"{job_name}.yaml"

    if cfg_path.exists():
        print(f"ERROR: config already exists: {cfg_path}", file=sys.stderr)
        print(f"  Delete it or choose a different job name.", file=sys.stderr)
        sys.exit(1)

    jobs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    template = textwrap.dedent(f"""\
        # ── LoRA Training Job Config ──────────────────────────────────────────────────
        # Generated by: python3 lora_train.py --build {job_name} --base-dir {base_dir}
        #
        # Usage:
        #   python3 lora_train.py --config jobs/{job_name}.yaml
        #   python3 lora_train.py --config jobs/{job_name}.yaml --smoke
        #   python3 lora_train.py --config jobs/{job_name}.yaml --resume
        #
        # Artifact tree:
        #   {job_dir}/
        #   ├── data/          ← put your JSONL dataset files here
        #   └── runs/
        #       └── r001/      ← created on first run
        #           ├── {job_name}.log
        #           ├── {job_name}_status.json
        #           ├── {job_name}.pid
        #           ├── config.yaml
        #           ├── adapter/
        #           └── checkpoints/
        # ─────────────────────────────────────────────────────────────────────────────

        job:
          name: {job_name}            # REQUIRED
          description: ""             # human-readable description
          labels: []                  # optional tags
          agent_context: |            # context for AI agents managing this job
            <describe the adapter purpose, dataset, and post-training steps>

        hardware:
          driver: amd_rocm            # REQUIRED — options: amd_rocm | nvidia_cuda | cpu
          lm_head_cpu_bridge: false   # true = keep lm_head on CPU (required on gfx1103)
          amd_rocm:                   # only needed if driver: amd_rocm
            no_hipblaslt: true
            rocblas_tensile_libpath: ""   # REQUIRED for amd_rocm
                                          # find: find /opt/rocm* -path "*/rocblas/library" -type d
            pytorch_cuda_alloc_conf: expandable_segments:True
            serialize_kernel: 3           # 3 = mandatory on gfx1103
            transformers_offline: true
          # nvidia_cuda:              # only needed if driver: nvidia_cuda
          #   tf32: true

        model:
          path: ""                    # REQUIRED — local HuggingFace model directory
          attn_implementation: eager  # eager | flash_attention_2 | sdpa

        lora:
          r: 8                        # LoRA rank
          alpha: 16                   # usually 2×r
          target_modules: [q_proj, k_proj, v_proj, o_proj]
                                      # Architecture-specific — must match your model's layers.
                                      # Llama/Qwen/Mistral: [q_proj, k_proj, v_proj, o_proj]
                                      # Falcon: [query_key_value, dense]
                                      # GPT-2/NeoX: [c_attn, c_proj]
          dropout: 0.0
          bias: none
          task_type: CAUSAL_LM

        training:
          epochs: 3
          grad_accum: 4               # effective batch size (batch is fixed at 1)
          max_seq_len: 128            # truncation length — adjust to your data
          seed: 42
          optimizer: adafactor        # adafactor | adam (adam not yet implemented)

        data:
          # dir defaults to: {data_dir}
          # Override only if data lives elsewhere:
          # dir: /other/path/to/data
          pattern: "*.jsonl"          # glob for dataset files; supports messages + alpaca format

        output:
          base_dir: {base_dir}        # REQUIRED — root for all job artifacts
          run_id: auto                # auto = r001/r002/...; or specify e.g. r003
          checkpoint_every_steps: 100
          # Optional per-artifact overrides (rarely needed):
          # log: /custom/{job_name}.log
          # status: /custom/{job_name}_status.json
          # pid: /custom/{job_name}.pid
          # adapter_dir: /fast/ssd/adapter
          # checkpoint_dir: /fast/ssd/checkpoints
        """)

    cfg_path.write_text(template)

    print(f"Job scaffold created:")
    print(f"  Config:   {cfg_path}")
    print(f"  Data dir: {data_dir}  ← place your JSONL files here")
    print(f"  Runs dir: {runs_dir}  ← populated on first training run")
    print()
    print(f"Required fields to fill in config:")
    print(f"  hardware.driver")
    print(f"  model.path")
    if args.base_dir:
        print(f"  hardware.amd_rocm.rocblas_tensile_libpath  (if using amd_rocm)")
    print()
    print(f"Run smoke test:")
    print(f"  python3 {__file__} --config {cfg_path} --smoke")
    sys.exit(0)

# ── Config load (--config mode) ───────────────────────────────────────────────
try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML required — pip install pyyaml")

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# ── Required field validation ─────────────────────────────────────────────────
def _get(d, dotpath, default=None):
    keys = dotpath.split(".")
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d

REQUIRED_FIELDS = [
    ("job.name",        "job name used as directory prefix and artifact prefix"),
    ("model.path",      "local path to the HuggingFace model directory"),
    ("hardware.driver", "hardware driver — amd_rocm | nvidia_cuda | cpu"),
    ("output.base_dir", "root directory for all job artifacts"),
]

errors = []
for field, desc in REQUIRED_FIELDS:
    val = _get(cfg, field)
    if not val:
        errors.append(f"  {field:30s}  — {desc}")

# Conditional: amd_rocm.rocblas_tensile_libpath required if driver=amd_rocm
if _get(cfg, "hardware.driver") == "amd_rocm":
    val = _get(cfg, "hardware.amd_rocm.rocblas_tensile_libpath")
    if not val:
        errors.append(
            f"  hardware.amd_rocm.rocblas_tensile_libpath  "
            f"— path to native rocBLAS kernels (e.g. /opt/rocm-7.2.2/lib/rocblas/library)"
        )

if errors:
    print(f"ERROR: config '{args.config}' is missing required fields:", file=sys.stderr)
    for e in errors:
        print(e, file=sys.stderr)
    print(f"\nRun 'python3 {__file__} --build JOB_NAME --base-dir PATH' to generate a template.",
          file=sys.stderr)
    sys.exit(1)

def expand(p):
    return str(Path(p).expanduser().resolve()) if p else None

# ── Hardware setup (must run before torch import) ─────────────────────────────
def apply_hardware(cfg):
    drv = _get(cfg, "hardware.driver")
    if drv == "amd_rocm":
        amd = _get(cfg, "hardware.amd_rocm", {})
        if amd.get("no_hipblaslt"):
            os.environ["PYTORCH_NO_HIPBLASLT"] = "1"
        os.environ["ROCBLAS_TENSILE_LIBPATH"] = str(amd["rocblas_tensile_libpath"])
        if amd.get("pytorch_cuda_alloc_conf"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(amd["pytorch_cuda_alloc_conf"])
        if amd.get("serialize_kernel") is not None:
            os.environ["AMD_SERIALIZE_KERNEL"] = str(amd["serialize_kernel"])
        if amd.get("transformers_offline"):
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # nvidia_cuda: TF32 set post-import; cpu: nothing needed

apply_hardware(cfg)

# ── Torch import (after env vars) ─────────────────────────────────────────────
import json, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.optimization import Adafactor

if _get(cfg, "hardware.driver") == "nvidia_cuda":
    nvidia = _get(cfg, "hardware.nvidia_cuda", {})
    if nvidia.get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# ── Resolve artifact paths ────────────────────────────────────────────────────
job_cfg   = cfg.get("job",      {})
out_cfg   = cfg.get("output",   {})
data_cfg  = cfg.get("data",     {})
train_cfg = cfg.get("training", {})
model_cfg = cfg.get("model",    {})
lora_cfg  = cfg.get("lora",     {})
hw        = cfg.get("hardware", {})

JOB_NAME  = job_cfg["name"]
BASE_DIR  = expand(out_cfg["base_dir"])
JOB_DIR   = os.path.join(BASE_DIR, JOB_NAME)
RUNS_DIR  = os.path.join(JOB_DIR, "runs")

def _next_run_id():
    existing = sorted(
        d for d in glob.glob(os.path.join(RUNS_DIR, "r*"))
        if os.path.isdir(d) and os.path.basename(d)[1:].isdigit()
    )
    if not existing:
        return "r001"
    last_num = max(int(os.path.basename(d)[1:]) for d in existing)
    return f"r{last_num + 1:03d}"

run_id_cfg = out_cfg.get("run_id", "auto")

if args.resume:
    existing_runs = sorted(
        d for d in glob.glob(os.path.join(RUNS_DIR, "r*"))
        if os.path.isdir(d) and os.path.basename(d)[1:].isdigit()
    )
    RUN_ID = None
    for rd in reversed(existing_runs):
        if glob.glob(os.path.join(rd, "checkpoints", "*", "training_state.json")):
            RUN_ID = os.path.basename(rd)
            break
    if RUN_ID is None:
        print("[RESUME] No resumable run found — starting a new run")
        RUN_ID = _next_run_id() if run_id_cfg == "auto" else run_id_cfg
        args.resume = False
else:
    RUN_ID = _next_run_id() if run_id_cfg == "auto" else run_id_cfg

RUN_DIR     = os.path.join(RUNS_DIR, RUN_ID)
CKPT_DIR    = expand(out_cfg.get("checkpoint_dir")) or os.path.join(RUN_DIR, "checkpoints")
ADAPTER_DIR = expand(out_cfg.get("adapter_dir"))    or os.path.join(RUN_DIR, "adapter")
LOG_PATH    = expand(out_cfg.get("log"))             or os.path.join(RUN_DIR, f"{JOB_NAME}.log")
STATUS_FILE = expand(out_cfg.get("status"))          or os.path.join(RUN_DIR, f"{JOB_NAME}_status.json")
PID_FILE    = expand(out_cfg.get("pid"))             or os.path.join(RUN_DIR, f"{JOB_NAME}.pid")
DATA_DIR    = expand(data_cfg.get("dir"))            or os.path.join(JOB_DIR, "data")
MODEL_PATH  = expand(model_cfg["path"])

# Training hyperparams
MAX_SEQ_LEN    = int(train_cfg.get("max_seq_len", 128))
GRAD_ACCUM     = int(train_cfg.get("grad_accum", 4))
EPOCHS         = int(train_cfg.get("epochs", 3))
SEED           = int(train_cfg.get("seed", 42))
CKPT_STEPS     = int(out_cfg.get("checkpoint_every_steps", 100))
LM_HEAD_BRIDGE = bool(hw.get("lm_head_cpu_bridge", False))
ATTN_IMPL      = str(model_cfg.get("attn_implementation", "eager"))
LORA_R         = int(lora_cfg.get("r", 8))
LORA_ALPHA     = int(lora_cfg.get("alpha", 16))
LORA_DROPOUT   = float(lora_cfg.get("dropout", 0.0))
LORA_BIAS      = str(lora_cfg.get("bias", "none"))
TARGET_MODULES = list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))

SMOKE = args.smoke

# ── Create run directory and snapshot config ──────────────────────────────────
if not SMOKE:
    os.makedirs(RUN_DIR,     exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,    exist_ok=True)
    config_snapshot = os.path.join(RUN_DIR, "config.yaml")
    if not os.path.exists(config_snapshot):
        shutil.copy2(args.config, config_snapshot)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

log_mode = "a" if args.resume else "w"
log_f = open(LOG_PATH, log_mode, buffering=1) if not SMOKE else sys.stdout

def log(msg):
    print(msg, flush=True)
    if not SMOKE:
        print(msg, file=log_f, flush=True)

log(f"[INIT] lora_train.py  config={args.config}  smoke={SMOKE}  resume={args.resume}")
log(f"[INIT] job={JOB_NAME}  run={RUN_ID}")
log(f"[INIT] run_dir={RUN_DIR}")
log(f"[INIT] torch={torch.__version__}  driver={hw.get('driver')}  lm_head_bridge={LM_HEAD_BRIDGE}")
log(f"[INIT] data={DATA_DIR}  adapter={ADAPTER_DIR}")

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _state_path(d): return os.path.join(d, "training_state.json")
def _optim_path(d): return os.path.join(d, "optimizer_state.pt")

def find_latest_resumable_checkpoint():
    dirs = sorted(d for d in glob.glob(os.path.join(CKPT_DIR, "*")) if os.path.isdir(d))
    for d in reversed(dirs):
        if os.path.exists(_state_path(d)):
            return d
    return None

def save_checkpoint(tag, optimizer, epoch, example_idx, total_opt_steps, total_skipped):
    if SMOKE:
        return
    path = os.path.join(CKPT_DIR, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), _optim_path(path))
    state = {
        "epoch":           epoch,
        "example_idx":     example_idx,
        "total_opt_steps": total_opt_steps,
        "total_skipped":   total_skipped,
        "run_id":          RUN_ID,
        "config":          args.config,
        "tag":             tag,
    }
    with open(_state_path(path), "w") as f:
        json.dump(state, f, indent=2)
    log(f"[CKPT] Saved: {path}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
log(f"[1] Tokenizer OK  vocab={len(tokenizer)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
pattern = data_cfg.get("pattern", "*.jsonl")
all_examples = []
for fpath in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

if not all_examples:
    print(f"ERROR: no data files found at {DATA_DIR}/{pattern}", file=sys.stderr)
    print(f"  Place JSONL dataset files there, then retry.", file=sys.stderr)
    sys.exit(1)

log(f"[2] Loaded {len(all_examples)} examples from {DATA_DIR}/{pattern}")

def tokenize_ex(ex):
    if "messages" in ex:
        msgs = ex["messages"]
    else:
        msgs = [
            {"role": "user",      "content": ex.get("instruction", "") + "\n" + ex.get("input", "")},
            {"role": "assistant", "content": ex.get("output", "")},
        ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    toks = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN,
                     return_tensors="pt", padding=False)
    iids   = toks["input_ids"][0]
    labels = iids.clone()
    labels[:-1] = iids[1:]
    labels[-1]  = -100
    return iids, labels

tokenized = [
    (iids, lbls)
    for ex in all_examples
    for iids, lbls in [tokenize_ex(ex)]
    if len(iids) > 1
]
if SMOKE:
    tokenized = tokenized[:32]
log(f"[2] Tokenized {len(tokenized)} examples")

# ── Resume detection ──────────────────────────────────────────────────────────
resume_ckpt  = None
resume_state = None

if args.resume:
    resume_ckpt = find_latest_resumable_checkpoint()
    if resume_ckpt:
        with open(_state_path(resume_ckpt)) as f:
            resume_state = json.load(f)
        log(f"[RESUME] Checkpoint: {resume_ckpt}")
        log(f"[RESUME] epoch={resume_state['epoch']}  example_idx={resume_state['example_idx']}  "
            f"total_opt_steps={resume_state['total_opt_steps']}")
    else:
        log("[RESUME] No checkpoint in this run — starting fresh")

# ── Base model load ───────────────────────────────────────────────────────────
log("[3] Loading base model to CPU (float16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation=ATTN_IMPL,
)

log("[3] Moving params to GPU...")
for name, param in base_model.named_parameters():
    if not LM_HEAD_BRIDGE or "lm_head" not in name:
        param.data = param.data.to("cuda")
for name, buf in base_model.named_buffers():
    if (not LM_HEAD_BRIDGE or "lm_head" not in name) and buf.device.type == "cpu":
        buf.data = buf.data.to("cuda")
torch.cuda.synchronize()
log(f"[3] VRAM={torch.cuda.memory_allocated()//1024//1024}MB")

if LM_HEAD_BRIDGE:
    _w = base_model.lm_head.weight.detach().float()
    _b = base_model.lm_head.bias.detach().float() if base_model.lm_head.bias is not None else None
    def _lm_head_bridge(x):
        return F.linear(x.to("cpu").float(), _w, _b).to(x.device, dtype=x.dtype)
    base_model.lm_head.forward = _lm_head_bridge
    log("[3] lm_head FP32 CPU-bridge installed")

# ── LoRA adapter ──────────────────────────────────────────────────────────────
if resume_ckpt:
    log(f"[4] Loading LoRA adapter from checkpoint for continued training...")
    model = PeftModel.from_pretrained(base_model, resume_ckpt, is_trainable=True)
else:
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

model.print_trainable_parameters()
log(f"[4] LoRA r={LORA_R} alpha={LORA_ALPHA} on {TARGET_MODULES}")

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=True, relative_step=True, warmup_init=True,
)
if resume_ckpt and os.path.exists(_optim_path(resume_ckpt)):
    log("[5] Restoring optimizer state...")
    optimizer.load_state_dict(torch.load(_optim_path(resume_ckpt)))

epochs_to_run  = 1 if SMOKE else EPOCHS
expected_steps = epochs_to_run * len(tokenized) // GRAD_ACCUM
log(f"[5] Adafactor | {expected_steps} expected steps | {epochs_to_run} epoch(s)")

# ── Training state ────────────────────────────────────────────────────────────
total_opt_steps    = resume_state["total_opt_steps"] if resume_state else 0
total_skipped      = resume_state["total_skipped"]   if resume_state else 0
start_epoch        = resume_state["epoch"]            if resume_state else 0
resume_example_idx = resume_state["example_idx"]      if resume_state else -1

# ── Training loop ─────────────────────────────────────────────────────────────
log("[6] Training START")
random.seed(SEED)

# Advance RNG past already-completed epoch shuffles so data order is reproduced exactly.
for _ in range(start_epoch):
    idxs_dummy = list(range(len(tokenized)))
    random.shuffle(idxs_dummy)

for epoch in range(start_epoch, epochs_to_run):
    idxs = list(range(len(tokenized)))
    random.shuffle(idxs)

    accum_loss        = 0.0
    accum_n           = 0
    epoch_skip        = 0
    resuming_in_epoch = (epoch == start_epoch and resume_example_idx >= 0)

    for i, idx in enumerate(idxs):
        # Skip examples already processed when resuming mid-epoch.
        # resume_example_idx is always at an accumulation boundary (set after optimizer.step).
        if resuming_in_epoch and i <= resume_example_idx:
            continue
        resuming_in_epoch = False

        iids, lbls = tokenized[idx]
        iids = iids.unsqueeze(0).to("cuda")
        lbls = lbls.unsqueeze(0).to("cuda")

        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out  = model(input_ids=iids, labels=lbls)
                loss = out.loss / GRAD_ACCUM
            loss.backward()
            # synchronize() inside try: surfaces deferred AMD HIP errors as Python
            # RuntimeError here rather than in a C++ tensor destructor (std::terminate).
            torch.cuda.synchronize()
            accum_loss += loss.detach().cpu().item() * GRAD_ACCUM
            accum_n    += 1

        except Exception as e:
            log(f"[SKIP] ep={epoch+1} i={i} idx={idx} err={str(e)[:120]}")
            try:
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            except Exception as cleanup_err:
                log(f"[GPU_POISON] Cleanup failed: {str(cleanup_err)[:100]}")
                log("[GPU_POISON] GPU irrecoverable — exiting 42 for guardian restart")
                if not SMOKE:
                    log_f.flush()
                sys.exit(42)
            accum_loss = 0.0
            accum_n    = 0
            epoch_skip    += 1
            total_skipped += 1
            continue

        is_accum_boundary = ((i + 1) % GRAD_ACCUM == 0) or ((i + 1) == len(idxs))
        if is_accum_boundary:
            optimizer.step()
            optimizer.zero_grad()
            total_opt_steps += 1
            vram     = torch.cuda.memory_allocated() // 1024 // 1024
            avg_loss = accum_loss / accum_n if accum_n > 0 else 0.0
            log(f"[step {total_opt_steps:4d}/{expected_steps}] ep={epoch+1} "
                f"loss={avg_loss:.4f} VRAM={vram}MB skip_ep={epoch_skip}")
            accum_loss = 0.0
            accum_n    = 0

            if total_opt_steps % CKPT_STEPS == 0:
                save_checkpoint(
                    f"ep{epoch+1}_step{total_opt_steps}",
                    optimizer, epoch, i, total_opt_steps, total_skipped,
                )

    log(f"[EPOCH {epoch+1}/{epochs_to_run} DONE] steps={total_opt_steps} skipped={epoch_skip}")
    save_checkpoint(
        f"ep{epoch+1}_final",
        optimizer, epoch + 1, -1, total_opt_steps, total_skipped,
    )

log(f"[DONE] Total steps={total_opt_steps} total_skipped={total_skipped}")
log(f"[SAVE] → {ADAPTER_DIR}")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
log("[SAVE] Done.")
if not SMOKE:
    log_f.close()

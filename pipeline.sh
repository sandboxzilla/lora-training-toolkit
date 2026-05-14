#!/bin/bash
# LoRA training pipeline — runs entirely on ub02
# Handles: wait for RT01 → GGUF convert → deploy → test → Wave1 adapters (if datasets present)
set -uo pipefail

BASE_MODEL=~/.cache/huggingface/hub/qwen3-8b
LORA_DIR=~/lora_training
OUTPUT_DIR=$LORA_DIR/output
ADAPTER_DIR=~/models/lora/adapters
LLAMA_SERVER=http://localhost:8080
LOG=$LORA_DIR/pipeline.log
TRAIN_PID_FILE=$LORA_DIR/current_train.pid

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

mkdir -p "$ADAPTER_DIR"
log "Pipeline started"

convert_and_deploy() {
    local adapter_key="$1"   # e.g. rt01
    local output_name="$2"   # e.g. router_adapter
    local adapter_dir="$OUTPUT_DIR/$output_name"

    # If trainer.save_model() didn't write adapter_config.json, use last checkpoint
    if [ ! -f "$adapter_dir/adapter_config.json" ]; then
        log "adapter_config.json missing in $adapter_dir — trying last checkpoint"
        last_ckpt=$(ls -d "$adapter_dir"/checkpoint-* 2>/dev/null | sort -V | tail -1)
        if [ -n "$last_ckpt" ]; then
            log "Copying checkpoint: $last_ckpt → $adapter_dir"
            cp -r "$last_ckpt"/. "$adapter_dir/"
        else
            log "ERROR: No checkpoint for $output_name — skipping GGUF"
            return 1
        fi
    fi

    log "Converting $output_name to GGUF..."
    cd ~/src/llama.cpp
    ~/lora_train_env/bin/python convert_lora_to_gguf.py \
        --base "$BASE_MODEL" \
        "$adapter_dir" \
        --outfile "$OUTPUT_DIR/${output_name}.gguf" 2>&1 | tee -a "$LOG"

    if [ ! -f "$OUTPUT_DIR/${output_name}.gguf" ]; then
        log "ERROR: GGUF conversion failed for $output_name"
        return 1
    fi

    cp "$OUTPUT_DIR/${output_name}.gguf" "$ADAPTER_DIR/"
    log "Deployed: $ADAPTER_DIR/${output_name}.gguf"
}

test_adapter() {
    local adapter_key="$1"
    local gguf_path="$ADAPTER_DIR/$2.gguf"

    log "Hot-swapping $adapter_key..."
    swap_result=$(curl -sf -X POST "$LLAMA_SERVER/lora-adapters" \
        -H 'Content-Type: application/json' \
        -d "[{\"id\":0,\"path\":\"$gguf_path\",\"scale\":1.0}]" 2>&1 || echo "SWAP_FAILED")
    log "Swap result: $swap_result"

    # Per-adapter test prompt
    case "$adapter_key" in
        rt01) sys="You are a router. Respond JSON: {\"agent\":\"<name>\",\"reason\":\"<one sentence>\"}. Agents: code_reviewer,doc_reviewer,arch_reviewer,test_reviewer,rag_synthesis."
              usr="Review this Python function for off-by-one errors." ;;
        dr01) sys="You are a document reviewer. Identify issues in the provided document."
              usr="Review this API specification for completeness." ;;
        cr01) sys="You are a code reviewer. Identify bugs, security issues, and style violations."
              usr="Review this SQL query: SELECT * FROM users WHERE id='$user_id'" ;;
        ar01) sys="You are an architecture reviewer. Evaluate the design for scalability and correctness."
              usr="Review this microservice architecture: 3 services sharing a single database." ;;
        tr01) sys="You are a test reviewer. Evaluate test quality and coverage."
              usr="Review this test: assert add(1,1) == 2" ;;
        *) log "No test defined for $adapter_key"; return ;;
    esac

    response=$(curl -sf -X POST "$LLAMA_SERVER/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"qwen3\",\"messages\":[{\"role\":\"system\",\"content\":$(echo "$sys" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")},{\"role\":\"user\",\"content\":$(echo "$usr" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")}],\"max_tokens\":256,\"temperature\":0}" 2>&1 || echo "QUERY_FAILED")
    log "[$adapter_key test] $response"
}

train_adapter() {
    local adapter_key="$1"
    local dataset="$2"
    local output_name="$3"

    if [ ! -f "$LORA_DIR/$dataset" ]; then
        log "SKIP $adapter_key — dataset $dataset not found"
        return 0
    fi

    log "Training $adapter_key (dataset: $dataset)..."
    HSA_OVERRIDE_GFX_VERSION=11.0.0 AMD_SERIALIZE_KERNEL=3 HSA_ENABLE_SDMA=0 \
    ~/lora_train_env/bin/python ~/lora_training/train_lora_rocm.py \
        --adapter "$adapter_key" \
        --model-path "$BASE_MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --gpu-mem 9GiB --cpu-mem 16GiB \
        >> "$LORA_DIR/train_${adapter_key}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "WARNING: $adapter_key training exited with code $rc — checking for checkpoints"
    fi
    convert_and_deploy "$adapter_key" "$output_name"
    test_adapter "$adapter_key" "$output_name"
}

# --- Wait for RT01 (PID from file, or passed as arg) ---
RT01_PID="${1:-}"
if [ -z "$RT01_PID" ] && [ -f "$TRAIN_PID_FILE" ]; then
    RT01_PID=$(cat "$TRAIN_PID_FILE")
fi

if [ -n "$RT01_PID" ]; then
    log "Waiting for RT01 (PID $RT01_PID)..."
    while kill -0 "$RT01_PID" 2>/dev/null; do
        sleep 60
    done
    log "RT01 process ended"
fi

# --- RT01 post-processing ---
convert_and_deploy "rt01" "router_adapter"
test_adapter "rt01" "router_adapter"

# --- Wave 1 adapters (train only if dataset present) ---
train_adapter "dr01" "doc_reviewer_dataset.jsonl"   "doc_reviewer"
train_adapter "cr01" "code_reviewer_dataset.jsonl"  "code_reviewer"
train_adapter "ar01" "arch_reviewer_dataset.jsonl"  "arch_reviewer"
train_adapter "tr01" "test_reviewer_dataset.jsonl"  "test_reviewer"

log "=== PIPELINE COMPLETE ==="
log "Adapters in $ADAPTER_DIR:"
ls -lh "$ADAPTER_DIR/" 2>/dev/null | tee -a "$LOG"

# Clear the active LoRA (reset to base model)
curl -sf -X POST "$LLAMA_SERVER/lora-adapters" -H 'Content-Type: application/json' -d '[]' >> "$LOG" 2>&1 || true
log "Base model restored (LoRA cleared)"

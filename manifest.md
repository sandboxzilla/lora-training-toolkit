# AgentHub LoRA Training Dataset — Manifest

**Generated**: 2026-04-12 15TC
**Output**: `dataset.jsonl`
**Format**: ChatML (Qwen3 instruction-tuning)

---

## 1. Summary Statistics

| Metric | Value |
| :----- | ----: |
| Total training pairs | 4,276 |
| Estimated tokens | ~1,341,958 |
| Deduplicated pairs removed | 4,127 |

---

## 2. Source Breakdown

| Source | Pairs |
| :----- | ----: |
| Domain knowledge (hardcoded) | 11 |
| CONTEXT.md | 49 |
| CLAUDE.md | 10 |
| backend/scripts/ub02/retriever-service.js | 2 |
| SQL schema files (12 files) | 26 |
| Markdown docs (312 files) | 8,305 |

---

## 3. Training Pair Format

Each line in `dataset.jsonl` is a JSON object:

```json
{
  "messages": [
    {"role": "system",    "content": "<condensed system prompt>"},
    {"role": "user",      "content": "<question about AgentHub>"},
    {"role": "assistant", "content": "<answer>"}
  ]
}
```

This is the ChatML format used by Qwen3. Compatible with:
- `llama.cpp` LoRA training (via `finetune` binary or llama-finetune)
- `unsloth` (FastLanguageModel.from_pretrained + LoraConfig)
- `axolotl` (sharegpt format with minor renaming)
- `LLaMA-Factory` (sharegpt format)

---

## 4. Sample Pairs (first 5)

### Sample 1

**User**: What is AgentHub?

**Assistant**: AgentHub is a multi-tenant AI agent orchestration platform. It coordinates AI agents (Worker Bees) via MQTT messaging, stores state in MariaDB on ub01, serves a REST API on port 3000, and provides a Flutter Web dashboard. Agents run in Podman containers on cluster nodes. The project follows the AHC ...

---

### Sample 2

**User**: What is the AgentHub cluster topology?

**Assistant**: The cluster consists of:
- ub01 (192.168.11.111): Control plane — MariaDB ah_db01, MQTT :1883/:9001, Redis, API :3000
- ub02 (192.168.11.65): AI inference — Qwen3-9B :8080, Qwen3-4B embeddings :8081, retriever-service :8082
- n20, n40, n50, n60: Fleet nodes on 192.168.11.x (wlan) / 192.168.1.x (eth)...

---

### Sample 3

**User**: What is the AHC coordinate system?

**Assistant**: AHC (AgentHub Coordinate) is the document governance standard. Every document has a coordinate like sbz_org01.sbz_p01.ah_g01._ah_opg01.ah_opj01.DOC-001. Coordinates encode org → program → group → sub-group → project → document ID. The File Path Anchor (FPA) declares whether a file is an Internal Log...

---

### Sample 4

**User**: What MQTT topics does AgentHub use?

**Assistant**: Key MQTT topic patterns:
- `starbush/fleet/{node}/heartbeat` — node heartbeat
- `starbush/fleet/{node}/telemetry` — CPU/mem/load telemetry
- `system/hives/{node}/heartbeat` — hive heartbeat (indexed by telemetry_bridge)
- `agenthub/fleet/cmd/{node}` — inbound commands to fleet nodes
- `agenthub/agen...

---

### Sample 5

**User**: What is the retriever-service and what can it do?

**Assistant**: The retriever-service is a Node.js agentic loop running on ub02:8082. It uses Qwen3-9B via llama.cpp on port 8080 with an OpenAI-compatible API. It provides 16 tools:
shell_exec, ssh_exec, file_read, file_write — system operations
web_search, web_fetch — internet access
http_request — arbitrary HTTP...

---

## 5. Training Recommendations

| Parameter | Recommended Value | Notes |
| :-------- | :---------------- | :---- |
| LoRA rank | 16 | Sufficient for domain Q&A adaptation |
| LoRA alpha | 32 | 2× rank |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | All attention + FFN |
| Learning rate | 2e-4 | Standard for LoRA |
| Batch size | 2 | CPU memory constraint |
| Gradient accumulation | 8 | Effective batch = 16 |
| Epochs | 2-3 | Small domain dataset |
| Max seq length | 2048 | Keep context window manageable |
| Warmup steps | 50 | ~5% of total steps |
| Optimizer | adamw_torch | Or paged_adamw_8bit if GPU available |
| Framework | unsloth (GPU) or llama.cpp finetune (CPU, slow) | |

**GPU estimate (A100 40GB)**: ~2-4 hours for 2 epochs
**CPU estimate (ub02, 14 threads)**: ~10-20 days per epoch — not recommended

## 6. Decision Guidance

Fine-tune **if**:
- Qwen consistently ignores AHC naming conventions despite RAG
- Qwen mis-identifies cluster topology (wrong IPs, wrong ports)
- Behavioral patterns (tool preference, response style) need baking in

Stay with RAG + system prompt **if**:
- Current responses are accurate with RAG context
- Domain knowledge is evolving rapidly (LoRA would go stale)
- No GPU budget available

**Current state**: retriever-service uses `rag_query` + `DEFAULT_SYSTEM_PROMPT`.
Test those first before committing to LoRA.

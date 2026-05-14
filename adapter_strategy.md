# AgentHub LoRA Adapter Ecosystem — Strategy & Roadmap

### File: adapter_strategy.md
### FPA: ILF
### PDC: ARC.RTM
### ID: sbz_org01.sbz_p01.ah_g01._ah_opg01.ah_opj01.LRA-001
### Class: Architecture / AI Inference
### Status: ACTIVE — Planning Phase
### Author: Claude Code
### Reviewer: Project Lead

---

## 1. Overview

This document defines the planned LoRA adapter ecosystem for AgentHub's ub02 inference stack.
The goal is a set of specialist adapters loaded dynamically per task, with a router adapter that
selects the right specialist based on the incoming request.

**Primary base model**: `Qwen3-8B-Q4_K_M.gguf` (pure transformer, port 8083)
- Pure transformer: ~30-50 tok/s generation on 780M iGPU
- llama.cpp supports hot-swap via `POST /lora` with per-adapter `scale` parameter

**Secondary base model**: `Qwen2.5-7B-Instruct-Q4_K_M` (port TBD, pending download)
- Stronger code/tool-use training baseline from the 2.5 series
- Candidate base for code_tools and rag_synthesis adapters

**Hybrid model (retained)**: `Qwen3.5-9B-Q4_K_M.gguf` (port 8080)
- Kept as fallback / long-context jobs where 32K window matters more than speed

---

## 2. Hot-Swap Mechanism

llama.cpp's `POST /lora` endpoint allows runtime adapter switching without model reload:

```json
POST http://ub02:8083/lora
[
  { "id": 0, "path": "/home/erol/models/lora/adapters/agenthub_domain.gguf", "scale": 1.0 },
  { "id": 1, "path": "/home/erol/models/lora/adapters/code_tools.gguf",       "scale": 0.0 }
]
```

- `scale: 1.0` — adapter active, weights applied to inference
- `scale: 0.0` — adapter loaded in VRAM but dormant (zero-cost to activate later)
- Multiple adapters can be co-resident; only `scale > 0` adapters affect output
- `--lora-init-without-apply` flag (set in llama-8b.service) is required for this

**Constraint**: llama.cpp applies one effective LoRA stack per inference call. Multiple active
adapters (scale > 0) are additively merged at inference time. This is useful for blending
domain + skill adapters but increases VRAM pressure. Start with one adapter active at a time.

---

## 3. Planned Specialist Adapters

| ID | Filename | Base | Domain | Training Source | Status |
|:---|:---------|:-----|:-------|:----------------|:-------|
| LRA-D01 | `agenthub_domain.gguf` | Qwen3-8B | AHC conventions, cluster topology, MQTT, doc standards | `dataset.jsonl` (4,276 pairs) | READY TO TRAIN |
| LRA-C01 | `code_tools.gguf` | Qwen2.5-7B | Shell/SSH tool use, file ops, JS/TS patterns | `backend/src/**` + tool call traces | PLANNED |
| LRA-R01 | `rag_synthesis.gguf` | Qwen3-8B | RAG chunk condensation, citation formatting | RAG query logs from retriever-service | PLANNED |
| LRA-W01 | `web_osint.gguf` | Qwen3-8B | Web search strategy, multi-hop OSINT, RSS parsing | Successful agent traces | PLANNED |
| LRA-DM01 | `document_modifier.gguf` | Qwen3-8B | Template-driven document and header modification, including governed-chain propagation | Approved before/after document diffs, template pairs, review-closeout edits | PLANNED |
| LRA-RT01 | `router.gguf` | Qwen3-8B | Query classification → adapter selection | Synthetic from all adapter sets | PLANNED |

---

## 4. Router Adapter (LRA-RT01)

The router is a LoRA adapter trained on a classification task: given an incoming query, output
a JSON list of adapter IDs to activate.

### 4.1. Input / Output

```
System: You are an AgentHub request router. Classify the query into adapter IDs.
        Output JSON only: {"adapters": ["agenthub_domain"], "confidence": 0.9}
        Available: agenthub_domain, code_tools, rag_synthesis, web_osint

User:   What MQTT topics does the heartbeat service publish to?
Assistant: {"adapters": ["agenthub_domain"], "confidence": 0.95}
```

### 4.2. Training Data Generation

The router training set is synthetic — generated from the specialist adapter training pairs by
labeling each pair with the adapter it belongs to:

```python
# build_router_dataset.py  (to be created)
# For each pair in dataset.jsonl -> label "agenthub_domain"
# For each pair in code_tools_dataset.jsonl -> label "code_tools"
# Generate 500-1000 pairs per adapter class
# Output: router_dataset.jsonl
```

### 4.3. Router Invocation in retriever-service

The retriever-service will call the router as the **first hop** of every agent run:

```javascript
// In runAgent(), before the main loop:
const routerResponse = await callLLM(pool, routerSystemPrompt, query, [], 256);
const { adapters } = JSON.parse(routerResponse);
await activateAdapters(adapters);   // POST /lora to 8083
// then proceed with main agentic loop
```

---

## 5. Port Assignment

| Model | Port | Role | Note |
|:------|:-----|:-----|:-----|
| Qwen3.5-9B (hybrid Mamba) | 8080 | Primary inference | Current default |
| Qwen3-4B (embeddings) | 8081 | RAG embeddings | --embeddings mode |
| retriever-service | 8082 | Agentic loop | Node.js |
| Qwen3-8B (pure transformer + LoRA) | 8083 | Specialist inference | --lora-init-without-apply |
| Qwen2.5-7B-Instruct | 8084 | Code/tool specialist | TBD — pending download |

---

## 6. Training Pipeline

### 6.1. Phase 1 — Domain Adapter (LRA-D01)

Prerequisites:
- GPU with 8GB+ VRAM or cloud (A100/L4) — ub02 780M iGPU may be sufficient for rank-16 LoRA
- `unsloth` or `axolotl` framework
- `dataset.jsonl` already generated (`build_dataset.py`)

```bash
# On ub02 or cloud GPU
pip install unsloth
python3 train_lora.py   --base_model Qwen/Qwen3-8B   --dataset /home/erol/models/lora/training/dataset.jsonl   --output /home/erol/models/lora/adapters/agenthub_domain   --rank 16 --alpha 32 --epochs 3 --lr 2e-4
```

Output: GGUF-converted adapter at `/home/erol/models/lora/adapters/agenthub_domain.gguf`

### 6.2. Phase 2 — Collect Specialist Training Data

- **code_tools**: Extract tool-call traces from `comms/AGENT_BUS.md` + retriever-service logs
- **rag_synthesis**: Capture (chunks_in, answer_out) pairs from RAG queries
- **web_osint**: Capture successful multi-hop search traces
- **document_modifier**: Capture approved before/after governance-document edits, especially header migrations, version-history bumps, and template-preserving transformations. Keep the corpus template-driven and diff-based instead of hardcoding file-specific strings.

### 6.3. Phase 3 — Router Training

Once 3+ specialist datasets exist, build synthetic router classification data and train LRA-RT01.

---

## 7. VRAM Budget (780M iGPU, 12.9GB dynamic)

| Component | VRAM |
|:----------|:-----|
| Qwen3-8B base model | ~5.0 GB |
| KV cache (32K ctx) | ~1.0 GB |
| Compute buffers | ~0.5 GB |
| LoRA adapter (rank 16) | ~50 MB |
| Available for co-loaded adapters | ~6.4 GB |

Up to 3-4 adapters can be co-resident in VRAM simultaneously (50MB each). The base model
and KV cache are the dominant costs.

---

## 8. Open Questions (deferred)

- **Multi-adapter blending**: llama.cpp supports additive merge of multiple active adapters.
  Feasibility of simultaneously applying domain + skill adapters TBD.
- **Per-request adapter switching**: llama.cpp `POST /lora` is server-global, not per-request.
  Concurrent requests with different adapters would require multiple server instances or a
  queuing layer. The existing `SerialQueue` in retriever-service already serializes requests
  and is the correct insertion point for adapter swap calls.
- **Adapter versioning**: as training data evolves, adapters will need version IDs aligned
  with the `dataset.jsonl` manifest versioning.

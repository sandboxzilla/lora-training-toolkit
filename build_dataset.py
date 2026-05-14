#!/usr/bin/env python3
"""
build_dataset.py — AgentHub LoRA training corpus builder
Produces:
  - dataset.jsonl   : ChatML-format pairs for Qwen3 instruction-tuning
  - manifest.md     : review manifest (sources, stats, samples)

Usage: python3 build_dataset.py
"""

import os, re, json, textwrap
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[4]   # agent_hub_main/
DOCS_DIR    = REPO_ROOT / "docs"
INIT_DIR    = REPO_ROOT / "backend" / "init-scripts"
SCRIPTS_DIR = REPO_ROOT / "backend" / "scripts" / "ub02"
OUT_DIR     = Path(__file__).parent
DATASET_OUT = OUT_DIR / "dataset.jsonl"
MANIFEST_OUT= OUT_DIR / "manifest.md"

# Qwen3 system prompt (condensed, same as DEFAULT_SYSTEM_PROMPT in retriever-service)
SYSTEM = (
    "You are an agentic AI assistant named Qwen, running as retriever-service on ub02 "
    "(192.168.11.65) in the AgentHub cluster. "
    "Cluster: ub01 (192.168.11.111) = MariaDB ah_db01 + MQTT :1883 + Redis + API :3000. "
    "ub02 = Qwen3-9B :8080, embeddings :8081, retriever-service :8082. "
    "Fleet nodes n20/n40/n50/n60 at 192.168.11.x. Vault on n60:8200. "
    "You have tools: shell_exec, ssh_exec, file_read, file_write, web_search, web_fetch, "
    "http_request, db_query, kv_get, kv_set, vault_get, mqtt_read, mqtt_publish, rag_query."
)

# Docs to include (primary design docs, skip .reports / tasks / .plans)
SKIP_DIRS = {".reports", "tasks", ".plans", ".hold", ".arc", ".claude", ".architect"}

# Key SQL schema files (schema-only, not seed data)
SQL_INCLUDE = [
    "01-schema.sql", "05-schema.sql", "08-multitenant-vault.sql",
    "09-finops-billing.sql", "10-communications.sql", "10-hierarchical-rag.sql",
    "40-agent-control.sql", "60-hlmm-schema.sql", "61-doccore-schema.sql",
    "65-ah-db01-core-tables.sql", "67-doccore-pdc-metadata.sql",
    "68-doccore-vector-schema.sql",
]

# Extra source files to include verbatim
EXTRA_FILES = [
    REPO_ROOT / "CONTEXT.md",
    REPO_ROOT / "CLAUDE.md",
    SCRIPTS_DIR / "retriever-service.js",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(path: Path) -> bool:
    """Return True if any component of the path is in SKIP_DIRS."""
    return any(part in SKIP_DIRS for part in path.parts)


def strip_front_matter(text: str) -> str:
    """Remove YAML front-matter if present."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip()
    return text


def split_sections(text: str, min_chars: int = 200) -> list[tuple[str, str]]:
    """
    Split markdown by ## headings.
    Returns list of (heading, body) tuples with body >= min_chars.
    """
    pattern = re.compile(r'^#{1,3} +(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start   = m.end()
        end     = matches[i+1].start() if i+1 < len(matches) else len(text)
        body    = text[start:end].strip()
        if len(body) >= min_chars:
            sections.append((heading, body))
    return sections


def trim(text: str, max_chars: int = 3000) -> str:
    """Trim text to max_chars, appending ellipsis if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit('\n', 1)[0] + "\n\n[...truncated for training...]"


def make_pair(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ── Source collectors ─────────────────────────────────────────────────────────

def collect_docs() -> list[tuple[str, str]]:
    """Yield (label, full_text) for primary markdown docs."""
    result = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        if should_skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        label = str(path.relative_to(REPO_ROOT))
        result.append((label, strip_front_matter(text)))
    return result


def collect_sql() -> list[tuple[str, str]]:
    result = []
    for name in SQL_INCLUDE:
        p = INIT_DIR / name
        if p.exists():
            result.append((f"init-scripts/{name}", p.read_text(errors="replace")))
    return result


def collect_extra() -> list[tuple[str, str]]:
    result = []
    for p in EXTRA_FILES:
        if p.exists():
            label = str(p.relative_to(REPO_ROOT))
            text = p.read_text(encoding="utf-8", errors="replace")
            if p.suffix == ".md":
                text = strip_front_matter(text)
            result.append((label, text))
    return result


# ── Pair generators ───────────────────────────────────────────────────────────

def pairs_from_doc(label: str, text: str) -> list[dict]:
    """Generate multiple Q&A pairs from a markdown document."""
    pairs = []
    filename = label.split("/")[-1].replace(".md", "").replace("_", " ")

    # 1. Full-doc summarization pair
    first_1000 = text[:1000].strip()
    if first_1000:
        pairs.append(make_pair(
            f"What is the purpose and content of the AgentHub document '{filename}'?",
            trim(first_1000, 1500),
        ))

    # 2. Per-section Q&A pairs
    for heading, body in split_sections(text):
        q = f"In the AgentHub documentation, what does the section '{heading}' cover?"
        pairs.append(make_pair(q, trim(body, 2000)))

    # 3. "Where is X documented?" pair — reversed lookup
    pairs.append(make_pair(
        f"Where in the AgentHub repository can I find documentation about '{filename}'?",
        f"This topic is documented in `{label}` in the AgentHub repository.",
    ))

    return pairs


def pairs_from_sql(label: str, text: str) -> list[dict]:
    """Generate Q&A pairs from SQL schema files."""
    pairs = []
    name = label.split("/")[-1].replace(".sql", "").replace("-", " ")

    # Extract CREATE TABLE names
    tables = re.findall(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?`?(\w+)`?', text, re.IGNORECASE)
    if tables:
        pairs.append(make_pair(
            f"What tables are defined in the AgentHub database migration '{name}'?",
            f"The following tables are defined: {', '.join(tables)}.\n\nFull schema:\n```sql\n{trim(text, 2000)}\n```",
        ))

    # CREATE TABLE bodies as individual pairs
    ct_pattern = re.compile(
        r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?`?(\w+)`?\s*\(([^;]+?)\);',
        re.IGNORECASE | re.DOTALL,
    )
    for m in ct_pattern.finditer(text):
        tname, body = m.group(1), m.group(2).strip()
        pairs.append(make_pair(
            f"What are the columns and constraints of the `{tname}` table in the AgentHub database?",
            f"```sql\nCREATE TABLE `{tname}` (\n{body}\n);\n```",
        ))

    return pairs


def pairs_from_extra(label: str, text: str) -> list[dict]:
    """Pairs for key extra files (CONTEXT, CLAUDE, retriever-service)."""
    pairs = []
    name = label.split("/")[-1]

    if name in ("CONTEXT.md", "CLAUDE.md"):
        # Section-level pairs only (these are large)
        for heading, body in split_sections(text, min_chars=100):
            q = f"What does the AgentHub '{name}' say about '{heading}'?"
            pairs.append(make_pair(q, trim(body, 2000)))
    elif name == "retriever-service.js":
        # Chunk by // ── section comments
        chunks = re.split(r'// ── (.+?) ─+', text)
        section_name = "preamble"
        for i, chunk in enumerate(chunks):
            if i % 2 == 1:
                section_name = chunk.strip()
                continue
            code = chunk.strip()
            if len(code) > 300:
                pairs.append(make_pair(
                    f"In the AgentHub retriever-service.js, what does the '{section_name}' section do?",
                    trim(f"```javascript\n{code}\n```", 2500),
                ))
        # Full tool list
        tools = re.findall(r"(\w+):\s+(?:make\w+|exec\w+|\w+),", text)
        if tools:
            pairs.append(make_pair(
                "What tools does the AgentHub Qwen retriever-service expose?",
                "The retriever-service exposes these tools:\n" + "\n".join(f"- `{t}`" for t in tools),
            ))
    return pairs


# ── Hardcoded domain knowledge pairs ─────────────────────────────────────────

DOMAIN_PAIRS = [
    make_pair(
        "What is AgentHub?",
        "AgentHub is a multi-tenant AI agent orchestration platform. It coordinates AI agents "
        "(Worker Bees) via MQTT messaging, stores state in MariaDB on ub01, serves a REST API "
        "on port 3000, and provides a Flutter Web dashboard. Agents run in Podman containers on "
        "cluster nodes. The project follows the AHC coordinate system for document governance."
    ),
    make_pair(
        "What is the AgentHub cluster topology?",
        "The cluster consists of:\n"
        "- ub01 (192.168.11.111): Control plane — MariaDB ah_db01, MQTT :1883/:9001, Redis, API :3000\n"
        "- ub02 (192.168.11.65): AI inference — Qwen3-9B :8080, Qwen3-4B embeddings :8081, retriever-service :8082\n"
        "- n20, n40, n50, n60: Fleet nodes on 192.168.11.x (wlan) / 192.168.1.x (eth)\n"
        "- n60 (192.168.11.60): HashiCorp Vault :8200\n"
        "Mosquitto MQTT bridges on n50 forward starbush/fleet/# to ub01."
    ),
    make_pair(
        "What is the AHC coordinate system?",
        "AHC (AgentHub Coordinate) is the document governance standard. Every document has a "
        "coordinate like sbz_org01.sbz_p01.ah_g01._ah_opg01.ah_opj01.DOC-001. "
        "Coordinates encode org → program → group → sub-group → project → document ID. "
        "The File Path Anchor (FPA) declares whether a file is an Internal Logical File (ILF) "
        "or External Interface File (EIF). The PDC (Product Delivery Category) classifies the "
        "document type: ARC, RTM, COR, DAT, INT, etc."
    ),
    make_pair(
        "What MQTT topics does AgentHub use?",
        "Key MQTT topic patterns:\n"
        "- `starbush/fleet/{node}/heartbeat` — node heartbeat\n"
        "- `starbush/fleet/{node}/telemetry` — CPU/mem/load telemetry\n"
        "- `system/hives/{node}/heartbeat` — hive heartbeat (indexed by telemetry_bridge)\n"
        "- `agenthub/fleet/cmd/{node}` — inbound commands to fleet nodes\n"
        "- `agenthub/agents/{id}/status` — agent status updates\n"
        "MQTT broker: ub01:1883 (TCP) and ub01:9001 (WebSocket for Flutter browser)."
    ),
    make_pair(
        "What is the retriever-service and what can it do?",
        "The retriever-service is a Node.js agentic loop running on ub02:8082. It uses Qwen3-9B "
        "via llama.cpp on port 8080 with an OpenAI-compatible API. It provides 16 tools:\n"
        "shell_exec, ssh_exec, file_read, file_write — system operations\n"
        "web_search, web_fetch — internet access\n"
        "http_request — arbitrary HTTP calls\n"
        "db_query — parameterized SQL against ah_db01 on ub01\n"
        "kv_get, kv_set — persistent key-value store in ~/.cache/agenthub/kv_store.json\n"
        "vault_get — read secrets from Vault on n60:8200\n"
        "secret_get — read local secrets from ~/.config/agenthub/secrets/\n"
        "mqtt_read, mqtt_publish — MQTT messaging\n"
        "rag_query — vector similarity search over doccore embeddings\n"
        "Requests are serialized via SerialQueue to prevent KV cache exhaustion."
    ),
    make_pair(
        "How do I access HashiCorp Vault from the AgentHub cluster?",
        "Vault runs on n60 at http://192.168.11.60:8200. On ub02, the service token is at "
        "~/.config/agenthub/secrets/vault_token (agenthub-runtime policy). "
        "On n50, AppRole credentials are at ~/.config/vault/role_id and ~/.config/vault/secret_id; "
        "the login script posts to /v1/auth/approle/login to get a client token. "
        "Use the vault_get tool in retriever-service to read secrets without handling auth manually. "
        "Always prefer vault_get over secret_get for credentials."
    ),
    make_pair(
        "What database schema does AgentHub use?",
        "AgentHub uses MariaDB 11.4 on ub01, database ah_db01. Key table groups:\n"
        "- Core: tenants, users, projects, agents, tasks\n"
        "- RBAC: roles, permissions, tenant_users, team_memberships\n"
        "- FinOps: tenant_budgets, tenant_usage_logs\n"
        "- Doccore: doccore_nodes, doccore_sections, doccore_section_embeddings (VECTOR 2560)\n"
        "- HLMM: hlmm_memory_nodes, hlmm_edges, hlmm_tier_transitions\n"
        "- Agent control: agent_control_policies, agent_commands, agent_chat_sessions, agent_chat_messages\n"
        "- Knowledge: knowledge_embeddings (VECTOR 2560) with HNSW cosine index\n"
        "All queries must be scoped by tenant_id from JWT claims."
    ),
    make_pair(
        "What are the AgentHub agent roles?",
        "AgentHub defines four agent roles:\n"
        "1. Lead Architect — proposes designs, writes architecture docs, assigns tasks\n"
        "2. Sentinel Architect — reviews and approves designs; cannot review their own work\n"
        "3. Worker Bee — implements tasks, writes tests, runs CI checks\n"
        "4. Lead / Manager — clarifies missions, splits work, preserves audit trails\n"
        "Claude Code acts as all roles depending on context. Gemini CLI handles research and "
        "single-file edits. Codex CLI handles boilerplate and test writing."
    ),
    make_pair(
        "What is the AgentHub RAG pipeline?",
        "RAG (Retrieval-Augmented Generation) uses:\n"
        "- Embeddings: Qwen3-4B on ub02:8081, /v1/embeddings endpoint, VECTOR(2560)\n"
        "- Embedding mode: hybrid_fused (combines dense + BM25 keyword search)\n"
        "- Tables: knowledge_embeddings and doccore_section_embeddings\n"
        "- Index: HNSW cosine similarity\n"
        "- Ingest: ingest-worker.js on ub02 via PM2, refresh-agenthub-docs.sh triggers re-index\n"
        "- Query: rag_query tool in retriever-service sends a question, gets top-K chunks back"
    ),
    make_pair(
        "What is the HLMM in AgentHub?",
        "HLMM (Hierarchical Long-term Memory Manager) is a tiered memory system for agents. "
        "It stores memory nodes with tier levels (T0=working, T1=session, T2=long-term, T3=archival). "
        "Nodes are promoted/demoted based on access frequency and staleness scores. "
        "The schema is in 60-hlmm-schema.sql. Distillation_worker.ts handles promotion via "
        "LLM-based summarization using Qwen3-4B embeddings."
    ),
    make_pair(
        "How are llama.cpp models managed on ub02?",
        "Two systemd services manage llama.cpp on ub02:\n"
        "- llama-9b.service: Qwen3.5-9B Q4_K_M on port 8080, -c 32768, --parallel 1 (chat/queries)\n"
        "- llama-4b.service: Qwen3-4B Q4_K_M on port 8081, -c 512, --parallel 4, --embeddings (doccore)\n"
        "Both use User=erol, Environment PATH includes /home/erol/.local/bin, "
        "LD_LIBRARY_PATH=/home/erol/.local/lib for libmtmd.so. "
        "Model files in /home/erol/models/. Services auto-restart on failure."
    ),
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_pairs = []
    source_stats = []

    # Hardcoded domain knowledge
    all_pairs.extend(DOMAIN_PAIRS)
    source_stats.append(("Domain knowledge (hardcoded)", len(DOMAIN_PAIRS), 0))

    # Extra files (CONTEXT, CLAUDE, retriever-service)
    for label, text in collect_extra():
        pairs = pairs_from_extra(label, text)
        all_pairs.extend(pairs)
        source_stats.append((label, len(pairs), len(text)))

    # SQL schema files
    sql_count = 0
    for label, text in collect_sql():
        pairs = pairs_from_sql(label, text)
        all_pairs.extend(pairs)
        sql_count += len(pairs)
    source_stats.append((f"SQL schema files ({len(SQL_INCLUDE)} files)", sql_count, 0))

    # Markdown docs
    doc_pairs = 0
    doc_files = 0
    for label, text in collect_docs():
        pairs = pairs_from_doc(label, text)
        all_pairs.extend(pairs)
        doc_pairs += len(pairs)
        doc_files += 1
    source_stats.append((f"Markdown docs ({doc_files} files)", doc_pairs, 0))

    # Deduplicate by user content (keep first occurrence)
    seen = set()
    deduped = []
    for p in all_pairs:
        key = p["messages"][1]["content"][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    all_pairs = deduped

    # Estimate tokens (rough: 4 chars ≈ 1 token)
    total_chars = sum(
        len(m["content"]) for p in all_pairs for m in p["messages"]
    )
    est_tokens = total_chars // 4

    # Write JSONL
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write manifest
    sample_pairs = all_pairs[:5]
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        f.write(f"""# AgentHub LoRA Training Dataset — Manifest

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %UTC')}
**Output**: `dataset.jsonl`
**Format**: ChatML (Qwen3 instruction-tuning)

---

## 1. Summary Statistics

| Metric | Value |
| :----- | ----: |
| Total training pairs | {len(all_pairs):,} |
| Estimated tokens | ~{est_tokens:,} |
| Deduplicated pairs removed | {sum(s[1] for s in source_stats) - len(all_pairs):,} |

---

## 2. Source Breakdown

| Source | Pairs |
| :----- | ----: |
""")
        for label, count, _ in source_stats:
            f.write(f"| {label} | {count:,} |\n")

        f.write("""
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

""")
        for i, p in enumerate(sample_pairs, 1):
            user = p["messages"][1]["content"]
            asst = p["messages"][2]["content"][:300]
            f.write(f"### Sample {i}\n\n")
            f.write(f"**User**: {user[:200]}\n\n")
            f.write(f"**Assistant**: {asst}{'...' if len(p['messages'][2]['content']) > 300 else ''}\n\n")
            f.write("---\n\n")

        f.write("""## 5. Training Recommendations

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
""")

    print(f"Done.")
    print(f"  Pairs written : {len(all_pairs):,}")
    print(f"  Est. tokens   : ~{est_tokens:,}")
    print(f"  Output JSONL  : {DATASET_OUT}")
    print(f"  Manifest      : {MANIFEST_OUT}")


if __name__ == "__main__":
    main()

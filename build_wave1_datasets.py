#!/usr/bin/env python3
"""
build_wave1_datasets.py — Synthetic dataset generator for Wave 1 reviewer adapters.

Uses the local llama.cpp server (localhost:8080) to generate ideal reviewer
responses for seed inputs, producing ChatML-format JSONL training files for:
  - LRA-CR01: code_reviewer_dataset.jsonl
  - LRA-DR01: doc_reviewer_dataset.jsonl
  - LRA-AR01: arch_reviewer_dataset.jsonl
  - LRA-TR01: test_reviewer_dataset.jsonl

Usage:
  python3 build_wave1_datasets.py [--adapter cr01|dr01|ar01|tr01|all]
                                  [--server http://localhost:8080]
                                  [--samples N]   # max per adapter (default 150)
                                  [--output-dir DIR]

Requirements: requests (pip install requests)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

# ---------------------------------------------------------------------------
# System prompts — define each reviewer's expected output structure
# ---------------------------------------------------------------------------

CR01_SYSTEM = (
    "You are a senior code reviewer for the AgentHub project. "
    "Analyze the provided code for: bugs (logic errors, off-by-one, null dereference), "
    "security issues (injection, auth bypass, insecure defaults), "
    "performance problems, and style violations. "
    "Respond with a structured JSON review:\n"
    "{\n"
    "  \"verdict\": \"PASS\" | \"NEEDS_REVISION\" | \"FAIL\",\n"
    "  \"summary\": \"<one sentence>\",\n"
    "  \"findings\": [\n"
    "    {\"severity\": \"BLK|NBK|INFO\", \"line\": <int|null>, "
    "\"type\": \"<category>\", \"description\": \"<detail>\", \"fix\": \"<suggestion>\"}\n"
    "  ]\n"
    "}\n"
    "BLK = must fix before merge. NBK = should fix. INFO = advisory.\n"
    "Output only the JSON object. No prose before or after."
)

DR01_SYSTEM = (
    "You are a document reviewer for the AgentHub project following the AgentHub "
    "Documentation Standard (o2004). Review documents for: structural compliance "
    "(required header fields, version table, ToC), content accuracy, completeness, "
    "and clarity. Findings must be classified BLK (blocking — document cannot be "
    "approved until resolved) or NBK (non-blocking — should be fixed).\n"
    "Respond with a structured JSON review:\n"
    "{\n"
    "  \"verdict\": \"APPROVED\" | \"APPROVED_WITH_NBK\" | \"REJECTED\",\n"
    "  \"summary\": \"<one sentence>\",\n"
    "  \"findings\": [\n"
    "    {\"severity\": \"BLK|NBK\", \"section\": \"<section or field>\", "
    "\"description\": \"<detail>\", \"fix\": \"<suggestion>\"}\n"
    "  ]\n"
    "}\n"
    "Output only the JSON object. No prose before or after."
)

AR01_SYSTEM = (
    "You are an architecture reviewer for the AgentHub project. "
    "Evaluate architectural designs, proposals, and diagrams for: "
    "correctness (does the design achieve its stated goals), "
    "scalability (will it hold under 10x load), "
    "security (attack surfaces, trust boundaries), "
    "coupling and cohesion (are boundaries right), "
    "and alignment with AgentHub principles (multi-tenant, event-driven, no direct DB access by agents). "
    "Respond with a structured JSON review:\n"
    "{\n"
    "  \"verdict\": \"APPROVED\" | \"NEEDS_REVISION\" | \"REJECTED\",\n"
    "  \"summary\": \"<one sentence>\",\n"
    "  \"strengths\": [\"<item>\"],\n"
    "  \"findings\": [\n"
    "    {\"severity\": \"BLK|NBK\", \"component\": \"<component>\", "
    "\"description\": \"<detail>\", \"fix\": \"<suggestion>\"}\n"
    "  ]\n"
    "}\n"
    "Output only the JSON object. No prose before or after."
)

TR01_SYSTEM = (
    "You are a test reviewer for the AgentHub project. "
    "Evaluate test suites and individual tests for: "
    "coverage (are the happy path and error paths tested), "
    "correctness (do assertions verify the right thing), "
    "isolation (proper mocking, no shared state), "
    "brittleness (does the test break on irrelevant changes), "
    "and completeness (are edge cases covered). "
    "Respond with a structured JSON review:\n"
    "{\n"
    "  \"verdict\": \"PASS\" | \"NEEDS_REVISION\" | \"FAIL\",\n"
    "  \"coverage_assessment\": \"<brief>\",\n"
    "  \"findings\": [\n"
    "    {\"severity\": \"BLK|NBK|INFO\", \"test\": \"<test name or line>\", "
    "\"description\": \"<detail>\", \"fix\": \"<suggestion>\"}\n"
    "  ]\n"
    "}\n"
    "Output only the JSON object. No prose before or after."
)

# ---------------------------------------------------------------------------
# Seed inputs — diverse inputs that a real reviewer would encounter
# ---------------------------------------------------------------------------

CODE_SEEDS = [
    # SQL injection
    ("Review this Node.js route handler:\n```js\napp.get('/user', async (req, res) => {\n"
     "  const id = req.query.id;\n"
     "  const row = await db.query(`SELECT * FROM users WHERE id = ${id}`);\n"
     "  res.json(row);\n});\n```"),
    # Missing auth check
    ("Review this Express middleware:\n```js\nrouter.delete('/api/agent/:id', async (req, res) => {\n"
     "  await db.query('DELETE FROM agents WHERE id = ?', [req.params.id]);\n"
     "  res.json({ deleted: true });\n});\n```"),
    # Off-by-one
    ("Review this Python function:\n```python\ndef get_last_n(items, n):\n"
     "    return items[len(items) - n:len(items) + 1]\n```"),
    # Race condition / missing lock
    ("Review this TypeScript method:\n```ts\nasync function incrementCounter(key: string) {\n"
     "  const val = await redis.get(key);\n"
     "  await redis.set(key, parseInt(val || '0') + 1);\n"
     "  return parseInt(val || '0') + 1;\n}\n```"),
    # Memory leak — event listener not removed
    ("Review this JavaScript component cleanup:\n```js\nfunction startPolling(handler) {\n"
     "  const id = setInterval(handler, 1000);\n"
     "  window.addEventListener('resize', handler);\n"
     "  return { stop: () => clearInterval(id) };\n}\n```"),
    # Hardcoded secret
    ("Review this configuration loader:\n```python\nDB_PASSWORD = 'admin123'\nJWT_SECRET = 'secret'\n"
     "def get_db():\n    return connect(host='localhost', password=DB_PASSWORD)\n```"),
    # Missing error handling
    ("Review this async function:\n```ts\nasync function fetchUserData(userId: string) {\n"
     "  const response = await fetch(`/api/users/${userId}`);\n"
     "  const data = await response.json();\n"
     "  return data;\n}\n```"),
    # N+1 query
    ("Review this ORM usage:\n```python\ndef get_agents_with_tasks():\n"
     "    agents = Agent.objects.all()\n"
     "    for agent in agents:\n"
     "        agent.task_count = Task.objects.filter(agent=agent).count()\n"
     "    return agents\n```"),
    # Correct and clean code
    ("Review this TypeScript JWT validation:\n```ts\nexport async function validateToken(token: string): Promise<JWTPayload> {\n"
     "  try {\n    return await jwtVerify(token, publicKey, { algorithms: ['RS256'] });\n"
     "  } catch (err) {\n    throw new UnauthorizedError('Invalid or expired token');\n  }\n}\n```"),
    # Integer overflow
    ("Review this C-like calculation:\n```js\nfunction paginate(total, page, pageSize) {\n"
     "  const offset = page * pageSize;\n"
     "  return { offset, limit: pageSize, total };\n}\n```"),
    # XSS
    ("Review this template rendering:\n```js\nfunction renderMessage(msg) {\n"
     "  document.getElementById('output').innerHTML = msg.text;\n}\n```"),
    # Good multi-tenant query
    ("Review this database query function:\n```ts\nasync function getTasks(tenantId: string, userId: string) {\n"
     "  return db.query(\n    'SELECT * FROM tasks WHERE tenant_id = ? AND assigned_to = ?',\n"
     "    [tenantId, userId]\n  );\n}\n```"),
    # Missing rate limiting
    ("Review this login endpoint:\n```ts\napp.post('/login', async (req, res) => {\n"
     "  const { email, password } = req.body;\n"
     "  const user = await User.findByEmail(email);\n"
     "  if (user && await bcrypt.compare(password, user.hash)) {\n"
     "    res.json({ token: signJWT(user) });\n"
     "  } else { res.status(401).json({ error: 'Invalid credentials' }); }\n});\n```"),
    # TOCTOU
    ("Review this file handler:\n```python\nimport os\ndef safe_read(path):\n"
     "    if os.path.exists(path):\n"
     "        with open(path) as f:\n            return f.read()\n    return None\n```"),
    # Clean MQTT publish
    ("Review this MQTT publisher:\n```ts\nasync function publishEvent(topic: string, payload: object) {\n"
     "  if (!mqttClient.connected) throw new Error('MQTT not connected');\n"
     "  const msg = JSON.stringify({ ...payload, ts: new Date().toISOString() });\n"
     "  await mqttClient.publishAsync(topic, msg, { qos: 1 });\n}\n```"),
]

DOC_SEEDS = [
    # Missing required header fields
    ("Review this document header:\n```\n# AgentHub Deployment Guide\n\n"
     "This document describes how to deploy AgentHub to production.\n\n"
     "## Prerequisites\n- Docker\n- Node.js 20\n```"),
    # Missing version table
    ("Review this API specification excerpt:\n```markdown\n# REST API Manual\n\n"
     "| Field | Value |\n|-------|-------|\n| File | api_manual.md | ID | api001 |\n\n"
     "## Endpoints\n### POST /api/agents\nCreates a new agent.\n```"),
    # Good header
    ("Review this document header:\n```markdown\n### File: o2001_deployment_runbook.md\n"
     "### FPA: ILF\n### PDC: OPS.DPL\n### ID: sbz_org01.sbz_p01.ah_g01._ah_opg01.o2001\n"
     "### Status: ACTIVE / MANDATORY\n### Class: Operations Runbook\n### Author: DevOps Lead\n"
     "### Reviewer: Sentinel Architect\n\n## Version History\n| Version | Date | Author | Status | Notes |\n"
     "| 1.0 | 2026-03-17 | DevOps Lead | Active | Initial |\n\n## Table of Contents\n...\n```"),
    # Proposed described as implemented
    ("Review this API documentation excerpt:\n```markdown\n## POST /api/embeddings\n"
     "Generates embeddings for a text input using the Qwen3-Embedding-4B model.\n\n"
     "**Request:**\n```json\n{\"text\": \"string\"}\n```\n\n"
     "**Response:**\n```json\n{\"embedding\": [...]}\n```\n```"),
    # Missing reviewer
    ("Review this document header:\n```markdown\n| Field | Value |\n|-------|-------|\n"
     "| File | design_doc.md | Author | Lead Architect |\n"
     "| Status | DRAFT | Class | Architecture Decision |\n```"),
    # Unclear section
    ("Review this runbook section:\n```markdown\n## 3. Deployment\nDeploy the thing. Make sure "
     "it works. Test afterwards.\n\n## 4. Rollback\nDo the reverse of deployment.\n```"),
    # Good standards-compliant doc excerpt
    ("Review this architecture document section:\n```markdown\n## 3. Component Boundaries\n\n"
     "### 3.1. Orchestrator\nThe orchestrator receives task envelopes via MQTT topic "
     "`agenthub/tasks/new`. It validates the envelope schema, checks the tenant budget, "
     "and dispatches to the appropriate agent. The orchestrator never accesses the database "
     "directly — all persistence is via the DBSS gateway.\n\n"
     "### 3.2. DBSS Gateway\nThe DBSS gateway is the sole database-facing component. "
     "Agents communicate with it via MQTT request/reply.\n```"),
    # Version history missing dates
    ("Review this version history table:\n```markdown\n| Version | Date | Author | Status | Notes |\n"
     "|---------|------|--------|--------|-------|\n"
     "| 1.0 | TBD | Lead | Draft | Initial |\n"
     "| 1.1 | TBD | Lead | Active | Fixes |\n```"),
]

ARCH_SEEDS = [
    # Single point of failure
    ("Review this architecture:\nThe orchestrator is a single Node.js process that receives "
     "all MQTT messages, routes them to agents, and writes results back to the database. "
     "All agents connect directly to this orchestrator over TCP. If the orchestrator goes down, "
     "all agent operations stop until it restarts."),
    # Direct DB access by agents
    ("Review this proposed agent design:\nThe Code Reviewer agent will have a MariaDB "
     "connection pool and query the `tasks`, `code_reviews`, and `agents` tables directly "
     "to fetch work items and store results. This avoids the overhead of MQTT message passing."),
    # Good event-driven multi-tenant design
    ("Review this architecture:\nEach tenant's task queue is a separate MQTT topic namespace "
     "`agenthub/{tenant_id}/tasks/`. The orchestrator subscribes to all tenant namespaces and "
     "routes tasks to agents. Agent results are published to `agenthub/{tenant_id}/results/`. "
     "The DBSS gateway is the only component with database access."),
    # Shared mutable state
    ("Review this microservice design:\nThe embedding service and the RAG service share a "
     "Redis instance for caching. Both services read and write to the same key namespace "
     "without any coordination. Cache invalidation is done by TTL only."),
    # Hardcoded tenant limits
    ("Review this multi-tenancy design:\nEach tenant is limited to 100 concurrent agents, "
     "hardcoded in the orchestrator source code. Tenant budgets and quotas are stored in "
     "a config file that requires a restart to update."),
    # Good LoRA architecture
    ("Review this AI model serving architecture:\nA single llama.cpp server hosts the base "
     "Qwen3-8B model. Role-specific behavior is provided by LoRA adapters that are hot-swapped "
     "per request using the /lora-adapters API. The router adapter runs first to select the "
     "appropriate specialist adapter, which is then loaded for the main task. After task "
     "completion, the adapter is unloaded."),
    # Missing authentication layer
    ("Review this API gateway design:\nThe API gateway forwards all requests to backend "
     "services based on URL path. Services implement their own authentication. Some internal "
     "services have no authentication because they are 'internal only' and not exposed "
     "directly to the internet."),
    # Circuit breaker missing
    ("Review this service integration:\nThe orchestrator calls the embedding service synchronously "
     "for every incoming task. If the embedding service is slow or down, the orchestrator "
     "blocks and eventually times out, causing tasks to fail with a generic error."),
]

TEST_SEEDS = [
    # No error path testing
    ("Review these tests:\n```js\ndescribe('userService', () => {\n"
     "  it('creates a user', async () => {\n"
     "    const user = await userService.create({ email: 'a@b.com', password: 'pass' });\n"
     "    expect(user.id).toBeDefined();\n"
     "    expect(user.email).toBe('a@b.com');\n"
     "  });\n});\n```"),
    # Testing implementation not behavior
    ("Review this test:\n```js\nit('calls db.insert once', async () => {\n"
     "  const spy = jest.spyOn(db, 'insert');\n"
     "  await createAgent({ name: 'test' });\n"
     "  expect(spy).toHaveBeenCalledTimes(1);\n});\n```"),
    # Good comprehensive tests
    ("Review these tests:\n```ts\ndescribe('validateToken', () => {\n"
     "  it('accepts valid RS256 token', async () => {\n"
     "    const token = await signTestToken({ userId: '1', tenantId: 'acme' });\n"
     "    const payload = await validateToken(token);\n"
     "    expect(payload.userId).toBe('1');\n"
     "  });\n\n"
     "  it('rejects expired token', async () => {\n"
     "    const token = await signTestToken({ exp: Math.floor(Date.now()/1000) - 3600 });\n"
     "    await expect(validateToken(token)).rejects.toThrow('Invalid or expired token');\n"
     "  });\n\n"
     "  it('rejects wrong algorithm', async () => {\n"
     "    const token = jwt.sign({ userId: '1' }, 'secret', { algorithm: 'HS256' });\n"
     "    await expect(validateToken(token)).rejects.toThrow();\n"
     "  });\n});\n```"),
    # Shared state between tests
    ("Review these tests:\n```js\nlet db;\nbeforeAll(async () => { db = await connectDB(); });\n\n"
     "it('creates user', async () => {\n"
     "  await db.query('INSERT INTO users VALUES (1, \\\"alice\\\")');\n"
     "  const u = await db.query('SELECT * FROM users WHERE id = 1');\n"
     "  expect(u.name).toBe('alice');\n});\n\n"
     "it('lists users', async () => {\n"
     "  const users = await db.query('SELECT * FROM users');\n"
     "  expect(users.length).toBe(1);\n});\n```"),
    # Missing tenant isolation test
    ("Review these multi-tenant tests:\n```ts\ndescribe('TaskService', () => {\n"
     "  it('creates task', async () => {\n"
     "    const task = await taskService.create({ title: 'test', tenantId: 'acme' });\n"
     "    expect(task.id).toBeDefined();\n"
     "  });\n\n"
     "  it('lists tasks', async () => {\n"
     "    const tasks = await taskService.list({ tenantId: 'acme' });\n"
     "    expect(Array.isArray(tasks)).toBe(true);\n"
     "  });\n});\n```"),
    # Mock not reset
    ("Review these tests:\n```js\nconst mockFetch = jest.fn();\nglobal.fetch = mockFetch;\n\n"
     "it('fetches user', async () => {\n"
     "  mockFetch.mockResolvedValue({ json: () => ({ id: 1 }) });\n"
     "  const user = await getUser(1);\n"
     "  expect(user.id).toBe(1);\n});\n\n"
     "it('handles 404', async () => {\n"
     "  mockFetch.mockResolvedValue({ status: 404, json: () => ({ error: 'not found' }) });\n"
     "  await expect(getUser(99)).rejects.toThrow('not found');\n});\n```"),
    # Good integration test
    ("Review this integration test:\n```ts\ndescribe('POST /api/agents (integration)', () => {\n"
     "  let server: Express; let token: string;\n"
     "  beforeAll(async () => {\n"
     "    server = await startTestServer();\n"
     "    token = await getTestToken({ role: 'admin', tenantId: 'test-tenant' });\n"
     "  });\n"
     "  afterAll(() => server.close());\n\n"
     "  it('creates agent and returns 201', async () => {\n"
     "    const res = await request(server).post('/api/agents')\n"
     "      .set('Authorization', `Bearer ${token}`)\n"
     "      .send({ name: 'test-agent', type: 'code_reviewer' });\n"
     "    expect(res.status).toBe(201);\n"
     "    expect(res.body.id).toMatch(/^[0-9a-f-]+$/);\n"
     "  });\n"
     "  it('rejects unauthenticated request', async () => {\n"
     "    const res = await request(server).post('/api/agents').send({ name: 'x' });\n"
     "    expect(res.status).toBe(401);\n"
     "  });\n});\n```"),
]

ADAPTER_CONFIG = {
    "cr01": {
        "system": CR01_SYSTEM,
        "seeds":  CODE_SEEDS,
        "output": "code_reviewer_dataset.jsonl",
        "label":  "LRA-CR01",
    },
    "dr01": {
        "system": DR01_SYSTEM,
        "seeds":  DOC_SEEDS,
        "output": "doc_reviewer_dataset.jsonl",
        "label":  "LRA-DR01",
    },
    "ar01": {
        "system": AR01_SYSTEM,
        "seeds":  ARCH_SEEDS,
        "output": "arch_reviewer_dataset.jsonl",
        "label":  "LRA-AR01",
    },
    "tr01": {
        "system": TR01_SYSTEM,
        "seeds":  TEST_SEEDS,
        "output": "test_reviewer_dataset.jsonl",
        "label":  "LRA-TR01",
    },
}


def call_llm(server: str, system: str, user: str, max_tokens: int = 512) -> str | None:
    """Call the local llama.cpp /v1/chat/completions endpoint."""
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    try:
        resp = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
        return None


def make_variations(seed: str, n: int = 3) -> list[str]:
    """Return the seed plus n-1 paraphrased/extended variants."""
    variants = [seed]
    # Simple variations: prepend different framing phrases
    prefixes = [
        "Please review the following:\n",
        "Conduct a thorough review of this:\n",
        "Provide a detailed code review for:\n",
        "Assess the quality of:\n",
        "Identify issues in:\n",
    ]
    for i in range(min(n - 1, len(prefixes))):
        variants.append(prefixes[i] + seed)
    return variants[:n]


def build_dataset(adapter_key: str, cfg: dict, args) -> int:
    """Generate and write training examples for one adapter. Returns count written."""
    output_path = Path(args.output_dir) / cfg["output"]
    print(f"\n{'='*60}")
    print(f"Building {cfg['label']} → {output_path}")
    print(f"{'='*60}")

    examples = []
    seeds = cfg["seeds"]
    random.shuffle(seeds)

    for i, seed in enumerate(seeds):
        # Generate 2-3 variations of each seed to bulk up the dataset
        variations = make_variations(seed, n=min(3, max(1, args.samples // len(seeds))))
        for v, user_msg in enumerate(variations):
            if len(examples) >= args.samples:
                break
            print(f"  [{i+1}/{len(seeds)} var {v+1}] generating...", end=" ", flush=True)
            response = call_llm(args.server, cfg["system"], user_msg, max_tokens=512)
            if response is None:
                print("SKIP")
                continue
            # Validate it looks like JSON (reviewers output JSON)
            try:
                json.loads(response)
                print(f"OK ({len(response)} chars)")
            except json.JSONDecodeError:
                # Model didn't produce valid JSON — still usable but warn
                print(f"non-JSON ({len(response)} chars)")
            examples.append({
                "messages": [
                    {"role": "system",    "content": cfg["system"]},
                    {"role": "user",      "content": user_msg.strip()},
                    {"role": "assistant", "content": response},
                ]
            })
            # Polite pause to avoid overwhelming the server
            time.sleep(0.3)
        if len(examples) >= args.samples:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n  Written: {len(examples)} examples → {output_path}")
    return len(examples)


def parse_args():
    p = argparse.ArgumentParser(description="Build Wave 1 reviewer adapter training datasets")
    p.add_argument("--adapter",    default="all",  choices=["all"] + list(ADAPTER_CONFIG))
    p.add_argument("--server",     default="http://localhost:8080")
    p.add_argument("--samples",    type=int, default=150, help="Target examples per adapter")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    return p.parse_args()


def main():
    args = parse_args()
    # Health check
    try:
        r = requests.get(f"{args.server}/health", timeout=5)
        r.raise_for_status()
        print(f"Server {args.server} is up: {r.json()}")
    except Exception as e:
        sys.exit(f"Server not reachable: {e}")

    adapters = list(ADAPTER_CONFIG) if args.adapter == "all" else [args.adapter]
    totals = {}
    for key in adapters:
        n = build_dataset(key, ADAPTER_CONFIG[key], args)
        totals[key] = n

    print("\n=== SUMMARY ===")
    for key, n in totals.items():
        status = "OK" if n >= 50 else "WARNING: low sample count"
        print(f"  {key}: {n} examples — {status}")

    print("\nNext step: train all adapters:")
    for key in adapters:
        print(f"  python3 train_lora_rocm.py --adapter {key} --model-path ~/.cache/huggingface/hub/qwen3-8b --output-dir output --cpu-mem 24GiB")


if __name__ == "__main__":
    main()

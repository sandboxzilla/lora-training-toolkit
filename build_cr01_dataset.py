#!/usr/bin/env python3
"""
build_cr01_dataset.py — LRA-CR01 Code Reviewer corpus builder (v2)

Design principle: NO AgentHub-specific rules, defect descriptions, or code
standards are hardcoded in training examples. All rule content is fetched from
DocCore at build time. The model learns to USE retrieved rules, not recall them.

Produces:
  - code_reviewer_dataset.jsonl
  - code_reviewer_manifest.md

Usage:
  python3 build_cr01_dataset.py [--repo-root PATH] [--retriever URL] [--offline]
"""

import os, re, json, random, textwrap, subprocess, argparse
from pathlib import Path
from datetime import datetime
from retriever_client import RetrieverClient, offline_note, make_system_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root",  type=Path,
                    default=None)
parser.add_argument("--retriever",  default="http://localhost:8082")
parser.add_argument("--tenant",     default="agenthub")
parser.add_argument("--top-k",      type=int, default=5)
parser.add_argument("--offline",    action="store_true")
args = parser.parse_args()
if args.repo_root is None:
    _p = Path(__file__).resolve()
    for _parent in _p.parents:
        if (_parent / "backend" / "src").exists():
            args.repo_root = _parent
            break
    if args.repo_root is None:
        parser.error("--repo-root required: cannot auto-detect repo root from " + str(_p))

REPO_ROOT    = args.repo_root
BACKEND_SRC  = REPO_ROOT / "backend" / "src"
SCRIPTS_DIR  = REPO_ROOT / "backend" / "scripts" / "ub02"
OUT_DIR      = Path(__file__).parent
DATASET_OUT  = OUT_DIR / "code_reviewer_dataset.jsonl"
MANIFEST_OUT = OUT_DIR / "code_reviewer_manifest.md"

MAX_CHUNK_LINES = 60
MIN_CHUNK_LINES = 10

SYSTEM = make_system_prompt(
    role="Code Reviewer",
    review_subject=(
        "TypeScript and JavaScript source code for AgentHub-specific defects. "
        "Query DocCore for the current rules on tenant isolation, JWT handling, "
        "shell execution safety, MQTT topic format, and API contract requirements "
        "before identifying violations."
    ),
    output_format=(
        "Produce BLK (blocking) and NBK (non-blocking) findings with exact line references. "
        "Each finding must cite the retrieved rule it violates. "
        "End with: APPROVED | APPROVED_WITH_NOTES | BLOCKED"
    )
)

# ── Baseline queries for CR01 ─────────────────────────────────────────────────
# These are fetched once per build run and injected into all training examples.
# The content comes from DocCore — never hardcoded here.
CR01_QUERIES = [
    {"purpose": "tenant_isolation",
     "query": "tenant_id scoping database queries JWT claims multi-tenant isolation rules"},
    {"purpose": "jwt_auth",
     "query": "JWT authentication middleware req.user claims token decode security rules"},
    {"purpose": "shell_safety",
     "query": "execFile exec shell execution safety allowlist command injection prevention"},
    {"purpose": "mqtt_topics",
     "query": "MQTT topic format convention agenthub starbush fleet topic band documentation"},
    {"purpose": "api_contract",
     "query": "API route contract auth middleware HTTP method REST endpoint requirements"},
]

# ── Pattern detectors (builder-only — NOT embedded in training examples) ──────
# These detect issues in source files to identify training signal.
# They do not appear in the training data as rules.
TENANT_OK_RE    = re.compile(r"tenant_id\s*[=:]|AND\s+tenant_id|WHERE.*tenant_id")
DB_QUERY_RE     = re.compile(r"(?:SELECT|INSERT|UPDATE|DELETE|getConnection|pool\s*\.)",
                              re.IGNORECASE)
JWT_CLAIM_OK_RE = re.compile(r"req\.(user|claims|tenant)\.(tenant_id|userId|sub)")
JWT_RAW_RE      = re.compile(r"req\.headers\[.authorization.\]|jwt\.decode\(")
SHELL_SAFE_RE   = re.compile(r"execFile|spawnSync.*\[|allowlist", re.IGNORECASE)
SHELL_UNSAFE_RE = re.compile(r"exec\s*\(.*\$\{|child_process.*exec\s*\(`", re.IGNORECASE)
MQTT_FORMAT_OK  = re.compile(r"starbush/fleet/[^/`'\"]+/(?:heartbeat|telemetry)|agenthub/fleet/cmd/")
MQTT_TOPIC_RE   = re.compile(r"['\"`]starbush/|['\"`]agenthub/")

# ── Issue classifiers (return issue type, not rule text) ─────────────────────

def classify_issues(chunk: str) -> list[str]:
    """Return list of issue type strings (not rule text)."""
    issues = []
    if DB_QUERY_RE.search(chunk) and not TENANT_OK_RE.search(chunk):
        if "async " in chunk or "function " in chunk or "=>" in chunk:
            issues.append("missing_tenant_isolation")
    if JWT_RAW_RE.search(chunk) and not JWT_CLAIM_OK_RE.search(chunk):
        issues.append("raw_jwt_decode")
    if SHELL_UNSAFE_RE.search(chunk):
        issues.append("unsafe_shell_exec")
    if MQTT_TOPIC_RE.search(chunk) and not MQTT_FORMAT_OK.search(chunk):
        issues.append("mqtt_topic_format")
    return issues


def issues_to_finding(issues: list[str]) -> str:
    """
    Convert issue types to finding text that REFERENCES retrieved context.
    No specific rule text is embedded — the model learns to say
    "per the rules above" rather than reciting the rule.
    """
    finding_parts = []
    for issue in issues:
        if issue == "missing_tenant_isolation":
            finding_parts.append(
                "BLK: Database operation without tenant isolation. "
                "Per the tenant isolation rules retrieved above, this pattern violates "
                "the multi-tenant data boundary requirement."
            )
        elif issue == "raw_jwt_decode":
            finding_parts.append(
                "BLK: Raw JWT decode bypassing auth middleware. "
                "Per the JWT authentication rules retrieved above, tokens must be "
                "processed through the established middleware, not re-decoded inline."
            )
        elif issue == "unsafe_shell_exec":
            finding_parts.append(
                "BLK: Unsafe shell execution pattern detected. "
                "Per the shell safety rules retrieved above, this pattern is prohibited."
            )
        elif issue == "mqtt_topic_format":
            finding_parts.append(
                "NBK: MQTT topic string does not match the documented format. "
                "Per the MQTT topic conventions retrieved above, verify this topic "
                "conforms to the registered band specification."
            )
    return "\n\n".join(finding_parts)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def chunk_code(content: str) -> list[tuple[str, int]]:
    lines = content.splitlines()
    chunks = []
    step = MAX_CHUNK_LINES // 2
    for start in range(0, len(lines), step):
        chunk = lines[start:start + MAX_CHUNK_LINES]
        if len(chunk) < MIN_CHUNK_LINES:
            break
        chunks.append(("\n".join(chunk), start + 1))
    return chunks


def pair(user_msg: str, assistant_msg: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": user_msg.strip()},
        {"role": "assistant", "content": assistant_msg.strip()},
    ]}


def format_user(rel: str, start: int, chunk: str, context: str) -> str:
    return (
        f"Review the following code from `{rel}` (lines {start}–"
        f"{start + chunk.count(chr(10))}) for AgentHub-specific defects.\n\n"
        f"### Governance Context\n{context}\n\n"
        f"```typescript\n{chunk}\n```"
    )

# ── Git diff pairs ─────────────────────────────────────────────────────────────

def get_git_diffs(limit: int = 40) -> list[tuple[str, str, str]]:
    try:
        log_out = subprocess.check_output(
            ["git", "log", "--oneline", f"-{limit}", "--", "backend/src", "backend/scripts"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    result = []
    for line in log_out.strip().splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        commit_hash, commit_msg = parts
        try:
            diff = subprocess.check_output(
                ["git", "show", "--stat", "--unified=5", commit_hash,
                 "--", "backend/src", "backend/scripts/ub02/retriever-service.js"],
                cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            )
            diff_lines = [l for l in diff.splitlines()
                          if l.startswith("+") or l.startswith("@@") or l.startswith("diff")]
            diff_text = "\n".join(diff_lines[:80])
            if len(diff_text) > 200:
                result.append((commit_hash, diff_text, commit_msg))
        except Exception:
            continue
    return result

# ── Defect injectors (inject issue type, finding references context) ──────────

INJECTIONS = [
    (
        lambda c: re.sub(r"\s*AND\s+tenant_id\s*=\s*[^\s,)]+", "", c, count=1),
        "missing_tenant_isolation",
    ),
    (
        lambda c: re.sub(r"execFile\(([^,]+),\s*\[([^\]]+)\]", r"exec(`\1 \2`", c, count=1),
        "unsafe_shell_exec",
    ),
    (
        lambda c: c.replace("req.user.tenant_id",
                             "jwt.decode(req.headers['authorization']).tenant_id"),
        "raw_jwt_decode",
    ),
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    client = RetrieverClient(args.retriever, args.tenant, args.top_k)

    if client.offline:
        print(offline_note(args.retriever))

    print("Fetching governance context from DocCore...")
    context = client.fetch(CR01_QUERIES)
    print(f"  Context length: {len(context)} chars")

    all_pairs: list[dict] = []
    clean_chunks: list[tuple[str, str, int]] = []

    source_files = (
        list(BACKEND_SRC.rglob("*.ts")) +
        [SCRIPTS_DIR / "retriever-service.js"]
    )
    print(f"Processing {len(source_files)} source files...")

    for src_file in source_files:
        content = read_file(src_file)
        if not content:
            continue
        rel = str(src_file.relative_to(REPO_ROOT))
        for chunk, start in chunk_code(content):
            issues = classify_issues(chunk)
            if issues:
                finding = issues_to_finding(issues)
                has_blk = any(t in ["missing_tenant_isolation", "raw_jwt_decode", "unsafe_shell_exec"]
                              for t in issues)
                verdict = "BLOCKED" if has_blk else "APPROVED_WITH_NOTES"
                all_pairs.append(pair(
                    format_user(rel, start, chunk, context),
                    f"{finding}\n\nVerdict: {verdict}"
                ))
            else:
                clean_chunks.append((chunk, rel, start))
                # Add clean approval pairs (model learns what compliant code looks like)
                if random.random() < 0.3:
                    all_pairs.append(pair(
                        format_user(rel, start, chunk, context),
                        "No violations detected against the retrieved rules.\n\nVerdict: APPROVED"
                    ))

    print(f"  Issue pairs: {len(all_pairs)}")

    # Git diff pairs
    diffs = get_git_diffs(limit=50)
    for commit_hash, diff_text, commit_msg in diffs:
        additions = "\n".join(l[1:] for l in diff_text.splitlines() if l.startswith("+"))
        issues = classify_issues(additions)
        finding = issues_to_finding(issues) if issues else "No violations detected."
        verdict = "BLOCKED" if any("BLK" in f for f in finding.split("\n")) else "APPROVED"
        all_pairs.append(pair(
            f"Review this git diff from commit `{commit_hash[:8]}` "
            f"('{commit_msg}') for AgentHub code quality.\n\n"
            f"### Governance Context\n{context}\n\n"
            f"```diff\n{diff_text}\n```",
            f"{finding}\n\nVerdict: {verdict}"
        ))
    print(f"  Git diff pairs: {len(diffs)}")

    # Defect injection
    inject_target = max(30, len(all_pairs) // 3)
    injected = 0
    for _ in range(inject_target * 3):
        if not clean_chunks or injected >= inject_target:
            break
        chunk, rel, start = random.choice(clean_chunks)
        mutator, issue_type = random.choice(INJECTIONS)
        mutated = mutator(chunk)
        if mutated != chunk:
            finding = issues_to_finding([issue_type])
            all_pairs.append(pair(
                format_user(rel, start, mutated, context),
                f"{finding}\n\nVerdict: BLOCKED"
            ))
            injected += 1
    print(f"  Injected pairs: {injected}")

    random.shuffle(all_pairs)

    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = textwrap.dedent(f"""\
        # LRA-CR01 Code Reviewer — Training Dataset Manifest (v2)

        **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%MZ')}
        **Retriever**: {'LIVE — ' + args.retriever if not client.offline else 'OFFLINE — placeholder context'}
        **Total pairs**: {len(all_pairs)}

        ## Design
        No AgentHub-specific rules are hardcoded in training examples.
        All governance context is fetched from DocCore at build time.
        Findings reference "retrieved rules above" — not embedded rule text.
        Rebuild dataset when rules change; no code changes needed.

        ## Statistics
        | Category | Count |
        |:---------|------:|
        | Source annotation + clean pairs | {len(all_pairs) - len(diffs) - injected} |
        | Git diff pairs | {len(diffs)} |
        | Defect injection pairs | {injected} |
        | **Total** | **{len(all_pairs)}** |
    """)
    MANIFEST_OUT.write_text(manifest, encoding="utf-8")
    print(f"Total pairs: {len(all_pairs)} | Output: {DATASET_OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_ar01_dataset.py — LRA-AR01 Architecture Reviewer corpus builder (v2)

Design principle: NO AgentHub-specific constraints (DBSS rules, service placement
specs, tenant isolation requirements, MQTT topology) are hardcoded in training
examples. All constraint content is fetched from DocCore at build time.

Produces:
  - arch_reviewer_dataset.jsonl
  - arch_reviewer_manifest.md

Usage:
  python3 build_ar01_dataset.py [--repo-root PATH] [--retriever URL] [--offline]
"""

import os, re, json, random, textwrap, argparse
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
DOCS_DIR     = REPO_ROOT / "docs"
OUT_DIR      = Path(__file__).parent
DATASET_OUT  = OUT_DIR / "arch_reviewer_dataset.jsonl"
MANIFEST_OUT = OUT_DIR / "arch_reviewer_manifest.md"

ARCH_DOC_RE = re.compile(
    r"architecture|proposal|high.level.design|low.level.design|hld|lld",
    re.IGNORECASE
)

SYSTEM = make_system_prompt(
    role="Architecture Reviewer",
    review_subject=(
        "Architecture proposals, HLDs, and design documents for AgentHub constraint violations. "
        "Query DocCore for the current architectural constraints (service boundaries, data isolation "
        "requirements, infrastructure placement rules, communication protocol rules) before "
        "identifying violations."
    ),
    output_format=(
        "Produce BLK (blocking) and NBK (non-blocking) findings. "
        "Each finding must cite the specific constraint retrieved from DocCore that is violated. "
        "End with: APPROVED | CONDITIONALLY_APPROVED | BLOCKED"
    )
)

# ── Baseline queries for AR01 ─────────────────────────────────────────────────
AR01_QUERIES = [
    {"purpose": "data_boundary_rules",
     "query": "DBSS database access rules agents direct query MariaDB sentry boundary"},
    {"purpose": "tenant_isolation",
     "query": "tenant isolation architecture all data paths scope by tenant_id JWT"},
    {"purpose": "service_placement",
     "query": "ub01 ub02 service placement inference control plane infrastructure topology"},
    {"purpose": "api_first",
     "query": "API-first architecture REST MQTT service communication no cross-service database"},
    {"purpose": "mqtt_topology",
     "query": "MQTT topic band documentation starbush agenthub fleet cmd registered topics"},
    {"purpose": "agent_sandbox",
     "query": "Worker Bee Podman container sandbox host execution isolation rules"},
]

# ── Issue classifiers (detect violation TYPE, no rule text) ──────────────────
ISSUE_SIGNALS = [
    ("direct_db_access",
     re.compile(r"agents.*directly.*access.*database|agent.*query.*mariadb|bee.*direct.*db",
                re.IGNORECASE)),
    ("cross_service_db",
     re.compile(r"service.*share.*database|two.*services.*same.*db|microservice.*direct.*mariadb",
                re.IGNORECASE)),
    ("wrong_inference_host",
     re.compile(r"inference.*ub01|llama.*ub01|qwen.*ub01|model.*control.plane",
                re.IGNORECASE)),
    ("missing_tenant_scope",
     re.compile(r"without.*tenant_id|no.*tenant.*scope|skip.*tenant.*filter",
                re.IGNORECASE)),
    ("undocumented_mqtt",
     re.compile(r"new.*mqtt.*topic|custom.*topic|ad.hoc.*topic", re.IGNORECASE)),
    ("sandbox_escape",
     re.compile(r"host.*exec.*bee|shell.*worker.bee|podman.*exec.*outside",
                re.IGNORECASE)),
]

POSITIVE_SIGNALS = [
    ("correct_api_layer",
     re.compile(r"api.*layer|rest.*endpoint|through.*api|via.*api", re.IGNORECASE)),
    ("correct_tenant_jwt",
     re.compile(r"tenant_id.*jwt|jwt.*tenant_id|claims.*tenant", re.IGNORECASE)),
    ("correct_inference_host",
     re.compile(r"ub02.*inference|inference.*ub02|model.*ub02", re.IGNORECASE)),
]

BLOCKING_ISSUES = {"direct_db_access", "cross_service_db", "wrong_inference_host",
                   "missing_tenant_scope", "sandbox_escape"}

def classify_issues(text: str) -> tuple[list[str], list[str]]:
    text_lower = text.lower()
    violations = [t for t, r in ISSUE_SIGNALS if r.search(text_lower)]
    positives  = [t for t, r in POSITIVE_SIGNALS if r.search(text_lower)]
    return violations, positives


def issues_to_finding(violations: list[str], positives: list[str]) -> str:
    parts = []
    for v in violations:
        severity = "BLK" if v in BLOCKING_ISSUES else "NBK"
        parts.append(
            f"{severity}: Architectural violation — {v.replace('_', ' ')}. "
            f"Per the {v.replace('_', ' ')} constraint retrieved above, "
            f"this design pattern is not permitted."
        )
    for p in positives:
        parts.append(
            f"VT: Design correctly applies {p.replace('_', ' ')} pattern "
            f"per the retrieved constraint."
        )
    return "\n\n".join(parts)

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def extract_abstract(content: str, max_chars: int = 1500) -> str:
    for heading in ["Executive Summary", "Overview", "Introduction", "Summary", "Proposal"]:
        m = re.search(rf"##\s+\d*\.?\s*{heading}(.*?)(?=\n##|\Z)",
                      content, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0)[:max_chars]
    lines = content.splitlines()
    body_start = next((i for i, l in enumerate(lines) if l.strip() == "---"), 0) + 1
    return "\n".join(lines[body_start:body_start + 40])[:max_chars]


def find_paired_review(doc_path: Path) -> str | None:
    reports_dir = doc_path.parent / ".reports"
    if not reports_dir.exists():
        return None
    stem = doc_path.stem
    for review_file in reports_dir.glob("*.md"):
        if stem[:8] in review_file.stem or stem[:12] in review_file.stem:
            content = read_md(review_file)
            if content:
                return content
    return None


def extract_review_findings(review_content: str) -> str:
    findings = []
    for m in re.finditer(
        r"\*\*(BLK|NBK)-\d+.*?\*\*.*?(?=\n\*\*(?:BLK|NBK)|\n##|\Z)",
        review_content, re.DOTALL
    ):
        findings.append(m.group(0).strip()[:500])
    verdict_m = re.search(
        r"(?:Verdict|##\s+\d*\.?\s*4\.?\s*Verdict)(.*?)(?=\n##|\Z)",
        review_content, re.IGNORECASE | re.DOTALL
    )
    verdict = verdict_m.group(0).strip()[:300] if verdict_m else ""
    return "\n\n".join(findings) + ("\n\n" + verdict if verdict else "")


def pair(user_msg: str, assistant_msg: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": user_msg.strip()},
        {"role": "assistant", "content": assistant_msg.strip()},
    ]}

# ── Synthetic injection descriptions (issue type only, no rule text) ──────────
SYNTHETIC_INJECTIONS = [
    ("The agent will directly query MariaDB on ub01 to retrieve task state.",
     ["direct_db_access"]),
    ("The new inference endpoint will run on ub01 alongside the API server.",
     ["wrong_inference_host"]),
    ("Data is returned without filtering by tenant — each client filters client-side.",
     ["missing_tenant_scope"]),
    ("Services will share a common database connection pool for efficiency.",
     ["cross_service_db"]),
    ("Worker Bees will shell-exec on the host to access cluster resources.",
     ["sandbox_escape"]),
    ("The new ML service will publish to a custom agenthub/ml/results topic.",
     ["undocumented_mqtt"]),
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    client = RetrieverClient(args.retriever, args.tenant, args.top_k)
    if client.offline:
        print(offline_note(args.retriever))

    print("Fetching governance context from DocCore...")
    context = client.fetch(AR01_QUERIES)
    print(f"  Context length: {len(context)} chars")

    all_pairs: list[dict] = []
    base_texts: list[str] = []
    skip_dirs = {".reports", ".plans", ".arc", ".backup", ".architect"}

    for dirpath, dirnames, filenames in os.walk(DOCS_DIR):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fname in filenames:
            if not fname.endswith(".md"):
                continue
            fpath = Path(dirpath) / fname
            if not ARCH_DOC_RE.search(fpath.stem):
                continue
            content = read_md(fpath)
            if not content or len(content) < 500:
                continue
            abstract = extract_abstract(content)
            if not abstract:
                continue

            rel = str(fpath.relative_to(REPO_ROOT))
            user_msg = (
                f"Review the following architecture proposal from `{rel}` "
                f"for AgentHub constraint compliance.\n\n"
                f"### Governance Context\n{context}\n\n"
                f"{abstract}"
            )

            paired_review = find_paired_review(fpath)
            if paired_review:
                findings = extract_review_findings(paired_review)
                if findings:
                    all_pairs.append(pair(user_msg, findings))
                    base_texts.append(abstract)
                    continue

            violations, positives = classify_issues(content)
            if violations or positives:
                finding = issues_to_finding(violations, positives)
                is_blocked = any(v in BLOCKING_ISSUES for v in violations)
                verdict = "BLOCKED" if is_blocked else ("CONDITIONALLY_APPROVED" if violations else "APPROVED")
                all_pairs.append(pair(user_msg, f"{finding}\n\nVerdict: {verdict}"))
            else:
                all_pairs.append(pair(
                    user_msg,
                    "No constraint violations detected against the retrieved rules.\n\nVerdict: APPROVED"
                ))
            base_texts.append(abstract)

    print(f"  Real proposal pairs: {len(all_pairs)}")

    # Synthetic injection pairs
    inject_count = max(20, len(all_pairs) // 2)
    injected = 0
    for _ in range(inject_count):
        base = random.choice(base_texts) if base_texts else "The system proposes a new service."
        injection_text, issue_types = random.choice(SYNTHETIC_INJECTIONS)
        finding = issues_to_finding(issue_types, [])
        all_pairs.append(pair(
            f"Review the following architecture description for AgentHub constraint compliance.\n\n"
            f"### Governance Context\n{context}\n\n"
            f"{base[:800]}\n\nAdditional design note: {injection_text}",
            f"{finding}\n\nVerdict: BLOCKED"
        ))
        injected += 1

    random.shuffle(all_pairs)

    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = textwrap.dedent(f"""\
        # LRA-AR01 Architecture Reviewer — Training Dataset Manifest (v2)

        **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%MZ')}
        **Retriever**: {'LIVE — ' + args.retriever if not client.offline else 'OFFLINE — placeholder context'}
        **Total pairs**: {len(all_pairs)}

        ## Design
        No AgentHub architectural constraints are hardcoded in training examples.
        All constraint content is fetched from DocCore at build time.
        Findings cite "retrieved constraint above" — not embedded rule text.
        Rebuild dataset when architecture rules change; no code changes needed.

        ## Statistics
        | Category | Count |
        |:---------|------:|
        | Real proposal pairs (with/without paired review) | {len(all_pairs) - injected} |
        | Synthetic injection pairs | {injected} |
        | **Total** | **{len(all_pairs)}** |
    """)
    MANIFEST_OUT.write_text(manifest, encoding="utf-8")
    print(f"Total pairs: {len(all_pairs)} | Output: {DATASET_OUT}")


if __name__ == "__main__":
    main()

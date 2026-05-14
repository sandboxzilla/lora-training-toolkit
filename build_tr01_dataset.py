#!/usr/bin/env python3
"""
build_tr01_dataset.py — LRA-TR01 Test Reviewer corpus builder (v3)

Design principle: NO AgentHub test infrastructure specifics (jest.config.js settings,
mock patterns, resetMocks behavior) are hardcoded in training examples. All test
infrastructure rules are fetched from DocCore at build time.

v3 additions:
  - Paired source+test examples: includes source file alongside test to assess coverage
  - Git diff pairs for test file changes
  - Larger code snippets (3000 → 4000 chars) to exploit seq_len=4096

Produces:
  - test_reviewer_dataset.jsonl
  - test_reviewer_manifest.md

Usage:
  python3 build_tr01_dataset.py [--repo-root PATH] [--retriever URL] [--offline]
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
TESTS_DIR    = REPO_ROOT / "backend" / "__tests__"
SRC_DIR      = REPO_ROOT / "backend" / "src"
OUT_DIR      = Path(__file__).parent
DATASET_OUT  = OUT_DIR / "test_reviewer_dataset.jsonl"
MANIFEST_OUT = OUT_DIR / "test_reviewer_manifest.md"

MAX_CHUNK_LINES = 70
MIN_CHUNK_LINES = 10
MAX_CODE_CHARS  = 4000   # raised from 3000 — seq_len=4096 allows this

SYSTEM = make_system_prompt(
    role="Test Reviewer",
    review_subject=(
        "Jest test files for coverage quality and AgentHub-specific test infrastructure compliance. "
        "Query DocCore for the current test infrastructure rules (jest config settings, mock patterns, "
        "required coverage standards) before identifying defects."
    ),
    output_format=(
        "Produce BLK (blocking) and NBK (non-blocking) findings. "
        "Each finding must cite the test infrastructure rule retrieved from DocCore that is violated. "
        "End with: ADEQUATE | INADEQUATE | BLOCKED"
    )
)

# ── Baseline queries for TR01 ─────────────────────────────────────────────────
TR01_QUERIES = [
    {"purpose": "jest_config",
     "query": "jest.config.js resetMocks setupFilesAfterEnv test infrastructure settings AgentHub"},
    {"purpose": "mock_patterns",
     "query": "pool mock pattern mockPool getConnection beforeEach restoration resetMocks true"},
    {"purpose": "mqtt_mock",
     "query": "MQTT mock mqtt.connect implementation restoration beforeEach test pattern"},
    {"purpose": "coverage_standards",
     "query": "test coverage negative path error path requirements acceptance criteria"},
    {"purpose": "timer_patterns",
     "query": "jest fake timers useFakeTimers MQTT callback advanceTimersByTime runAllTimers"},
]

# ── Issue classifiers (detect TYPE, no rule text) ────────────────────────────

def has_negative_paths(content: str) -> bool:
    return bool(re.search(
        r"rejects|throws|\.mockRejected|error|Error|catch|toThrow|toBe.*false|"
        r"status.*[45]\d\d|expect.*null|expect.*undefined",
        content, re.IGNORECASE
    ))


def has_mock_restoration(content: str) -> bool:
    return bool(re.search(
        r"beforeEach.*mockResolvedValue|beforeEach.*mockImplementation|"
        r"mockPool.*getConnection.*mockResolvedValue|mqtt\.connect.*mockImplementation",
        content, re.DOTALL
    ))


def has_assertions(content: str) -> bool:
    return bool(re.search(r"expect\s*\(", content))


def has_resetModules_misuse(content: str) -> bool:
    return bool(re.search(r"beforeEach.*jest\.resetModules|jest\.resetModules.*beforeEach",
                           content, re.DOTALL | re.IGNORECASE))


def has_fake_timer_mqtt(content: str) -> bool:
    return bool(re.search(r"useFakeTimers", content, re.IGNORECASE)) and \
           bool(re.search(r"mqtt", content, re.IGNORECASE))


def needs_pool_mock(content: str) -> bool:
    return bool(re.search(r"pool|getConnection|db_query|MariaDB", content, re.IGNORECASE))


def classify_issues(content: str) -> list[str]:
    issues = []
    if needs_pool_mock(content) and not has_mock_restoration(content):
        issues.append("missing_mock_restoration")
    if not has_negative_paths(content) and len(content.splitlines()) > 30:
        issues.append("happy_path_bias")
    if has_resetModules_misuse(content):
        issues.append("resetModules_misuse")
    if has_fake_timer_mqtt(content):
        issues.append("fake_timer_mqtt_loop")
    if not has_assertions(content[:2000]):
        issues.append("no_assertions")
    return issues


def issues_to_finding(issues: list[str]) -> str:
    BLOCKING = {"missing_mock_restoration", "resetModules_misuse", "no_assertions"}
    parts = []
    for issue in issues:
        severity = "BLK" if issue in BLOCKING else "NBK"
        parts.append(
            f"{severity}: Test infrastructure defect — {issue.replace('_', ' ')}. "
            f"Per the test infrastructure rules retrieved above, this pattern is non-compliant."
        )
    return "\n\n".join(parts)


def pair(user_msg: str, assistant_msg: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": user_msg.strip()},
        {"role": "assistant", "content": assistant_msg.strip()},
    ]}


def chunk_test(content: str) -> list[tuple[str, int]]:
    lines = content.splitlines()
    chunks = []
    current: list[str] = []
    start = 1
    for i, line in enumerate(lines, 1):
        current.append(line)
        if len(current) >= MAX_CHUNK_LINES:
            chunks.append(("\n".join(current), start))
            start = i + 1
            current = []
    if len(current) >= MIN_CHUNK_LINES:
        chunks.append(("\n".join(current), start))
    return chunks

# ── Source file finder ────────────────────────────────────────────────────────

def find_source_for_test(test_path: Path) -> Path | None:
    """
    Given a test file path, attempt to find the corresponding source file.
    e.g. __tests__/unit/auth.test.ts → src/api/middleware/auth.ts
    """
    stem = re.sub(r"\.test$", "", test_path.stem)
    # Walk backend/src looking for a file whose stem matches
    for src_file in SRC_DIR.rglob("*.ts"):
        if src_file.stem == stem:
            return src_file
    return None


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

# ── Git diff pairs for test files ────────────────────────────────────────────

def get_git_test_diffs(limit: int = 40) -> list[tuple[str, str, str]]:
    """
    Pull recent git commits that touched test files.
    Returns list of (commit_hash, diff_text, commit_msg).
    """
    try:
        log_out = subprocess.check_output(
            ["git", "log", "--oneline", f"-{limit}", "--", "backend/__tests__"],
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
                 "--", "backend/__tests__"],
                cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
            )
            diff_lines = [l for l in diff.splitlines()
                          if l.startswith("+") or l.startswith("@@") or l.startswith("diff")]
            diff_text = "\n".join(diff_lines[:100])
            if len(diff_text) > 200:
                result.append((commit_hash, diff_text, commit_msg))
        except Exception:
            continue
    return result

# ── Defect injectors (inject issue type only) ─────────────────────────────────

INJECTIONS = [
    (
        lambda c: re.sub(
            r"beforeEach\s*\(\s*\(\s*\)\s*=>\s*\{[^}]{0,200}mockPool\.getConnection[^}]{0,200}\}",
            "beforeEach(() => {\n  // mock not restored\n});",
            c, count=1, flags=re.DOTALL
        ),
        "missing_mock_restoration",
    ),
    (
        lambda c: c.replace("jest.resetAllMocks()", "jest.resetModules()"),
        "resetModules_misuse",
    ),
    (
        lambda c: re.sub(r"^\s*expect\(.*;\s*$", "  // assertion removed",
                          c, flags=re.MULTILINE),
        "no_assertions",
    ),
]

# ── Scenario pairs (abstract descriptions, context-referenced findings) ───────
SCENARIO_PAIRS = [
    (
        "The test suite uses jest.resetModules() in beforeEach for each test. Is this correct?",
        "missing_mock_restoration",
    ),
    (
        "A test for the orchestrator uses jest.useFakeTimers() with MQTT callbacks. "
        "The test completes but always takes 5 seconds. Is this adequate?",
        "fake_timer_mqtt_loop",
    ),
    (
        "All tests test the success path only. There are no error or rejection test cases. "
        "Test coverage report shows 90% line coverage. Is this adequate?",
        "happy_path_bias",
    ),
    (
        "The pool mock is set up in beforeAll but not restored in beforeEach. "
        "jest.config.js has resetMocks: true. Will this work?",
        "missing_mock_restoration",
    ),
]

# ── Coverage assessment helpers ───────────────────────────────────────────────

def assess_paired_coverage(src_content: str, test_content: str) -> str:
    """
    Given source and test code, produce a coverage assessment finding.
    Returns finding text referencing retrieved rules, not hardcoded standards.
    """
    issues = []
    src_functions = re.findall(
        r"(?:async\s+)?function\s+\w+|(?:const|let)\s+\w+\s*=\s*(?:async\s+)?\(",
        src_content
    )
    src_fn_count = len(src_functions)

    described_blocks = len(re.findall(r"\bdescribe\s*\(", test_content))
    it_blocks = len(re.findall(r"\b(?:it|test)\s*\(", test_content))
    expect_calls = len(re.findall(r"\bexpect\s*\(", test_content))

    if expect_calls == 0:
        issues.append("no_assertions")
    if not has_negative_paths(test_content) and src_fn_count > 2:
        issues.append("happy_path_bias")
    if needs_pool_mock(src_content) and not has_mock_restoration(test_content):
        issues.append("missing_mock_restoration")

    if issues:
        return issues_to_finding(issues)

    coverage_summary = (
        f"Test suite has {it_blocks} test case(s) across {described_blocks} describe block(s) "
        f"with {expect_calls} assertion(s) covering {src_fn_count} source function(s). "
        f"Coverage assessment: per the coverage standards retrieved above, this appears adequate."
    )
    return coverage_summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    client = RetrieverClient(args.retriever, args.tenant, args.top_k)
    if client.offline:
        print(offline_note(args.retriever))

    print("Fetching governance context from DocCore...")
    context = client.fetch(TR01_QUERIES)
    print(f"  Context length: {len(context)} chars")

    all_pairs: list[dict] = []
    clean_chunks: list[str] = []
    paired_count = 0

    test_files = (
        list((TESTS_DIR / "unit").glob("*.ts")) +
        list((TESTS_DIR / "integration").glob("*.ts")) +
        list(TESTS_DIR.glob("*.test.ts"))
    ) if TESTS_DIR.exists() else []

    print(f"Processing {len(test_files)} test files...")

    for test_file in test_files:
        try:
            content = test_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(test_file.relative_to(REPO_ROOT))

        issues = classify_issues(content)
        finding = issues_to_finding(issues) if issues else "No test infrastructure defects detected."
        verdict = ("BLOCKED" if any(i in {"missing_mock_restoration","resetModules_misuse","no_assertions"}
                                    for i in issues)
                   else ("INADEQUATE" if issues else "ADEQUATE"))

        # ── Standard test-only review ────────────────────────────────────────
        all_pairs.append(pair(
            f"Review the test file `{rel}` for AgentHub test quality and "
            f"infrastructure compliance.\n\n"
            f"### Test Infrastructure Rules\n{context}\n\n"
            f"```typescript\n{content[:MAX_CODE_CHARS]}\n```",
            f"{finding}\n\nVerdict: {verdict}"
        ))

        # ── Paired source+test review ─────────────────────────────────────────
        src_file = find_source_for_test(test_file)
        if src_file:
            src_content = read_file(src_file)
            if src_content:
                src_rel = str(src_file.relative_to(REPO_ROOT))
                coverage_finding = assess_paired_coverage(src_content, content)
                coverage_issues = classify_issues(content)
                coverage_verdict = (
                    "BLOCKED" if any(i in {"missing_mock_restoration","resetModules_misuse","no_assertions"}
                                     for i in coverage_issues)
                    else ("INADEQUATE" if coverage_issues else "ADEQUATE")
                )
                all_pairs.append(pair(
                    f"Review the test file `{rel}` for coverage of `{src_rel}`. "
                    f"Assess whether the tests adequately cover the source implementation.\n\n"
                    f"### Test Infrastructure Rules\n{context}\n\n"
                    f"**Source file (`{src_rel}`):**\n"
                    f"```typescript\n{src_content[:2000]}\n```\n\n"
                    f"**Test file (`{rel}`):**\n"
                    f"```typescript\n{content[:2000]}\n```",
                    f"{coverage_finding}\n\nVerdict: {coverage_verdict}"
                ))
                paired_count += 1

        # ── Chunk-level review ───────────────────────────────────────────────
        for chunk, start in chunk_test(content):
            chunk_issues = classify_issues(chunk)
            if not chunk_issues:
                clean_chunks.append(chunk)
            else:
                chunk_finding = issues_to_finding(chunk_issues)
                blk = any(i in {"missing_mock_restoration","resetModules_misuse","no_assertions"}
                          for i in chunk_issues)
                all_pairs.append(pair(
                    f"Review this test section from `{rel}` (starting line {start}) "
                    f"for test quality defects.\n\n"
                    f"### Test Infrastructure Rules\n{context}\n\n"
                    f"```typescript\n{chunk}\n```",
                    f"{chunk_finding}\n\nVerdict: {'BLOCKED' if blk else 'INADEQUATE'}"
                ))

    print(f"  Test file pairs: {len(all_pairs)} ({paired_count} paired source+test)")

    # ── Git diff pairs ────────────────────────────────────────────────────────
    diffs = get_git_test_diffs(limit=50)
    for commit_hash, diff_text, commit_msg in diffs:
        additions = "\n".join(l[1:] for l in diff_text.splitlines() if l.startswith("+"))
        issues = classify_issues(additions)
        finding = issues_to_finding(issues) if issues else "No test infrastructure defects detected."
        blk = any(i in {"missing_mock_restoration","resetModules_misuse","no_assertions"} for i in issues)
        verdict = "BLOCKED" if blk else ("INADEQUATE" if issues else "ADEQUATE")
        all_pairs.append(pair(
            f"Review this git diff from commit `{commit_hash[:8]}` "
            f"('{commit_msg}') for test quality and infrastructure compliance.\n\n"
            f"### Test Infrastructure Rules\n{context}\n\n"
            f"```diff\n{diff_text}\n```",
            f"{finding}\n\nVerdict: {verdict}"
        ))
    print(f"  Git diff pairs: {len(diffs)}")

    # ── Defect injection ──────────────────────────────────────────────────────
    inject_target = max(20, len(all_pairs) // 3)
    injected = 0
    for _ in range(inject_target * 3):
        if not clean_chunks or injected >= inject_target:
            break
        chunk = random.choice(clean_chunks)
        mutator, issue_type = random.choice(INJECTIONS)
        mutated = mutator(chunk)
        if mutated != chunk:
            finding = issues_to_finding([issue_type])
            all_pairs.append(pair(
                f"Review this test section for AgentHub test infrastructure defects.\n\n"
                f"### Test Infrastructure Rules\n{context}\n\n"
                f"```typescript\n{mutated[:2000]}\n```",
                f"{finding}\n\nVerdict: BLOCKED"
            ))
            injected += 1
    print(f"  Injected pairs: {injected}")

    # ── Scenario pairs ────────────────────────────────────────────────────────
    for scenario_text, issue_type in SCENARIO_PAIRS:
        finding = issues_to_finding([issue_type])
        blk = issue_type in {"missing_mock_restoration", "resetModules_misuse", "no_assertions"}
        all_pairs.append(pair(
            f"{scenario_text}\n\n### Test Infrastructure Rules\n{context}",
            f"{finding}\n\nVerdict: {'BLOCKED' if blk else 'INADEQUATE'}"
        ))

    random.shuffle(all_pairs)

    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    base_count = len(all_pairs) - injected - len(SCENARIO_PAIRS) - len(diffs)
    manifest = textwrap.dedent(f"""\
        # LRA-TR01 Test Reviewer — Training Dataset Manifest (v3)

        **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%MZ')}
        **Retriever**: {'LIVE — ' + args.retriever if not client.offline else 'OFFLINE — placeholder context'}
        **Total pairs**: {len(all_pairs)}

        ## Design
        No test infrastructure rules are hardcoded in training examples.
        All jest.config.js settings, mock patterns, and coverage standards
        are fetched from DocCore at build time.
        Findings cite "retrieved rules above" — not embedded specifics.

        v3: Added paired source+test examples and git diff pairs.
        TR01 now trains on actual TypeScript test code with full mock patterns,
        assertion structure, and paired source coverage assessment.

        ## Statistics
        | Category | Count |
        |:---------|------:|
        | Test file annotation pairs | {base_count} |
        | Paired source+test pairs | {paired_count} |
        | Git diff pairs | {len(diffs)} |
        | Defect injection pairs | {injected} |
        | Scenario pairs | {len(SCENARIO_PAIRS)} |
        | **Total** | **{len(all_pairs)}** |
    """)
    MANIFEST_OUT.write_text(manifest, encoding="utf-8")
    print(f"Total pairs: {len(all_pairs)} | Output: {DATASET_OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_dr01_dataset.py — LRA-DR01 Document Reviewer corpus builder

Produces:
  - doc_reviewer_dataset.jsonl  : ChatML pairs for document review training
  - doc_reviewer_manifest.md    : source stats and sample pairs

Strategy:
  1. Real pairs   — walk all .reports/ dirs, match review to reviewed doc, extract
                    (doc header + summary section, BLK/NBK findings + verdict)
  2. Synthetic    — inject known defects into clean headers and expect specific findings
                    Target: 60% real, 40% synthetic

Usage: python3 build_dr01_dataset.py [--repo-root PATH]
"""

import os, re, json, random, textwrap, argparse
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parents[4])
args = parser.parse_args()

REPO_ROOT    = args.repo_root
DOCS_DIR     = REPO_ROOT / "docs"
OUT_DIR      = Path(__file__).parent
DATASET_OUT  = OUT_DIR / "doc_reviewer_dataset.jsonl"
MANIFEST_OUT = OUT_DIR / "doc_reviewer_manifest.md"

SYSTEM = (
    "You are a Sentinel Architect for AgentHub. "
    "Your role is to review documents for AHC header compliance, version history coherence, "
    "status truthfulness, and register cross-reference discipline. "
    "Produce structured BLK (blocking) and NBK (non-blocking) findings. "
    "End every review with one of: APPROVED, CONDITIONALLY APPROVED, or BLOCKED."
)

# ── Required header fields ─────────────────────────────────────────────────────
REQUIRED_FIELDS = [
    "File", "Document Version", "Author", "Reviewer",
    "Program", "Project", "ID", "Status", "Role", "Class", "FPA", "PDC",
]

VALID_STATUSES   = {"ACTIVE", "DRAFT", "APPROVED", "PROPOSED", "RETIRED", "HOLD"}
VALID_FPA_VALUES = {"ILF", "EO", "EQ", "EI"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_reports_dirs(root: Path):
    for dirpath, dirnames, _ in os.walk(root):
        if ".reports" in dirnames:
            yield Path(dirpath) / ".reports"


def read_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def extract_header_block(content: str) -> str:
    """Extract the header table(s) at the top of a document (up to first ---+body)."""
    lines = content.splitlines()
    header_lines = []
    past_first_rule = False
    for line in lines[:80]:
        if line.strip() == "---":
            if past_first_rule:
                break
            past_first_rule = True
        header_lines.append(line)
    return "\n".join(header_lines)


def extract_version_history(content: str) -> str:
    """Extract the version history table."""
    m = re.search(
        r"##\s+(?:Version\s+)?History.*?\n((?:\|.*\n)+)",
        content, re.IGNORECASE
    )
    return m.group(0)[:800] if m else ""


def extract_findings(review_content: str) -> str:
    """Extract just the findings sections from a Sentinel review."""
    # Find BLK / NBK finding blocks
    findings = []
    for m in re.finditer(
        r"\*\*(BLK|NBK)-\d+.*?\*\*.*?(?=\n\*\*(?:BLK|NBK)|\n##|\Z)",
        review_content, re.DOTALL
    ):
        findings.append(m.group(0).strip()[:600])
    # Extract verdict
    verdict_m = re.search(
        r"(?:##\s+\d*\.?\s*Verdict|##\s+\d*\.?\s*4\.?\s*Verdict)(.*?)(?=\n##|\Z)",
        review_content, re.IGNORECASE | re.DOTALL
    )
    verdict = verdict_m.group(0).strip()[:400] if verdict_m else ""
    return "\n\n".join(findings) + ("\n\n" + verdict if verdict else "")


def find_reviewed_doc(report_path: Path, report_content: str) -> tuple[Path | None, str]:
    """Try to find the document that was reviewed."""
    # Look for 'Reviewed Document' field in the review
    m = re.search(r"Reviewed Document\s*\|?\s*([^\n|]+)", report_content)
    if m:
        candidate = m.group(1).strip().strip("`")
        # Try as relative to repo root
        full = REPO_ROOT / candidate
        if full.exists():
            return full, read_md(full)
    # Fallback: guess by stripping known review suffixes from filename
    stem = report_path.stem
    for suffix in ("_sentinel_review", "_review", "_audit"):
        if suffix in stem:
            doc_stem = stem[:stem.index(suffix)]
            parent = report_path.parent.parent  # .reports -> doc dir
            for candidate in parent.rglob(f"{doc_stem}*.md"):
                if ".reports" not in str(candidate):
                    return candidate, read_md(candidate)
    return None, ""


def pair(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM},
            {"role": "user",      "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()},
        ]
    }

# ── Defect injection for synthetic pairs ─────────────────────────────────────

DEFECT_TEMPLATES = [
    # (mutator_fn, expected_finding_text)
    (
        lambda h: re.sub(r"(Status\s*\|[^|]*\|[^|]*\|)\s*\w+", r"\1 ACTIVE", h),
        # But document is in Hold / Draft context
        "NBK: Status field shows ACTIVE but document is in Hold — verify promotion has been completed.",
    ),
    (
        lambda h: re.sub(r"\|\s*Reviewer\s*\|[^|]+", "| Reviewer | [Pending]", h),
        "NBK: Reviewer field shows [Pending] — document must not be promoted until a named reviewer signs off.",
    ),
    (
        lambda h: re.sub(r"\|\s*Author\s*\|[^|]+", "| Author | OpenAI", h),
        "BLK: Author field shows OpenAI — external LLM cannot be listed as author in AHC governed documents.",
    ),
    (
        lambda h: re.sub(r"\|\s*FPA\s*\|[^|]+", "| FPA | EXTERNAL", h),
        "BLK: FPA value EXTERNAL is not valid — allowed values are ILF, EO, EQ, EI.",
    ),
    (
        lambda h: re.sub(r"\|\s*ID\s*\|[^|]+", "| ID | std_g1023", h),
        "BLK: ID uses deprecated format std_g1023 — project uses {type_prefix}_{filename_number} format.",
    ),
    (
        lambda h: re.sub(r"(Document Version\s*\|[^|]*\|[^|]*\|)\s*[\d.]+", r"\1 1.0.0.1", h),
        "NBK: Document Version uses non-standard format 1.0.0.1 — use major.minor (e.g. 1.1).",
    ),
    (
        lambda h: re.sub(r"\|\s*PDC\s*\|[^|]+", "| PDC | UNKNOWN", h),
        "BLK: PDC value UNKNOWN is not recognized — verify the correct PDC code from the taxonomy.",
    ),
]


def inject_defect(clean_header: str) -> tuple[str, str]:
    """Apply one random defect and return (corrupted_header, expected_finding)."""
    mutator, finding = random.choice(DEFECT_TEMPLATES)
    corrupted = mutator(clean_header)
    if corrupted == clean_header:  # mutation had no effect (field not present)
        corrupted = clean_header + "\n| Reviewer | [Pending] |"
        finding = "NBK: Reviewer field shows [Pending] — document must not be promoted until a named reviewer signs off."
    return corrupted, finding


# ── Main extraction ────────────────────────────────────────────────────────────

def extract_real_pairs() -> list[dict]:
    pairs_out = []
    for reports_dir in find_reports_dirs(DOCS_DIR):
        for review_file in reports_dir.glob("*.md"):
            review_content = read_md(review_file)
            if not review_content:
                continue
            findings_text = extract_findings(review_content)
            if not findings_text:
                continue  # not a findings-format review

            doc_path, doc_content = find_reviewed_doc(review_file, review_content)
            doc_header = extract_header_block(doc_content) if doc_content else ""
            doc_version_history = extract_version_history(doc_content) if doc_content else ""

            if doc_header:
                user_msg = (
                    f"Review the following document header and version history for AHC compliance.\n\n"
                    f"### Document Header\n{doc_header}\n\n"
                    f"### Version History\n{doc_version_history}"
                )
            else:
                # Use the review's own 'Reviewed Document' reference as context
                doc_ref_m = re.search(r"Reviewed Document\s*\|\s*([^\n|]+)", review_content)
                doc_ref = doc_ref_m.group(1).strip() if doc_ref_m else review_file.stem
                user_msg = (
                    f"Review the document `{doc_ref}` for AHC compliance based on the following context.\n\n"
                    f"### Review Scope\n"
                    + re.sub(r"\n{3,}", "\n\n",
                              re.search(r"##\s+\d*\.?\s*(?:1\.?\s+)?Review Scope(.*?)(?=\n##)",
                                        review_content, re.DOTALL | re.IGNORECASE
                                        ).group(1)[:600]
                              if re.search(r"##\s+\d*\.?\s*(?:1\.?\s+)?Review Scope",
                                           review_content, re.IGNORECASE) else "")
                )

            pairs_out.append(pair(user_msg, findings_text))

    return pairs_out


def build_synthetic_pairs(real_pairs: list[dict], target_synthetic: int) -> list[dict]:
    """Generate synthetic defect-injection pairs to reach target count."""
    # Collect clean headers from real pairs
    clean_headers = []
    for p in real_pairs:
        user_text = p["messages"][1]["content"]
        header_m = re.search(r"### Document Header\n(.*?)(?=\n###|\Z)", user_text, re.DOTALL)
        if header_m:
            clean_headers.append(header_m.group(1))

    if not clean_headers:
        return []

    synthetic = []
    for _ in range(target_synthetic):
        clean = random.choice(clean_headers)
        corrupted, finding = inject_defect(clean)
        user_msg = (
            "Review the following document header for AHC compliance. "
            "Identify all header field defects.\n\n"
            f"### Document Header\n{corrupted}"
        )
        assistant_msg = (
            f"{finding}\n\n"
            "Verdict: BLOCKED — header defects must be resolved before promotion."
        )
        synthetic.append(pair(user_msg, assistant_msg))

    return synthetic


# ── Write outputs ─────────────────────────────────────────────────────────────

def write_dataset(all_pairs: list[dict]):
    with open(DATASET_OUT, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def write_manifest(real_count: int, synthetic_count: int, all_pairs: list[dict]):
    samples = "\n\n".join(
        f"### Sample {i+1}\n\n"
        f"**User**: {p['messages'][1]['content'][:300]}...\n\n"
        f"**Assistant**: {p['messages'][2]['content'][:300]}..."
        for i, p in enumerate(all_pairs[:3])
    )
    content = textwrap.dedent(f"""\
        # LRA-DR01 Document Reviewer — Training Dataset Manifest

        **Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%MTC')}
        **Output**: `doc_reviewer_dataset.jsonl`
        **Target adapter**: LRA-DR01 (doc_reviewer.gguf)

        ## Statistics

        | Metric | Value |
        |:-------|------:|
        | Real pairs (extracted from .reports/) | {real_count} |
        | Synthetic pairs (defect injection) | {synthetic_count} |
        | Total | {real_count + synthetic_count} |

        ## Source
        - Review artifacts: all `docs/**/.reports/*.md` files containing BLK/NBK findings
        - Synthetic defects: injected into clean headers from real pairs

        ## Samples

        {samples}
    """)
    MANIFEST_OUT.write_text(content, encoding="utf-8")


def main():
    random.seed(42)
    print("Extracting real (doc, review) pairs from .reports/ directories...")
    real_pairs = extract_real_pairs()
    print(f"  Real pairs extracted: {len(real_pairs)}")

    target_synthetic = max(0, int(len(real_pairs) * 0.67))  # ~40% of total
    print(f"  Generating {target_synthetic} synthetic defect-injection pairs...")
    synthetic_pairs = build_synthetic_pairs(real_pairs, target_synthetic)

    all_pairs = real_pairs + synthetic_pairs
    random.shuffle(all_pairs)

    write_dataset(all_pairs)
    write_manifest(len(real_pairs), len(synthetic_pairs), all_pairs)

    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Output: {DATASET_OUT}")
    print(f"  Manifest: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()

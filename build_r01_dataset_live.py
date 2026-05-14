#!/usr/bin/env python3
"""
build_r01_dataset_live.py — Build LRA-R01 training corpus from live DocCore rows.

This builder keeps the corpus grounded in the current DocCore database instead of
hardcoding governance text into the adapter. It reads live rows from
doccore_section_embeddings on ub01, turns each row into a retrieval-style prompt,
and writes ChatML examples for the RAG synthesis adapter.

Usage:
  python3 build_r01_dataset_live.py

Output:
  - rag_synthesis_dataset.jsonl
  - rag_synthesis_manifest.md
"""

from __future__ import annotations

import base64
import json
import pathlib
import re
import subprocess
import textwrap


OUT = pathlib.Path(__file__).with_name("rag_synthesis_dataset.jsonl")
MANIFEST = pathlib.Path(__file__).with_name("rag_synthesis_manifest.md")

SYSTEM_TEMPLATE = textwrap.dedent(
    """\
    You are a precise technical assistant.
    You are given the following retrieved documentation chunk from DocCore.
    Base your answer strictly on the retrieved context. If the context does not
    contain enough information, say so explicitly.

    Retrieved context:
    [1] (document: {document_id}; section: {section_id}; section_type: {section_type}; pdc: {pdc})
    {chunk}
    """
).strip()


def fetch_live_rows(limit: int = 300) -> list[dict[str, str]]:
    """
    Fetch live DocCore chunk rows from the ub01 MariaDB host via SSH.

    The query intentionally filters out .codex smoke-test artifacts so the corpus
    stays focused on real documentation content.
    """

    sql_cmd = textwrap.dedent(
        f"""\
        mariadb -N -B -h 127.0.0.1 -u mgr_usr01 -pmanager_password ah_db01 -e "
        SELECT document_id, section_id, section_type, COALESCE(pdc,''), TO_BASE64(chunk_text)
        FROM doccore_section_embeddings
        WHERE tenant_id='default-tenant-id'
          AND document_id NOT LIKE '.codex-%'
          AND LENGTH(chunk_text) > 80
        ORDER BY created_at DESC
        LIMIT {limit};
        "
        """
    ).strip()

    raw = subprocess.check_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "ub01",
            sql_cmd,
        ],
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        document_id, section_id, section_type, pdc, chunk_b64 = parts[:5]
        try:
            chunk = base64.b64decode(chunk_b64).decode("utf-8", "replace")
        except Exception:
            continue
        rows.append(
            {
                "document_id": document_id,
                "section_id": section_id,
                "section_type": section_type,
                "pdc": pdc,
                "chunk": chunk,
            }
        )
    return rows


def extract_title(chunk: str, document_id: str) -> str:
    for line in chunk.splitlines():
        text = line.strip()
        if text.startswith("#"):
            title = re.sub(r"^#+\s*", "", text).strip()
            if title:
                return title
    return document_id


def clean_answer(chunk: str) -> str:
    text = chunk.replace("\r", "").strip()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    paras = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    if not paras:
        return text[:1400].strip()

    # Keep the heading plus the first two body paragraphs to stay grounded but concise.
    selected: list[str] = []
    for para in paras[:3]:
        selected.append(para)
        if len("\n\n".join(selected)) >= 1000:
            break
    answer = "\n\n".join(selected)
    answer = re.sub(r"^#+\s*", "", answer, count=1)
    return answer[:900].strip()


def make_question(title: str, idx: int) -> str:
    templates = [
        "What does the AgentHub document section '{title}' cover?",
        "Summarize the live DocCore context for '{title}'.",
        "What are the key points in the '{title}' section?",
    ]
    return templates[idx % len(templates)].format(title=title)


def main() -> None:
    rows = fetch_live_rows()
    examples: list[dict] = []
    seen: set[tuple[str, str]] = set()

    templates = [
        "What does the AgentHub document section '{title}' cover?",
        "Summarize the live DocCore context for '{title}'.",
        "What are the key points in the '{title}' section?",
    ]

    for idx, row in enumerate(rows):
        title = extract_title(row["chunk"], row["document_id"])
        answer = clean_answer(row["chunk"])
        if not answer:
            continue

        system = SYSTEM_TEMPLATE.format(
            document_id=row["document_id"],
            section_id=row["section_id"],
            section_type=row["section_type"],
            pdc=row["pdc"] or "",
            chunk=row["chunk"][:1200].strip(),
        )

        for variant_idx, template in enumerate(templates):
            question = template.format(title=title)
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)

            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )

    if not examples:
        raise SystemExit("No R01 examples collected from live DocCore rows.")

    OUT.write_text(
        "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples),
        encoding="utf-8",
    )
    MANIFEST.write_text(
        textwrap.dedent(
            f"""\
            # RAG Synthesis Corpus Manifest

            - Source: live DocCore rows from `doccore_section_embeddings`
            - Tenant: `default-tenant-id`
            - Filter: excluded `.codex-*` smoke-test rows
            - Examples written: {len(examples)}
            - Build target: train the adapter to answer from retrieved DocCore context rather than memorizing governance text
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} examples to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_r01_dataset.py — Build training corpus for LRA-R01 (RAG Synthesis Adapter)

Produces: rag_synthesis_dataset.jsonl
Format:   ChatML messages — one JSON object per line with a "messages" key.

Sources (applied in this order):
  1. MQTT log replay  — reads retriever-service MQTT logs from a JSONL dump file.
                        Pairs are (doccore_query event, doccore_answer event).
                        Only pairs where the answer passed human review are included
                        (human_approved=true in the annotation file, if provided).
  2. Hand-labeled CSV — simple CSV with columns: query, chunks_json, ideal_answer.
                        chunks_json is a JSON array of {chunk_text, pdc, document_id}.
  3. DocCore synthetic — queries and chunk sets are generated from the DocCore corpus
                         directly; answers are written by a higher-capability model
                         (or hand-written) for bootstrap.

Usage:
  python3 build_r01_dataset.py [--output OUTPUT] [--mqtt-log FILE] [--labeled-csv FILE]
                                [--doccore-synthetic FILE] [--min-pairs N]

Output format (each line):
  {
    "messages": [
      {"role": "system",    "content": "<RAG synthesis system prompt with chunks>"},
      {"role": "user",      "content": "<original query>"},
      {"role": "assistant", "content": "<ideal grounded answer with [N] citations>"}
    ]
  }

Behavioral targets encoded in the system prompt:
  - All key facts from chunks must appear in the answer.
  - Citations [1]..[N] must map to actual chunk indices.
  - Explicitly say "The provided context does not contain…" when evidence is absent.
  - Never contradict a chunk. Never invent a citation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

SYSTEM_PREAMBLE = (
    "You are a precise technical assistant. You have access to the following "
    "retrieved documentation chunks:\n\n{chunks}\n\n"
    "Rules:\n"
    "1. Base your answer strictly on the provided chunks.\n"
    "2. Cite each chunk you use with its index number, e.g. [1], [2].\n"
    "3. If a chunk contains a relevant fact, include it — do not silently omit evidence.\n"
    "4. If the chunks do not contain enough information to answer the question, say so explicitly: "
    "'The provided context does not contain sufficient information to answer this question.'\n"
    "5. Never invent citations or claim a chunk says something it does not say."
)


def format_chunks(chunks: list[dict]) -> str:
    """Format a list of chunk dicts into a numbered context block."""
    lines = []
    for i, c in enumerate(chunks, 1):
        doc   = c.get("document_id") or c.get("doc_id") or "unknown"
        pdc   = c.get("pdc") or ""
        text  = (c.get("chunk_text") or c.get("content") or "").strip()
        header = f"[{i}] (doc: {doc}" + (f", pdc: {pdc}" if pdc else "") + ")"
        lines.append(header)
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def make_example(query: str, chunks: list[dict], ideal_answer: str) -> dict:
    chunk_block = format_chunks(chunks)
    system_content = SYSTEM_PREAMBLE.format(chunks=chunk_block)
    return {
        "messages": [
            {"role": "system",    "content": system_content},
            {"role": "user",      "content": query.strip()},
            {"role": "assistant", "content": ideal_answer.strip()},
        ]
    }


def load_mqtt_log(path: Path, annotations_path: Path | None) -> list[dict]:
    """
    Load pairs from a JSONL dump of MQTT messages on:
      agenthub/comms/doccore_query  — {query, tenant_id, ts, ...}
      agenthub/comms/doccore_answer — {query, answer, provenance, ts, ...}

    The dump format expected:
      {"topic": "agenthub/comms/doccore_query",  "payload": {...}, "ts": "..."}
      {"topic": "agenthub/comms/doccore_answer", "payload": {...}, "ts": "..."}

    Pairs are matched by query text (exact match). Provenance chunks are used
    as the retrieved context; ideal_answer is the logged answer.

    If annotations_path is provided, only pairs with human_approved=true are kept.
    """
    if not path.exists():
        print(f"  MQTT log not found: {path}", file=sys.stderr)
        return []

    approved_queries: set[str] | None = None
    if annotations_path and annotations_path.exists():
        approved_queries = set()
        with open(annotations_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("human_approved", "").strip().lower() in ("true", "1", "yes"):
                    approved_queries.add(row["query"].strip())

    queries: dict[str, dict] = {}   # query -> query event payload
    answers: dict[str, dict] = {}   # query -> answer event payload

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic   = event.get("topic", "")
            payload = event.get("payload", {})
            q_text  = (payload.get("query") or "").strip()
            if not q_text:
                continue
            if "doccore_query" in topic:
                queries[q_text] = payload
            elif "doccore_answer" in topic:
                answers[q_text] = payload

    examples = []
    for q_text, ans_payload in answers.items():
        if q_text not in queries:
            continue
        if approved_queries is not None and q_text not in approved_queries:
            continue
        provenance = ans_payload.get("provenance") or []
        # Reconstruct minimal chunk dicts from provenance
        chunks = [
            {
                "document_id": p.get("document_id"),
                "pdc":         p.get("pdc"),
                "chunk_text":  p.get("snippet") or "",
            }
            for p in provenance
        ]
        if not chunks:
            continue
        ideal_answer = (ans_payload.get("answer") or "").strip()
        if not ideal_answer:
            continue
        examples.append(make_example(q_text, chunks, ideal_answer))

    print(f"  MQTT log: {len(examples)} approved pairs loaded from {path}")
    return examples


def load_labeled_csv(path: Path) -> list[dict]:
    """
    Load hand-labeled pairs from CSV with columns:
      query, chunks_json, ideal_answer

    chunks_json: JSON array of {chunk_text, pdc, document_id}
    """
    if not path.exists():
        print(f"  Labeled CSV not found: {path}", file=sys.stderr)
        return []

    examples = []
    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            try:
                chunks = json.loads(row["chunks_json"])
            except (KeyError, json.JSONDecodeError) as e:
                print(f"  CSV row {i}: skip — {e}", file=sys.stderr)
                continue
            query        = (row.get("query") or "").strip()
            ideal_answer = (row.get("ideal_answer") or "").strip()
            if not query or not ideal_answer or not chunks:
                continue
            examples.append(make_example(query, chunks, ideal_answer))

    print(f"  Labeled CSV: {len(examples)} pairs loaded from {path}")
    return examples


def load_doccore_synthetic(path: Path) -> list[dict]:
    """
    Load pre-built synthetic examples from a JSONL file.
    Expected format per line:
      {"query": "...", "chunks": [...], "ideal_answer": "..."}
    """
    if not path.exists():
        print(f"  Synthetic JSONL not found: {path}", file=sys.stderr)
        return []

    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Synthetic line {i}: skip — {e}", file=sys.stderr)
                continue
            query        = (rec.get("query") or "").strip()
            chunks       = rec.get("chunks") or []
            ideal_answer = (rec.get("ideal_answer") or "").strip()
            if not query or not ideal_answer or not chunks:
                continue
            examples.append(make_example(query, chunks, ideal_answer))

    print(f"  Synthetic JSONL: {len(examples)} pairs loaded from {path}")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove exact-duplicate examples by (query, answer) pair."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for ex in examples:
        msgs   = ex["messages"]
        key    = (msgs[1]["content"], msgs[2]["content"])
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def parse_args():
    p = argparse.ArgumentParser(description="Build LRA-R01 RAG synthesis training corpus")
    p.add_argument("--output",           type=Path, default=Path(__file__).parent / "rag_synthesis_dataset.jsonl")
    p.add_argument("--mqtt-log",         type=Path, default=None,
                   help="JSONL dump of MQTT doccore_query + doccore_answer events")
    p.add_argument("--mqtt-annotations", type=Path, default=None,
                   help="CSV with columns: query, human_approved (true/false)")
    p.add_argument("--labeled-csv",      type=Path, default=None,
                   help="CSV with columns: query, chunks_json, ideal_answer")
    p.add_argument("--doccore-synthetic",type=Path, default=None,
                   help="JSONL with {query, chunks, ideal_answer} synthetic examples")
    p.add_argument("--min-pairs",        type=int,  default=50,
                   help="Warn (but don't abort) if total pairs < this (default: 50)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\nBuilding LRA-R01 dataset → {args.output}")

    all_examples: list[dict] = []

    if args.mqtt_log:
        all_examples.extend(load_mqtt_log(args.mqtt_log, args.mqtt_annotations))

    if args.labeled_csv:
        all_examples.extend(load_labeled_csv(args.labeled_csv))

    if args.doccore_synthetic:
        all_examples.extend(load_doccore_synthetic(args.doccore_synthetic))

    all_examples = deduplicate(all_examples)

    print(f"\nTotal unique pairs: {len(all_examples)}")
    if len(all_examples) < args.min_pairs:
        print(f"WARNING: only {len(all_examples)} pairs — target is 800. "
              f"Run after Wave 1 retriever traffic accumulates.", file=sys.stderr)

    if not all_examples:
        print("No pairs collected — output file not written.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Written: {args.output}  ({args.output.stat().st_size // 1024} KB)")
    print(f"\nNext step: add 'r01' to train_lora_rocm.py ADAPTERS registry and run:")
    print(f"  python3 train_lora_rocm.py --adapter r01")


if __name__ == "__main__":
    main()

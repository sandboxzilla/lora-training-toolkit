"""
retriever_client.py — Shared DocCore retrieval client for all LRA dataset builders

All LoRA adapter dataset builders import this module to fetch live governance
context from the retriever-service at build time. This ensures:

  - No governance rules, templates, or constraints are hardcoded in any builder
  - When rules change in DocCore, rebuilding the dataset picks up the changes
  - The model learns to USE retrieved context, not to recall rules from weights

Usage in a builder:
    from retriever_client import RetrieverClient, offline_note

    client = RetrieverClient(args.retriever, args.tenant, args.top_k)
    context = client.fetch(queries)   # returns str, empty if offline
    if client.offline:
        print(offline_note(args.retriever))
"""

import hashlib
import json
from pathlib import Path

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

OFFLINE_PLACEHOLDER = (
    "[CONTEXT PENDING — retriever unavailable at build time. "
    "Rebuild on ub02 with retriever-service running at {url} to populate live content. "
    "Do NOT train on a dataset that contains this placeholder.]"
)


class RetrieverClient:
    """
    Fetches DocCore context at dataset build time.
    Results are cached in memory so repeated queries for the same text
    do not make duplicate HTTP calls during a single build run.
    """

    def __init__(self, url: str = "http://localhost:8082",
                 tenant: str = "agenthub",
                 top_k: int = 5):
        self.url     = url.rstrip("/")
        self.tenant  = tenant
        self.top_k   = top_k
        self.offline = False
        self._cache: dict[str, str] = {}
        self._check()

    def _check(self) -> None:
        if not _HAS_REQUESTS:
            self.offline = True
            return
        try:
            r = _requests.get(f"{self.url}/health", timeout=5)
            self.offline = r.status_code != 200
        except Exception:
            self.offline = True

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(f"{self.url}|{self.tenant}|{query}".encode()).hexdigest()

    def _fetch_one(self, query: str, purpose: str) -> str:
        """Fetch a single query. Returns retrieved text or empty string."""
        key = self._cache_key(query)
        if key in self._cache:
            return self._cache[key]
        try:
            resp = _requests.post(
                f"{self.url}/retrieve",
                json={"query": query, "tenant_id": self.tenant, "top_k": self.top_k},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            chunks = data.get("chunks", [])
            sources = [c.get("source", "") for c in chunks if c.get("source")]

            block = f"[Retrieved for: {purpose}]\n"
            if answer:
                block += answer
            elif chunks:
                block += "\n".join(c.get("text", "") for c in chunks[:3])
            if sources:
                unique = list(dict.fromkeys(s for s in sources if s))[:3]
                block += f"\n(Sources: {', '.join(unique)})"

            self._cache[key] = block
            return block
        except Exception as e:
            print(f"  [rag warn] {purpose}: {e}")
            return ""

    def fetch(self, queries: list[dict]) -> str:
        """
        Execute a list of {purpose, query} dicts.
        Returns concatenated retrieved blocks, or placeholder if offline.
        """
        if self.offline:
            return OFFLINE_PLACEHOLDER.format(url=self.url)
        blocks = []
        for q in queries:
            purpose    = q.get("purpose", "general")
            query_text = q.get("query", "")
            if not query_text:
                continue
            print(f"    [rag] {purpose}: {query_text[:70]}...")
            block = self._fetch_one(query_text, purpose)
            if block:
                blocks.append(block)
        return "\n\n---\n\n".join(blocks) if blocks else ""

    def fetch_one(self, query: str, purpose: str = "") -> str:
        """Fetch a single query string."""
        if self.offline:
            return OFFLINE_PLACEHOLDER.format(url=self.url)
        return self._fetch_one(query, purpose or query[:40])


def offline_note(url: str) -> str:
    return (
        f"\n  *** OFFLINE BUILD ***\n"
        f"  Retriever unavailable at {url}.\n"
        f"  Pass 2 training examples contain placeholder context.\n"
        f"  Run on ub02 with: python3 <builder> --retriever http://localhost:8082\n"
        f"  Do NOT train on a dataset built offline — placeholder context will\n"
        f"  teach the model to expect garbage in its context window.\n"
    )


def make_system_prompt(role: str, review_subject: str, output_format: str) -> str:
    """
    Generate a system prompt that enforces query-first behavior.
    NO AgentHub-specific rules are embedded here — they are always retrieved.
    """
    return f"""\
You are a {role} for AgentHub.

## Mandatory protocol — query before reviewing
You have NO pre-loaded knowledge of AgentHub's specific rules, constraints,
templates, or standards. Before reviewing any artifact, you MUST use the
provided governance context (retrieved from DocCore) to understand the
applicable rules. Never apply rules from memory.

If governance context is not present in the prompt, emit a PENDING_CONTEXT
response with the specific rag_queries needed. Do not guess.

## What you review
{review_subject}

## Output format
{output_format}

## Universal rules (these never change and need no retrieval)
- PROVISIONAL_*, PROVISIONAL-*, *_PENDING_* in ID fields are always BLK
- citeturn*, fileciteturn* strings in document body are always BLK (citation artifacts)
- Tests with zero expect() assertions are always BLK
- SQL queries without a tenant_id filter in a multi-tenant handler are always BLK
"""

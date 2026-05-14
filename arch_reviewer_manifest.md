# LRA-AR01 Architecture Reviewer — Training Dataset Manifest (v2)

**Generated**: 2026-04-15 12:15Z
**Retriever**: OFFLINE — placeholder context
**Total pairs**: 73

## Design
No AgentHub architectural constraints are hardcoded in training examples.
All constraint content is fetched from DocCore at build time.
Findings cite "retrieved constraint above" — not embedded rule text.
Rebuild dataset when architecture rules change; no code changes needed.

## Statistics
| Category | Count |
|:---------|------:|
| Real proposal pairs (with/without paired review) | 49 |
| Synthetic injection pairs | 24 |
| **Total** | **73** |

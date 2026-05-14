# LRA-CR01 Code Reviewer — Training Dataset Manifest (v2)

**Generated**: 2026-04-15 12:15Z
**Retriever**: OFFLINE — placeholder context
**Total pairs**: 392

## Design
No AgentHub-specific rules are hardcoded in training examples.
All governance context is fetched from DocCore at build time.
Findings reference "retrieved rules above" — not embedded rule text.
Rebuild dataset when rules change; no code changes needed.

## Statistics
| Category | Count |
|:---------|------:|
| Source annotation + clean pairs | 331 |
| Git diff pairs | 45 |
| Defect injection pairs | 16 |
| **Total** | **392** |

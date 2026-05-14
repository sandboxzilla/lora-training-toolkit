        # LRA-DR01 Document Reviewer — Training Dataset Manifest (v2)

        **Generated**: 2026-04-15 12:15Z
        **Output**: `doc_reviewer_dataset.jsonl`
        **Target adapter**: LRA-DR01 (doc_reviewer.gguf)
        **Schema**: v2 — two-pass RAG-aware, live-retrieved context
        **Retriever**: **OFFLINE BUILD** — Pass 2 context is placeholder only. Rebuild with retriever online before training.

        ## Statistics

        | Category | Count |
        |:---------|------:|
        | Real pairs — BLK/NBK findings | 52 |
| Real pairs — APPROVED (JSON schema) | 13 |
| Defect-injection pairs | 20 |
| PROVISIONAL ID (Pass 1 + Pass 2) | 12 |
| Citation artifact pairs | 3 |
| File naming pairs | 3 |
| Full two-pass review pairs | 4 |
        | **Total** | **107** |

        ## Design Notes
        - No governance rules are hardcoded in this builder.
        - Pass 2 context is fetched from DocCore at build time using the same
          queries that the model emits at inference time.
        - When governance rules change (g0001, g1005, o2004), rebuild this
          dataset and retrain — no code changes required.
        - Verdict-only APPROVED real pairs are normalized to JSON schema.

        ## Samples

        ### Sample 1

**User**: Review the following document for AHC governance compliance. All governance context has been retrieved.

### Document Content
# Document Title

| Field | Value | Field | Value |
|-------|-------|-------|-------|
| File | o4411_public_dashboard_api_security_test_and_pentest_plan.md | Document Version...

**Assistant**: {
  "status": "COMPLETE",
  "review_id": "q0d01",
  "reviewed_document": "o4411_public_dashboard_api_security_test_and_pentest_plan",
  "output_dir": "docs/Hold/.reports/",
  "output_stem": "q0d01_o4411_public_dashboard_api_security_test_and_pentest_plan_review",
  "verdict": "NEEDS_REVISION",
  "ve...

### Sample 2

**User**: Review the document `sentinel_review_o4800_delegated_authority_hld_2026-04-05` for AHC compliance.

### Review Scope...

**Assistant**: **BLK-001**: RS256 references in the reviewed sections are now explicitly labeled as target state, and current HS256 usage is called out.

2. **BLK-002**: Section 11.0 labels the `grants` table as `[PROPOSED NEW TABLE]`.

3. **BLK-003**: Section 9.2 is labeled `[PROPOSED]`, and the live API contract...

### Sample 3

**User**: Review the document `sentinel_review_task4716_distributed_hive_validation_2026-04-05` for AHC compliance.

### Review Scope...

**Assistant**: **BLK-001 — UNRESOLVED**

The deprecated live smoke tests `VAL-001` through `VAL-005` are gone, and Section 12 is now rewritten as `VER-001` through `VER-005` in [task_4716_distributed_hive_runtime_integrated_validation.md](/home/erol/workspace/agent_hub.v03/agent_hub_main/docs/(arc)architecture_and...

### Sample 4

**User**: Review the document `o2828_hlmm_task_set_v11_review` for AHC compliance.

### Review Scope


**2.1.** Documents under review:

| Document | ID | Version |
|----------|----|---------|
| `docs/hlmm/tasks/task_2206_hlmm_foundation.md` | RUN-2206 | 1.1 |
| `docs/hlmm/tasks/task_2207_hlmm_package_assembl...

**Assistant**: ## 8. Verdict

**APPROVED**

All four BLK findings from RVW-2816 are RESOLVED. Both NBK findings from RVW-2816 are RESOLVED. BLK-005 (RVW-2816 re-review addendum) is RESOLVED via HLMM HLD creation. All three NBK findings from this review (NBK-001, NBK-002, NBK-003) are CLOSED.

The HLMM task set v1....

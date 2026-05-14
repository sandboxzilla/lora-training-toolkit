        # LRA-RT01 Router Adapter — Training Dataset Manifest (v2)

        **Generated**: 2026-04-15 12:15Z
        **Retriever**: OFFLINE — placeholder context
        **Total pairs**: 141

        ## Design
        Route taxonomy is NOT hardcoded. It is fetched from DocCore at build time.
        When adapters are added or routes change, rebuild dataset — no code changes.
        All justifications reference "the taxonomy above" from the retrieved context.

        ## Route Distribution
        | Route | Count |
        |:------|------:|
        | RT-AR | 6 |
| RT-BASE | 70 |
| RT-CR | 11 |
| RT-DOC | 11 |
| RT-DR | 25 |
| RT-FALLBACK | 2 |
| RT-LA | 3 |
| RT-LD | 2 |
| RT-TA | 7 |
| RT-TR | 4 |

        ## Statistics
        | Category | Count |
        |:---------|------:|
        | Synthetic examples | 24 |
        | Hard negatives | 7 |
        | Real task/bus pairs | 110 |
        | **Total** | **141** |

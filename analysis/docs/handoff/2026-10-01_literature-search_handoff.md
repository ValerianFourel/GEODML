# Literature search and saved prompt plan

Date: 2026-10-01. Active checkout: `.worktrees/threehour-relaunch-fix`, branch `threehour-relaunch-fix`, upstream `origin/codex/pilot-continuation`.

## Request and completed work

Valerian requested that the ten-prompt literature plan be saved as Markdown in the workspace and Downloads, and that the searches be executed. Canonical artifacts are in `analysis/docs/literature/`:

- `2026-10-01_geodml-literature-search-plan.md`: shared instructions and ten detailed prompts.
- `2026-10-01_geodml-literature-review.md`: synthesis, five read-first papers for each theme, competitor and estimand comparisons, validation priorities.
- `2026-10-01_geodml-annotated-bibliography.md`: theme tables and 93 distinct bibliographic entries.
- `2026-10-01_geodml-bibliography.json` and `.bib`: structured citations, provenance and inspection levels.
- `2026-10-01_geodml-search-log.md`: queries, failed requests, exclusions, deduplication and coverage limitations.
- Dated evidence JSON files: OpenAlex/Crossref responses and primary-source metadata/abstract checks.

The six user-facing files were copied to `/Users/valerianfourel/Hamburg/GEODML_Unified` and `/Users/valerianfourel/Downloads`. Both locations also contain `2026-10-01_geodml-literature-bundle.zip`, including the evidence files. These exports are outside the active checkout; the canonical source is committed here. No push was performed.

## Evidence and limitations

Executed public scholarly API searches, primary arXiv retrieval and ACL abstract checks across all ten themes. Twelve papers received targeted full-text inspection, 63 abstract inspection, and 18 metadata-only inspection. OpenAlex/Crossref rate limits caused some failed requests; these are recorded. Several wrong remembered identifiers and irrelevant title matches were rejected after online inspection, and are not in the bibliography. This is a broad literature map, not an exhaustive systematic review. The suggested 15–25 papers per theme was not reached in every theme; counts are explicit.

Recent 2025–2026 GEO papers overlap with source ownership, page structure, citation selection versus answer absorption, and agentic citation repair. Their abstracts narrow broad novelty claims; full methods remain to be inspected. Metadata presence does not establish peer review. E-GEO is a metadata-only lead. LLM2Vec-Gen's retrieved abstract describes an output-space embedding, reinforcing the need to re-embed proposals in the project's actual readiness spaces.

Scientific distinctions preserved: V2 readiness is an observed text property; historical randomized B is separate; page-feature DML remains observational absent justified identification; textual support, citations and causal reliance are separate; SI-v4 is development work, not a validated gold standard. No project headline results were changed or inferred from job counts.

## Verification

Checked 93 unique citation keys and normalized titles, unique DOI records, required author/year/theme fields, JSON parsing, 93 BibTeX entries, and every local Markdown file/anchor link. Export hashes match canonical files. Both ZIPs passed integrity checks. No application behavior changed; no inference or cluster jobs were run. Cluster facts from preceding handoffs were not rechecked live because this task involved only local documents and public literature; no claim about current allocation or queue state is made.

## Next work

Before a manuscript novelty claim, inspect the recent citation-absorption, structural-GEO and AgentGEO papers in full. Extend the thinner statistical and adaptive-intervention themes once the final estimands are fixed. External readiness validation and human SI-v4 calibration are recommendations, not completed experiments. This search and its reports are complete as a dated first-pass map; the plan and logs support further research without silently upgrading inspection levels.

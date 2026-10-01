# From Prompt Semantics to Source Rankings in Generative Search

Complete first manuscript draft, prepared 2026-10-02. Open `main.pdf`; edit `main.tex` and the four files in `sections/`. This is an evidence-limited working draft, not a submission-ready paper: missing research outputs are explicitly marked `[TODO: ...]` throughout.

## Format and contents

The old anonymous ACL PDF is preserved in `ARR_ACL_CycleOct2026/oldPaper/9568_What_Drives_LLM_Re_Rankin.pdf`. Its original LaTeX source was not located. This separate draft uses the official ACL style and bibliography style at revision `d5adc823ff0f80f98c80405ca0ab66c68e684409`, matching the old paper's format. Template hashes are in `evidence/template-provenance.json`. Full-width appendices are a working-draft readability choice; page limits and final submission formatting still need a later editorial pass.

The draft contains all requested sections, the two requested research questions, three distinct contributions, a design/counts table, a real example, concise strategy pseudocode, a workflow figure, a population-measurement figure, exact current instructional templates, model/configuration details, and a reviewer-concern mapping. Limitations follows Conclusion. The abstract contains 167 whitespace-delimited words including its TODO.

The main ranking questions have no invented answer. Generator order, judge-derived request relevance, and judge-derived answer support are kept separate. The example's all-zero scores are historical SI-v3 development outputs, not validated SI-v4 measurements. Final prompt files were read and rehashed; completion numbers remain dated manifest/bundle evidence rather than a new live census.

## Build

From this directory, with TeX Live and latexmk installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figures, templates, bibliography and small supporting artifacts are already included, so rebuilding the PDF does not need access to the private population or models. The ACL style selects `acl_natbib` itself; do not add a second `\bibliographystyle` command.

`prepare_artifacts.py` is a repository-local helper. Run it from the authoritative `analysis/paper/arr_oct2026_first_draft/` directory in the active checkout, where it can read the pipeline source files. It can regenerate the manuscript-only figures and summaries from locally available evidence:

```bash
python prepare_artifacts.py \
  --population-root /path/to/pinned/population/snapshot \
  --library-root /path/to/papers/_library \
  --example evidence/running-example.json
```

The population root must contain `final-audit/compliant-candidates.jsonl` and `final-audit/final-axis-map.jsonl` with the exact expected hashes. The script verifies 26,009 text-hash joins and extracts templates without importing or loading an inference model. It performs no search, inference, ranking analysis, or resampling. It uses NumPy and Matplotlib for a small descriptive plot. Citation metadata was checked against the existing primary-source library and three DOI records; the GEO publication details were verified on its downloaded PDF first page after Crossref returned HTTP 429. No large dataset or model is included.

## Evidence and remaining work

See `evidence/claim-source-map.md` for the source of each substantive assertion and `missing-evidence.md` for the smallest required completion tasks. `reviewer-response-map.md` maps all nine supplied reviewer concerns. The original reviewer files were not found; the supplied editorial text and old manuscript were used, without claiming access to the missing originals.

The evidence folder contains internal provenance, including archived paths and unredacted source metadata. It is a private author working bundle, **not an anonymous artifact release**. The PDF has an anonymous author field and makes no claim of public artifact availability. Build/runtime caches are excluded; the old manuscript, pipeline, datasets, configuration files and scientific outputs were not changed. No cluster connection, allocation, new inference, or expensive experiment was performed.

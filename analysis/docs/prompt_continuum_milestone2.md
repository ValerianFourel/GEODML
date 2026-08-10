# Prompt continuum: Milestone 2 calibration corpus

The calibration corpus makes the current deterministic `P = G(B, S)` scaffold
inspectable before any model inference. It reuses each surface seed `S` across a
regular diagnostic grid of `B` values, making both style invariance and the
piecewise-constant preference schedule visible.

The regular grid is not the assignment mechanism for the confirmatory
experiment. A later milestone will randomly sample continuous `B`; the grid in
this milestone exists only for calibration, visualization, and structural
auditing.

From the repository root, generate the default 220-record corpus locally:

```bash
python3 analysis/scripts/generate_prompt_calibration.py --output-dir analysis/output/prompt_calibration
```

The command writes:

- `analysis/output/prompt_calibration/prompt_calibration.jsonl`, containing the
  exact prompt template, `B`, `S`, `top_n`, complete style plan, stable prompt
  identity, generator metadata, timestamp, and manifest version for every row;
- `analysis/output/prompt_calibration/prompt_calibration_report.md`, containing
  corpus statistics, reproducibility checks, per-seed axis diagnostics,
  structural audits, and complete requested examples.

Existing artifacts are never overwritten implicitly. Pass `--overwrite` only
after reviewing the target path. The JSONL loader
`load_calibration_manifest(path)` reconstructs typed `PromptRecord` objects and
rejects missing fields, invalid bounds, corrupt hashes or IDs, inconsistent
metadata, and conflicting prompt-ID reuse.

The report must be read as an engineering audit, not semantic validation. The
`TemplatePromptGenerator` maps continuous-valued `B` onto a finite monotonic
phrase schedule, so adjacent grid values can produce identical prompt text.
This scaffold is not the final scientific generator; semantic generation and
validation are intentionally deferred.

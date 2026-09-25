# Randomized prompt-continuum project: implementation handoff

## Purpose and scientific boundary

The project asks how page properties influence prompted LLM reranking when the
prompt policy is assigned as `P = G(B, S)`:

- `B` is the assigned intensity of preference for first-party
  software-product sources;
- `S` controls surface realization without adding ranking criteria;
- `P` is the complete reranking prompt template.

Randomizing `B` can identify effects of the prompt policy. Page-feature effects
remain observational DML estimates unless page content is itself manipulated.
Prompt embeddings may later describe realized prompts, but they do not define
`B` and must not be called confounders.

The exact legacy neutral and biased prompt functions remain unchanged as
historical baselines. No result has been inferred from mocked data, unit tests,
or calibration artifacts.

## What has been implemented

### Milestone 1: deterministic foundation

`prompt_continuum.py` introduced typed request, style-plan, and prompt-record
objects plus `TemplatePromptGenerator`. The generator is deterministic in
`B`, `S`, `top_n`, and prompt-space version; uses local random state and stable
SHA-256 identities; and preserves `{QUERY}`, `{CANDIDATES}`, and `{TOP_N}`.

The implementation deliberately maps continuous `B` to a finite monotonic
phrase schedule. It is an engineering scaffold for tests and smoke runs, not a
scientifically validated semantic continuum.

### Milestone 2: prompt-only calibration corpus

`prompt_calibration.py` generates and reloads immutable JSONL manifests and
writes a Markdown audit report. The default diagnostic corpus crosses eleven
regular `B` values with twenty reused style seeds (220 prompts). It checks
reproducibility, identity integrity, style-policy separation, endpoints,
placeholders, output contracts, and forbidden criteria.

The calibration exposed the intended limitation: the scaffold produces five
distinct policy realizations, so adjacent assigned `B` values can share policy
wording. The regular grid is diagnostic; it is not the assignment scheme for a
confirmatory reranking experiment.

Generate the local calibration artifacts with:

```bash
python3 analysis/scripts/generate_prompt_calibration.py \
  --output-dir analysis/output/prompt_calibration
```

### Milestone 3A: offline policy-clause candidate pilot

Milestone 3A added a versioned semantic specification and meta-prompt, typed
policy-clause records, a provider-independent interface, a deterministic fake
provider, and a lazy adapter around the repository's local Hugging Face model
loader. `HybridPromptComposer` combines the existing style wrapper with a
generated policy clause while preserving the invariant task and output
contract. The original deterministic generator remains available.

Generation is intentionally offline: candidate clauses are persisted exactly
and will eventually be frozen before reranking. Reranking must never call the
generation model dynamically. New records start as `unvalidated`; structural
checks reject malformed JSON, forbidden/off-axis language, hard exclusions,
numeric leakage of `B`, missing metadata, and broken placeholders. These checks
do not establish semantic monotonicity or axis purity.

The exact local dry run is:

```bash
python3 analysis/scripts/generate_policy_clause_pilot.py \
  --mode dry-run \
  --output-dir analysis/output/policy_clause_pilot_dry_run \
  --master-seed 20260810
```

A later HoreKa allocation can invoke the cluster-ready generation command:

```bash
python3 analysis/scripts/generate_policy_clause_pilot.py \
  --mode generate \
  --provider local \
  --model "$POLICY_GENERATOR_MODEL" \
  --precision full \
  --output-dir "$POLICY_PILOT_OUTPUT_DIR" \
  --master-seed 20260810 \
  --number-style-seeds 8 \
  --number-b-values 8 \
  --include-anchors
```

This command is a template for a later Slurm wrapper. No model, GPU, HoreKa
job, reranker, embeddings, semantic judge, or DML estimator was run during
these milestones.

## Original search data: smallest useful download

The full raw archive remains the source of truth, but downloading 37.6 GB is
unnecessary when only the original frozen search pools are needed. From the
repository root:

```bash
python3 analysis/scripts/download_data.py \
  --component serp \
  --local-dir ./geodml_data
```

This materializes the phase-0 DuckDuckGo and SearXNG top-20/top-50 snapshots
under `geodml_data/data/serp/`. Public files can be downloaded anonymously;
set `HF_TOKEN` when authentication is required by the Hub environment.

Other supported scopes are `dataforseo` (consolidated DataForSEO tables and
manifests), `dataforseo-full` (including raw and checkpoint API responses),
`html` (compressed per-run HTML caches), `rerank` (ranking inputs/results),
`core` (analysis tables without bulky HTML/RAG caches), and `full` (the complete
raw snapshot, which remains the default for backward compatibility). The raw
archive contains captured HTML responses but no standalone CSS assets. Preview
any selection without network or filesystem changes:

```bash
python3 analysis/scripts/download_data.py --component serp --dry-run
```

See the root `README.md` and `docs/DATA_POINTERS.md` for the reproducibility,
reviewer, and full-raw dataset roles.

## Review, commit, and HoreKa handoff

The normal sequence is local implementation and tests, Git commit and push,
checkout of that exact SHA on HoreKa, Slurm submission, asynchronous execution,
and local inspection of returned artifacts. Scientific runs must record their
Git SHA, configuration, seeds, model and revision, environment, Slurm job ID,
resources, timestamps, logs, and artifact directory.

After the small pilot, return these files for local audit:

- `policy_clause_requests.jsonl`;
- `policy_clause_candidates.jsonl`;
- `candidate_full_prompts.jsonl`;
- `policy_clause_pilot_report.md`;
- Slurm stdout and stderr;
- the run configuration/manifest recording the exact Git SHA and model.

Do not use the candidate clauses for reranking or scientific inference yet.
The smallest next milestone is semantic validation of the frozen pilot bank:
axis purity, relevance preservation, and monotonic preference intensity,
including adjudication and rejection criteria defined before inspection.

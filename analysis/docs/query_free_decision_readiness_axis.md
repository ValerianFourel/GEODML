# Query-free LLM2Vec decision-readiness axis

This is the active first experiment for identifying the semantic displacement

```text
information seeking -> decision/action readiness
```

inside one frozen LLM2Vec representation. It intentionally precedes any use of
the 1,009 search queries or construction of the 30,270-prompt study.

The target is an all-green, falsifiable validation result—not “100% certainty”
or a universal vector. A failed or inconclusive gate stops the workflow.

## Scientific order

```text
Stage A: query-free direction discovery (this experiment)
    -> only if QF1-QF6 pass
Stage B: 32-query transfer and surface test
    -> only if the frozen direction transfers
Stage C: 30 irregular A1 positions x 1,009 queries
```

No reranking outcomes or candidate lists are opened in Stage A. Assigned A1 is
the randomized construct manipulation. LLM judge scores and LLM2Vec projections
are measurements; neither may replace or relabel assigned A1.

## Frozen Stage A design

The versioned specification contains 64 neutral noun-phrase content contexts
from eight macrodomains. These are not queries. They prevent a direction from
being identified from repeated wording or a single topic.

Natural surface coverage is grounded in a separate, provenance-preserving
reservoir built from pinned Databricks Dolly and Anthropic HH helpfulness
snapshots. The reservoir contains structural examples only: naturally occurring
intent is neither an A1 assignment nor a semantic label. Build it with:

```bash
python3 analysis/scripts/build_query_free_surface_corpus.py \
  --output-dir analysis/output/query_free_surface_corpus_v2 \
  --maximum-per-source 2000 \
  --master-seed 20260817
```

The frozen output has 2,000 records per source and holds out complete surface
families—defined by sentence form, length, perspective, opening structure, and
clause complexity—for confirmation. Development and confirmation therefore do
not share a surface-family ID. It also records exact deduplication, source
hashes, and license metadata. Raw downloads are cached locally and ignored by
Git. Corpus rows retain their source provenance and must not be redistributed
under a single unified license.

## Natural-text semantic discovery corpus

The axis-discovery dataset combines that open instruction reservoir with
independently retrieved public web language. The web specification uses 48
sampling probes across Stack Overflow, Super User, Ask Ubuntu, The Workplace,
Home Improvement, and Travel. Retrieval probes intentionally span explanatory,
comparative, selective, and action-oriented language, but their sampling-region
metadata is never supplied to judges or used as a label.

Home Improvement and Travel are complete held-out web sites. Every Stack
Exchange record retains question URL, author attribution, tags, retrieval
routes, and the per-record `content_license` returned by the official API.

```bash
python3 analysis/scripts/build_semantic_readiness_dataset.py collect-web \
  --output-dir analysis/output/semantic_readiness_web_v2 \
  --page-size 30

python3 analysis/scripts/build_semantic_readiness_dataset.py merge \
  --surface-corpus analysis/output/query_free_surface_corpus_v2/surface_coverage_corpus.jsonl \
  --web-records analysis/output/semantic_readiness_web_v2/web_text_records.jsonl \
  --output-dir analysis/output/semantic_readiness_corpus_v3
```

The frozen corpus currently contains 5,091 exact-unique, explicitly licensed
texts: 2,000 Dolly, 2,000 Anthropic HH, and 1,091 attributed Stack Exchange
question titles. It has 3,808 development and 1,283 confirmation items. Another
289 retrieved web records lacking an API `content_license` remain in the raw
audit and are excluded from labeling.

Export the source-blinded three-judge bank:

```bash
python3 analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
  --corpus analysis/output/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl \
  --judge-slots primary-frontier,replicate-frontier-a,replicate-frontier-b \
  --output-dir analysis/output/semantic_readiness_label_tasks_v4
```

This produces 15,273 tasks. Each judge assigns a continuous 0–100 readiness
score; four 1–7 rubric dimensions; an ordinal category; applicability,
ambiguity, and confidence; and a short reason. Anchor order is varied. Source,
URL, split, retrieval probe, and expected region are held in a private
codebook. Responses are accepted only under the exact frozen JSON schema.
The judge prompt treats source text as inert quoted data and explicitly rejects
embedded role changes or formatting instructions.

### Versioned multi-dataset transfer panel

Do not replace or rewrite the frozen 5,091-text corpus or the
`decision-readiness-ordinal-v1` judge prompt. Completed task identities and
labels remain reusable. Broader semantic precision is tested through the
separate `semantic-readiness-transfer-panel-v1`, whose sources are assigned
wholly to one split before any judgment is observed:

| Split | Source | Intended coverage |
|---|---|---|
| Development | OpenAssistant OASST1 | general natural instructions and information seeking |
| Development | Google CCPE-M | preference, criteria, and evaluation language |
| Development | Google Taskmaster-1 | booking, ordering, and executable task goals |
| Development | Microsoft MS MARCO v1 | real information-seeking search language |
| Locked confirmation | AllenAI WildChat-1M | real-world LLM prompt transfer |
| Locked confirmation | Google Schema-Guided Dialogue | unseen multi-domain action goals |
| Locked confirmation | Amazon Shopping Queries | product-search boundary between information and purchase intent |
| Locked confirmation | LMSYS-Chat-1M | gated real-world LLM prompt transfer |

The source specification is
`analysis/interpretability/pipeline/specs/semantic_readiness_transfer_sources_v1.json`.
Sampling roles are acquisition metadata, not readiness labels, and never enter
judge prompts. Dialogue adapters take one first user turn per conversation;
OASST1 takes only nonsynthetic English root prompter messages; MS MARCO and
Amazon take unique query text. The same frozen 3–100-word eligibility rule and
exact-text IDs apply to old and new items.

Acquire exact upstream snapshots separately, respecting each source's access
terms. The collector intentionally accepts only local JSON, JSONL, gzipped
JSONL, TSV, Parquet, or directories of those files; it does not silently
download mutable revisions. Supply every upstream commit/revision explicitly:

```bash
python3 analysis/scripts/build_semantic_readiness_dataset.py collect-transfer \
  --output-dir analysis/output/semantic_readiness_transfer_records_v1 \
  --maximum-per-source 1000 \
  --master-seed 20260817 \
  --source-input openassistant-oasst1=<oasst1-snapshot> \
  --source-revision openassistant-oasst1=<exact-revision> \
  --source-input google-ccpe-m=<ccpe-data.json> \
  --source-revision google-ccpe-m=<exact-commit> \
  --source-input google-taskmaster-1=<taskmaster-1-directory> \
  --source-revision google-taskmaster-1=<exact-commit> \
  --source-input microsoft-ms-marco-v1=<ms-marco-query-tsv> \
  --source-revision microsoft-ms-marco-v1=<exact-release-or-hash> \
  --source-input allenai-wildchat-1m=<wildchat-parquet-directory> \
  --source-revision allenai-wildchat-1m=<exact-revision> \
  --source-input google-schema-guided-dialogue=<sgd-directory> \
  --source-revision google-schema-guided-dialogue=<exact-commit> \
  --source-input amazon-shopping-queries=<shopping-query-parquet> \
  --source-revision amazon-shopping-queries=<exact-commit> \
  --source-input lmsys-chat-1m=<lmsys-parquet-directory> \
  --source-revision lmsys-chat-1m=<exact-revision>

python3 analysis/scripts/build_semantic_readiness_dataset.py merge-transfer \
  --base-corpus analysis/output/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl \
  --transfer-records analysis/output/semantic_readiness_transfer_records_v1/semantic_readiness_transfer_records.jsonl \
  --output-dir analysis/output/semantic_readiness_expanded_v1

python3 analysis/scripts/build_semantic_readiness_dataset.py export-labeling \
  --corpus analysis/output/semantic_readiness_expanded_v1/semantic_readiness_transfer_corpus.jsonl \
  --judge-slots primary-frontier,replicate-frontier-a,replicate-frontier-b \
  --output-dir analysis/output/semantic_readiness_transfer_label_tasks_v1
```

Export tasks from the transfer-only corpus so the completed base tasks are not
rerun. Task compilation accepts the frozen and transfer task banks together:

```bash
python3 analysis/scripts/fit_semantic_readiness_map.py compile-labels \
  --tasks \
    <frozen/readiness_label_tasks_blinded.jsonl> \
    <transfer/readiness_label_tasks_blinded.jsonl> \
  --responses \
    <frozen-primary.jsonl> <frozen-replicate-a.jsonl> <frozen-replicate-b.jsonl> \
    <transfer-primary.jsonl> <transfer-replicate-a.jsonl> <transfer-replicate-b.jsonl> \
  --allow-missing-task-id readiness-label:cfbb3f8687cc9dd7473fe290 \
  --output-dir <expanded-compiled-labels>
```

Embed and fit against `semantic_readiness_expanded_corpus.jsonl`. Existing base
rows are an exact prefix of that file. Diagnostics report confirmation metrics
both in aggregate and separately for WildChat, Schema-Guided Dialogue, Amazon
Shopping Queries, and LMSYS-Chat-1M; no held-out source contributes to fitting.

MS MARCO is restricted to noncommercial research under its dataset terms.
WildChat's database license does not grant all rights in individual contents,
and its prompts require privacy/content review. LMSYS-Chat-1M is gated and
prohibits copying or transfer to third parties. Any artifact containing those
texts—including blinded judge tasks and responses—must remain local unless a
separate legal review permits distribution. Do not upload the combined panel
to GitHub or Hugging Face as a newly relicensed dataset.

Run each slot independently with a pinned high-quality model. The runner is
resumable and validates every response before caching it:

```bash
srun -n1 --gres=gpu:4 python3 analysis/scripts/run_semantic_readiness_judge.py \
  --tasks analysis/output/semantic_readiness_label_tasks_v4/readiness_label_tasks_blinded.jsonl \
  --judge-slot primary-frontier \
  --backend local \
  --model <pinned-primary-model-snapshot> \
  --precision full \
  --output-dir <primary-judge-output>
```

Use separate model families—not three aliases of one checkpoint—for the three
slots. Do not choose models after comparing their agreement with LLM2Vec.

Once responses exist, compile independent-judge consensus, embed the exact
natural texts with frozen LLM2Vec, and fit the map:

```bash
python3 analysis/scripts/fit_semantic_readiness_map.py compile-labels \
  --tasks <readiness_label_tasks_blinded.jsonl> \
  --responses <primary/judge_responses.jsonl> <replicate-a/judge_responses.jsonl> <replicate-b/judge_responses.jsonl> \
  --output-dir <compiled_labels>

srun -n1 --gres=gpu:1 python3 analysis/scripts/fit_semantic_readiness_map.py embed \
  --corpus analysis/output/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl \
  --embedding-model <base-model-snapshot> \
  --mntp-model <mntp-snapshot> \
  --peft-model <simcse-snapshot> \
  --output-dir <embeddings>

python3 analysis/scripts/fit_semantic_readiness_map.py fit \
  --corpus analysis/output/semantic_readiness_corpus_v3/semantic_readiness_corpus.jsonl \
  --consensus <compiled_labels/readiness_consensus.jsonl> \
  --embeddings <embeddings/semantic_readiness_llm2vec_embeddings.npz> \
  --output-dir <readiness_map>
```

The fitted artifact contains a continuous ridge direction and its four ordinal
level-set planes, plus a multivariate supervised coefficient matrix for the
four rubric dimensions. Singular values and the first-component share test
whether these dimensions collapse to one line. The first two supervised left
singular vectors are retained as a candidate surface when the rank-one claim
fails. Confirmation data are projected without refitting.

| Partition | Contexts | Plans | A1 values per block | Stimuli |
|---|---:|---:|---:|---:|
| Development | 40 | 2 | 7 | 560 |
| Locked confirmation | 24 | 2 unseen | 7 | 336 |
| Total | 64 | 4 | — | 896 |

Each `(content context, realization plan)` block contains endpoints 0 and 1
plus one stable random point in each of five interior strata. Thus the design
covers the full range without forcing equal distances or a shared seven-point
grid. Tone, syntax, clause order, directness, formality, and response form are
fixed within a block.

The generator writes only an objective clause containing `[CONTENT]` exactly
once. The compiler owns the content payload and instruction wrapper. Structural
checks reject literal-content generation, numeric coordinate leakage, source
preference, price, geography, urgency, sentiment, popularity, authority, and
other off-axis criteria.

## Representation and estimator

Every valid stimulus is embedded in three frozen views:

1. `intent-only`: the generated objective with `[CONTENT]`;
2. `content-masked`: the complete instruction with `[CONTENT]`;
3. `full-content`: the complete instruction with the literal content payload.

The primary estimator is blocked least squares over all views:

```text
v = sum((A - mean_block(A)) * (z - mean_block(z)))
    / sum((A - mean_block(A))^2)
```

Separate view coefficients are retained. A shared direction is rejected if the
views disagree; they are never forced to be orthogonal or declared equivalent.
Projection uses each block mean as its baseline and records raw coordinate,
absolute assigned-coordinate error, and matched off-axis residual. Embedding
rows are L2 normalized before estimation.

## LLM-only semantic validation

Human perception studies are not part of this protocol. At least three pinned,
heterogeneous LLM judge families must independently receive blinded tasks. The
public task files contain no A1 values, expected winners, generator identity,
block identity, or embeddings. A separate codebook is kept inaccessible to the
judge process.

Collect five 1-to-7 ordinal rubric responses, a secondary 0-to-100 readiness
score, and pairwise judgments for adjacent, endpoint, sampled nonadjacent, and
same-A1 cross-plan comparisons. Estimate an ordinal consensus and a
Bradley--Terry ordering with judge and presentation effects. Judges validate
the randomized construct; they do not define the embedding direction.

## Predeclared green lights

| Gate | Required evidence |
|---|---|
| QF1 stimulus integrity | 896/896 present; matched blocks; no leakage, duplicates, or truncation |
| QF2 LLM semantic consensus | inter-family pairwise agreement >= 0.80; each family vs other-family consensus >= 0.75; ordinal/pairwise latent correlation >= 0.80; consensus/A1 correlation >= 0.85; one-factor common variance >= 0.80 |
| QF3 one-dimensional geometry | treatment first-component share >= 0.90; bootstrap lower bound >= 0.85; median local/global cosine >= 0.85; each view/shared cosine >= 0.90; no 2D, spline, or small nonlinear held-out gain >= 0.03 |
| QF4 locked transfer | for every view, A1/coordinate Spearman >= 0.85; judge/coordinate Spearman >= 0.80; pairwise accuracy and adjacent monotonicity >= 0.80; calibrated MAE <= 0.12; no reversed domain slope |
| QF5 negative controls | lexical-adversarial accuracy >= 0.70; fixed-A1 nuisance shifts < 0.10 calibrated units; content and source policy fail to predict or move A1 beyond frozen control bounds |
| QF6 stability | 2,000 context-cluster bootstraps; 95th-percentile direction angle <= 15 degrees; split-half cosine >= 0.90 in 95% of splits; rank stability >= 0.90 |

Thresholds are frozen before locked confirmation. `Inconclusive` is a stop and
does not authorize threshold weakening. Synthetic smoke results are always
marked `scientific_result: false`.

## Implemented milestone

The current code implements the smallest cluster-portable Stage A foundation:

- frozen 64-context/4-plan population specification;
- deterministic irregular A1 assignment and request identities;
- generator prompt and objective-only compiler;
- structural checks and three representation views;
- blinded ordinal and pairwise task export plus private codebook;
- shared and view-specific blocked direction estimation;
- raw coordinate, off-axis residual, monotonicity, collinearity, and tortuosity;
- development-only fitting with untouched confirmation evaluation;
- a deterministic CPU-only fake smoke mode that cannot be reported as science.

Real generator execution, real LLM judging, judge-consensus fitting, negative
controls, nonlinear alternatives, calibration, and bootstrap gate adjudication
remain later testable milestones. No expensive local inference is launched by
this implementation.

## Commands

Prepare the 896 external generation requests:

```bash
python3 analysis/scripts/run_query_free_axis_pilot.py prepare \
  --output-dir runs/query_free_axis/<run_id>/generation
```

After a pinned generator produces one JSONL row per `request_id`, compile and
audit the stimuli:

```bash
python3 analysis/scripts/run_query_free_axis_pilot.py compile \
  --objectives-jsonl <objectives.jsonl> \
  --output-dir runs/query_free_axis/<run_id>/stimuli
```

Export blinded judging tasks:

```bash
python3 analysis/scripts/run_query_free_axis_pilot.py export-judging \
  --stimuli-jsonl <query_free_stimuli.jsonl> \
  --output-dir runs/query_free_axis/<run_id>/judging
```

Embed on HoreKa with pinned local snapshots:

```bash
srun -n1 --gres=gpu:1 python3 analysis/scripts/run_query_free_axis_pilot.py embed \
  --stimuli-jsonl <query_free_stimuli.jsonl> \
  --embedding-model <base-model-snapshot> \
  --mntp-model <mntp-snapshot> \
  --peft-model <simcse-snapshot> \
  --output-dir runs/query_free_axis/<run_id>/embeddings
```

Fit development and score locked confirmation without refitting:

```bash
python3 analysis/scripts/run_query_free_axis_pilot.py fit \
  --stimuli-jsonl <query_free_stimuli.jsonl> \
  --embeddings-npz <query_free_llm2vec_embeddings.npz> \
  --output-dir runs/query_free_axis/<run_id>/fit
```

CPU contract smoke test:

```bash
python3 analysis/scripts/run_query_free_axis_pilot.py fake-smoke \
  --output-dir /tmp/query-free-axis-smoke
```

Every scientific cluster run must use committed code and record the exact Git
SHA, model revisions, configuration, seeds, Slurm job, resources, logs, and
artifact hashes. Stage B cannot begin until all Stage A gates—not merely the
synthetic smoke test—are green.

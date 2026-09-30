# SI-v3 pilot, answer style and judge choice — 2026-09-30

Follows [state of work](2026-09-30_state-of-work_handoff.md). Cluster facts are
pasted evidence from 30 Sep; recheck before acting.

## 1. Qwen bouts (HoreKa)

30 Sep ~05:00 (`sacct` since 27 Sep, all job types): 244 COMPLETED, 14 FAILED,
3 TIMEOUT, 3 CANCELLED, 37 RUNNING, 259 PENDING; account queue 296/295.
~640 of 893 bouts left. At ~37 nodes ≈ 2.6 days → done ~2–3 Oct (+ leftover
sweep). **Failures rose from 4 to ~14:** inspect `horeka-qwen-bouts.html` step 5
before approving any sweep. Sender `qwen-sender-all` still unconfirmed.
Limits: 50 running per user (reached 28 Sep), 295 queued per account; priority =
age + QOS (fair-share weight 0).

## 2. Nemotron SI-v3 pilot (pin `961ad0b`), job 5171199, hkn0401

Page `nemotron-si-v3-pilot20.html`: 20 new cells (10 Qwen + 10 Llama, seed
2026093010, excludes earlier 25), each generator a batch-style `run.sh` inside
one salloc; pass 1 + identical pass 2. Root:
`$W/reviews/nemotron-si-v3-pilot20-961ad0b-2026093010/` (`report-pass1/`,
`report-pass2/`: prompt, full answer, sources, judgments per cell).

| | Result |
|---|---|
| Validity | 4 runs `EXIT=0`; SI 53/53 and J1 10/10 per run; all `stop` |
| Time | first run 11.6 min (cold), then 6.6 min each; judging 54 s (Qwen) / 40 s (Llama) per 10 cells; ~1.2 req/s at concurrency 4 (~4,200/node-h) |
| Repeat | Llama 53/53 identical; Qwen 47/53 (4 off by 1, one top-source 5↔1 swap): batching nondeterminism |
| Grades | Qwen {0:3,1:5,2:20,3:24,5:1}; Llama {0:11,1:1,2:21,3:15,4:5}: compressed at 2–3, many ties, few 4–5 |

Problems: under-grading of clear support (Trello cell: 2 for near-verbatim
support); concatenated title lists chosen as evidence; unstable top source in
meta answers; `Snippet S1` wording not masked (only `(S1)`/`[S1]`); search ads
(`bing.com/aclick`) appear as sources. **Do not scale**: revise rubric anchors
(4–5), passage presentation and masking first; raise concurrency in the benchmark.
Page bug fixed locally: final timing heredoc renamed `PY_TIMING` (had reused `PY_TIME`).

## 3. Answer style (main open issue)

The generator instruction never asks for a user-facing answer. Word-pattern check
(3,000 Qwen cells): mentions snippets/S-IDs 93 % (Parallel) / 72 % (Reactive);
"evidence lacks the answer" 55 % / 33 % (overcounts). In samples, refusal-style
answers cluster on action-type prompts. This is observed behaviour under a fixed
instruction, not a corpus defect; it limits source-importance claims (enumeration
makes support circular).

- 200-answer sample (100 Qwen, 100 Llama, seed 2026093001) uploaded to private
  `geodml-experiment-v2-paper-private/analysis-samples/answer-style-200-20260930-0546/`
  and downloaded to Mac `~/Hamburg/GEODML_Unified/answer-style/`.
- Labelled by a subagent (one annotator, no second coder): `answer-style/codebook.md`,
  `labels.jsonl`, `summary.md`, `summarize_labels.py`. Direct answer: Qwen 56 %,
  Llama 91 %; evidence lacks answer 31 % / 9 %; narrates snippets 37 % / 0 %;
  answers a different question 3 % / 19 %; median length 1,062 / 407 chars.
  Drivers: action-ready prompts (Qwen 46 % evidence-lacks vs 10 % info) and
  Parallel-Expansion (Qwen 42 % vs 19 % Reactive). Evidence-talk is mainly a Qwen
  pattern; Llama's issue is short or off-target answers (tool lists for how-to).
- Corpus-wide classifier candidate: **Jev** (TypeSafe decision model; $0.042/M
  input, free output): all ~622k answers ≈ $16–17; validate on a few hundred first.
- Option on the table: small natural-answer generation pilot (new versioned
  instruction, same prompts/evidence) as a second condition. Not approved.

## 4. Judge assessment and Jev

**Nemotron (SI-v3) today:** reliable mechanics (212/212 valid, fast, Llama repeat
53/53) and usable as a yes/no support filter and for J1; **not yet reliable as a
fine-grained source ranker** (~80 % of grades are 2–3, misses near-verbatim
support, picks concatenated title lists, one unstable top source). Candidate fixes
(new version, test on the same 20 cells): 0/1/2 scale (none/minor/central), sharper
anchors with one example per grade, split title lists and drop boilerplate/ads,
pairwise "which supports more" for top positions, stratify by answer style,
compare against a stronger reference model.

**Jev as judge (proposal, not decided):** decision model returning score/choice
with probabilities; all ~4.1M SI+J1 requests ≈ 5–6B input tokens ≈ $210–260, no
GPU. Against: no passage provenance, brand new and unvalidated, closed API
(pin `jev-1.13`), data leaves the cluster, and it changes the "fixed judge"
decision. Recommended: head-to-head on the 106 pilot pairs and the 200 style
labels (Jev vs Nemotron vs a stronger reference) before any switch; likely outcome
a hybrid (Nemotron auditable primary, Jev second opinion/classifier). Awaiting
Valerian: build the client? direct API or OpenRouter?

**Generator instruction decision:** prompt for Astra at
`~/Hamburg/GEODML_Unified/astra-qwen-steering-prompt.md` (options A keep, B
change mid-run [rejected], C second versioned condition for both generators on a
subset, D regenerate, E rewrite). Lean: A + C. Qwen run continues unchanged.

## 5. Other state

- Llama copy on HoreKa (`$W/llama-hf/dataset`) was behind (307,512 vs 310,374 on
  the Hub); `temp.html` step 1 re-pulls only missing files. Confirm `PULL_EXIT=0`.
- JUPITER: Shell A archive still running (see state-of-work handoff).
- Preference saved: run batch-style `run.sh` inside one approved salloc and fill
  the hour, instead of separate sbatch jobs.

## 6. Next, in order

1. Astra's answer on the generator instruction (A / A+C); decide the second condition.
2. Jev head-to-head on the pilot pairs and 200 style labels (if approved), plus a
   revised Nemotron version (simpler scale, cleaner passages) on the same 20 cells.
3. Corpus-wide answer-style labels (Jev or Nemotron), validated on the 200.
4. Qwen: failed-bout review, then leftover sweep after all bouts end.
5. Throughput benchmark (higher concurrency) only after the judge protocol is fixed.

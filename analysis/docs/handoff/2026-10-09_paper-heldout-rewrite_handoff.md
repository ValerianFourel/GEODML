# Paper rewritten on the held-out results of both generators; stage figure; generated numbers

Written 2026-10-09 after `2026-10-09_fullrun-results-intent_handoff.md` (read it first for the run and the data). This session
updated the ACL paper (`ARR_ACL_CycleOct2026/paper_v2`, outside git) and added the code that generates its numbers and its new figure.

Evidence labels: **F** = fact seen in a file or a command's output; **P** = plan; **A** = assumption; **H** = estimate.

## 1. What changed in the paper [F]

- Headline numbers moved from Qwen exploration (237 keywords) to both generators on the 733 held-out keywords (PREREG B3).
  Table 1 now holds, per cell: the admission share of the cited-intent slope with its interval, the alignment share arising at the
  shortlist, the generator's own share ($I \to K$), $\Delta_{\mathrm{gen}}$ with its 90% interval, the intent and topic-plus-slot
  Pratt shares, the links-kept slope (all keywords) and the C1/C2 verdicts.
- New Figure 3 (`figures/fig10-intent-stages.pdf`): every stage on the prompt scale. (a) the agent's queries and the answer text
  against the prompt's own position; (b, c) the page sets $R_0$, $R$, $P$, $K$ per generator with the own-keyword oracle ceiling
  (held-out deciles). Queries, answers and $R_0$ come from the intent-stages curves on the native $z$ scale and are carried onto the
  percentile scale through the frozen prompt map rebuilt from the 207,354 answer coordinates; the map reproduces the exact page curves
  within 0.0007 (checked for $R$, $P$, $K$, both generators).
- Keyword sentence reversed (§1, §5.2): prompts stop naming their keyword along $x$ (−0.74 to −0.79 within keyword); among prompts that
  name it, the queries' keyword share barely moves (Qwen Reactive 0.66 → 0.61).
- §5.3: Qwen's selection does not respond to intent (Δ_gen −0.0003 / −0.0026, inside ±0.015); Llama narrowed (Δ_gen +0.0033 Reactive,
  +0.0141 Parallel = 15% of β_K; generator's own share 30% under Parallel).
- §5.4: domain size is associated with no step on the held-out keywords (retrieval 0.90–1.10, shortlist 1.02–1.10, ranking 0.98–1.09);
  the exploration-only retrieval association for Qwen (1.46 / 1.40) did not replicate; P3 includes 0 everywhere.
- §6: supply narrowed to a ceiling (0.76% of rows at $u \ge 0.8$; ceiling gap 0.37–0.43 at $x \ge 0.9$; utilisation of the own-row
  oracle 0.20–0.28; tercile contrasts at or below 0).
- §5.2: the funnel study's P1, P2, P4 did not replicate (four-cell rule); stated in one sentence with the table in Appendix I.
- Appendix: A3 rebuilt (takeaways on held-out numbers; generated tables for 4 stage models, keep and order models, decomposition,
  verdicts, predictions, intent in text); the stubs `app:protocol` and `app:judges` written; A2's topic table re-sourced to the
  held-out Qwen fits. The buggy complete-case shortlist odds ratios (5.29 / 4.80) are gone.
- Limitations: held-out and Llama no longer pending; reranker and corpus scope; page effects observational; decisive fixed-shortlist
  test not run; controls for wording.
- Page budget: `fig:cited`, `tab:stages` and `tab:shares` moved to the appendix; Table 1 compacted; §1–§7 trimmed by about 900 words.

Gate (`bash check_paper.sh`): **GATE: PASS**, 38 pages, 0 undefined references, 0 overfull boxes, Conclusion on page 8, Limitations
from page 9, hash scan clean, 0 pending macros, no banned words. [F]

## 2. Code added on this branch [F]

| File | What |
|---|---|
| `analysis/scripts/paper_numbers_fullrun.py` | Reads the bundle (`steelman-confirmation/*.json`, `paper-results/funnel-confirmation/results.json`, `decisions-confirmation/decisions.json`, `verdicts.json`, `intent-stages/results.json`, exploration files for the replication sentence) and writes `numbers-heldout.tex` (1,123 macros, each with a `% src:` line), `A3-generated-tables.tex` (12 tables as `\Athree*` macros) and `dossier.md`. `--coordinates` adds the first/last-bin stage means on the prompt scale ([A], mapped). Macro names carry no digits (`Aone` … `Ctwo`; cells `LP LR QP QR`). |
| `analysis/scripts/acl_figures.py` | `prompt_scale_map`, `_mapped`, `fig10_intent_stages`, `read_coordinates`; `render --coordinates --supply` draws `fig10-intent-stages`. |
| `analysis/tests/test_acl_figures.py` | New test: fig10 renders from synthetic curves, the map recovers a logistic prompt scale, a non-monotone map is refused. |
| `analysis/tests/test_paper_numbers_fullrun.py` | Formatters, digit-free block names; with the bundle present, all generated macro names are valid control sequences and the 12 tables balance. |

Tests: `python3 -m pytest -q analysis/tests/test_acl_figures.py analysis/tests/test_paper_numbers_fullrun.py` → 9 passed. [F]

Commands to regenerate are in `paper_v2/HANDOFF.md` §2 (bundle `~/Hamburg/geodml-inputs/fullrun-run2`, output into `paper_v2`).

## 3. Valerian's questions during the session

- *"We need a drawing of intent at each part of the pipeline, Reactive Loop and Parallel Expansion, and whether it saturates."*
  Done as Figure 3 from the existing results. What it shows [F]: the agent's queries follow the prompt (Qwen 0.26 → 0.83 across the
  bins of $x$, Llama 0.27 → 0.82); the pages the frozen search returns for them stay within 0.38–0.45; the shortlist and cited set tilt
  a little further (cited 0.39 → 0.48 Qwen, 0.38 → 0.50 Llama); the answer text recovers much of the prompt's position (Qwen 0.38 →
  0.65 and flattening past $x \approx 0.5$, Llama 0.38 → 0.79). The own-keyword oracle ceiling itself flattens near 0.55–0.58 for
  $x \ge 0.6$: saturation is a property of the page supply, not of the generator. Per method, only the slopes are available (Appendix I,
  intent-in-text table; Table 3 of the appendix for the page stages); the per-bin curves are pooled over methods (next point).
- *"We need GPU jobs to measure the intent of the sub-queries at each step."* Not needed: the full run already embedded the
  304,567 unique agent queries and the 207,354 answers with both LLM2Vec encoders (tasks `embed-queries-*`, `embed-answers-*`,
  `intent-analyze`; validity: views agree at Spearman 0.86 on queries, 0.83 on answers) and placed every stage on the axis
  (`paper-results/intent-stages/results.json`: slopes per model × engine × method, curves per model × engine in 20 bins). What is
  missing is only per-method *curves* for Q and A; that is a CPU-only re-aggregation of the stored per-answer stage values on HoreKa
  (the merged embeddings and the intent-stages inputs are in `$FR/run2`), about one node-hour [H], no GPU. The per-answer values are
  not on the Mac.

## 4. Open items (P)

1. Per-method Q/A curves (CPU job on HoreKa, see above) if the figure should split methods.
2. Regenerate the pipeline figure (fig3): image text still says "Section 5", stage letters Q and A, "pinned".
3. Human rating of the axis (about 200 prompts, agreement reported).
4. `analysis/steelman/report.py` template still prints "No held-out result exists" and a "Qwen, exploration" heading on the held-out
   split (numbers right, text wrong; noted in the previous handoff).
5. `fleet/CONTRACT.md` describes the first draft's float plan and the 237-keyword scope; the paper now follows `HANDOFF.md`.
6. Title choice; Lee 2026 / Pratt 1987 citations; unrecorded prompt-validation model and Llama-3.3-70B revision (carried over).

## 5. Cluster facts carried over, not rechecked this session [F from the previous handoff]

Run2 ledger 300/300 done; archive on the Hub at `derived/fullrun-run2/fullrun-run2-20261009T050224Z.tar.gz`; HoreKa copy under
`$W/reviews/page-readiness-20261004/fullrun-v1/run2`; Gemma SI-v4 Qwen bouts partly unfinished (11 budget-exhausted, 47 not started).
No cluster command was run today.

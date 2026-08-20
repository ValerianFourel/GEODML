# Two-embedding readiness robustness battery

Run this battery before using the readiness coordinates to design a large prompt
population. It evaluates the frozen Qwen and Mistral LLM2Vec maps on identical
development and confirmation items. It does not refit either 4096-dimensional
direction and does not change the label policy.

## Tests

The battery verifies input hashes, judge slots, label policy, frozen split, and
consensus equality. It then reports:

- direct frozen-map Spearman, Pearson, R-squared, and MAE;
- one-axis versus two-axis held-out regression across overall readiness and all
  four rubric dimensions;
- a small polynomial alternative;
- an additive cubic regression-spline alternative with development-only
  cross-validation of its ridge penalty;
- axis-1 spline monotonicity;
- development-fitted Procrustes alignment evaluated only on confirmation;
- 1,000 confirmation bootstraps;
- 200 development-label permutation controls;
- leave-one-source-out training followed by confirmation evaluation for each
  sufficiently large source.

The frozen checks use these diagnostic thresholds:

- scalar Spearman bootstrap lower bound at least 0.60 in both views;
- cross-view scalar lower bound at least 0.75;
- aligned axis-1 lower bound at least 0.70;
- aligned axis-2 lower bound at least 0.50;
- positive axis-2 macro R-squared gain in both views;
- spline macro R-squared no more than 0.03 below linear in either view;
- direction-free spline monotonicity at least 0.75 in both views;
- positive leave-one-source-out Spearman in at least 70% of eligible sources;
- one-sided label-permutation p-value at most 0.05 in both views.

These are exploratory robustness checks, not a preregistered causal gate. A
failure is reported as inconclusive and must not be fixed by weakening thresholds
after seeing downstream reranking outcomes.

## Command

```bash
python analysis/scripts/build_readiness_hf_dataset.py robustness-battery \
  --reference-dir "$QWEN_MAP_ROOT" \
  --candidate-dir "$MISTRAL_MAP_ROOT" \
  --output-dir "$BATTERY_ROOT" \
  --bootstrap-replicates 1000 \
  --permutation-replicates 200 \
  --minimum-source-items-per-split 50 \
  --random-seed 20260820 \
  --git-commit-sha "$ROBUSTNESS_COMMIT"
```

This stage uses only stored coordinates and labels, so it is CPU-only. Its
output includes `readiness_robustness_battery.json`, a compact Markdown report,
and an immutable manifest.

## Generated-question replication

After generating a small candidate bank, embed and project the identical text
through the Qwen and Mistral maps separately with `project-candidates`. Then use
`compare-projections` with the battery directory. The original development
corpus supplies the alignment; generated questions do not refit it.

This provides an honest out-of-sample check of whether both representations put
new questions in similar semantic locations. Raw Qwen and Mistral coordinates
must not be compared directly because their scales and orientations differ.

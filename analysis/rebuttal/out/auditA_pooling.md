# Audit A — Pooling implementation (evidence-cited)

**Answer: the pooled fit is NOT naive stacking. Engine, LLM model, pool depth,
and both prompt-condition axes enter the nuisance covariate set X as five
additive binary indicators (no explicit interaction terms with other
covariates; nonlinear interactions are learnable inside the LightGBM
nuisances, but the target parameter θ is a single pooled coefficient).**

Evidence, all from `scripts/dml_canonical.py`:

1. Cell dummies are constructed in `add_cell_dummies()` (scripts/dml_canonical.py:308-324):

```python
def add_cell_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add binary cell dummies, return df + list of dummy column names."""
    cell_cols = []
    if "search_engine" in df.columns:
        df["cell_engine_searxng"] = (df["search_engine"] == "searxng").astype(int)   # line 312
        cell_cols.append("cell_engine_searxng")
    if "llm_model" in df.columns:
        df["cell_model_qwen"] = (df["llm_model"] == "Qwen2.5-72B-Instruct").astype(int)  # line 315
        cell_cols.append("cell_model_qwen")
    if "pool_size" in df.columns:
        df["cell_pool_50"] = (df["pool_size"] == 50).astype(int)                     # line 318
        cell_cols.append("cell_pool_50")
    if "variant" in df.columns:
        df["cell_variant_biased"] = df["variant"].str.startswith("biased").astype(int)  # line 321
        df["cell_variant_rag"] = df["variant"].str.endswith("_rag").astype(int)         # line 322
        cell_cols += ["cell_variant_biased", "cell_variant_rag"]
```

2. The dummies are attached to both estimation frames before any fit, in
`main()`: `pool, _ = add_cell_dummies(pool)` (scripts/dml_canonical.py:414) and
`admitted, _ = add_cell_dummies(admitted)` (scripts/dml_canonical.py:421).

3. They are appended to the confounder set for every specification, in
`run_dml_for_outcome()` (scripts/dml_canonical.py:359-360):

```python
    cell_cols = [c for c in data.columns if c.startswith("cell_")]
    X_base = CONFOUNDERS + cell_cols
```

`X_base` is then passed as the `X_cols` argument of every `plr_estimate()`
call — Spec A (line 367) and the headline Spec B (line 384) — so the LightGBM
nuisances g(X) for both the outcome and the treatment always condition on
engine, model, pool depth, prompt bias, and RAG status.

4. The prompt condition is encoded as TWO indicators, not four: the 4
prompt–evidence conditions {biased, neutral, biased_rag, neutral_rag} are
factorized into `cell_variant_biased` (prompt axis) × `cell_variant_rag`
(evidence axis) (lines 321-322). This spans all 4 conditions since the design
is a full 2×2 cross; the biased×rag interaction cell is not separately
indicated in X, though tree nuisances can represent it.

5. No engine×covariate (or model×covariate) interaction features are
constructed anywhere in the estimation script — `grep -n "interact"
scripts/dml_canonical.py` returns nothing, and the only engineered columns are
the five `cell_*` indicators above. Effect heterogeneity across engines is
instead reported via split-sample slices (`iter_slices()`,
scripts/dml_canonical.py:330-350: `ENG:ddg` / `ENG:searxng`, `MOD:*`,
`POOL:*`, `VAR:*`), which re-estimate θ within each cell.

**One-paragraph summary for the rebuttal:** In the pooled specification the
search engine is included in the nuisance covariate vector X as a binary
indicator (`cell_engine_searxng`, scripts/dml_canonical.py:312), as are the
LLM (`cell_model_qwen`, line 315), SERP pool depth (`cell_pool_50`, line 318),
and the two prompt-condition axes (`cell_variant_biased` /
`cell_variant_rag`, lines 321-322); the list is concatenated to the 29
confounders for every fit (`X_base = CONFOUNDERS + cell_cols`, line 360).
Pooling is therefore covariate-adjusted stacking: both DML nuisance functions
condition on the design-cell indicators (and, being gradient-boosted trees,
can represent indicator×covariate interactions internally), while the
structural coefficient θ is constrained to be common across cells. No explicit
interaction features are engineered, and per-engine/per-model heterogeneity is
assessed by the split-sample slice estimates rather than interaction terms.

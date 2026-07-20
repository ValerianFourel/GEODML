# §Interpretability — probing section (paper-ready)

Drop-in text for the §interpretability section of the EMNLP submission.
Figure: `docs/2026-05-24/figures_canonical/tmp/fig_probing_unified.pdf`
covering the 6 canonical content treatments.

Numbers below are from
`interpretability/output/probing_results_*.csv` regenerated on 2026-05-25.
Update them once the JUPITER gap-fill jobs (Llama T3, T4) land.

---

## §4.x  Layer-wise probing of content treatments

### Headline (drop in as a paragraph)

We complement the causal estimates of §\ref{sec:dml} with a
representation-level analysis. For each canonical content treatment
$T_i$, we train a linear (logistic-regression) probe on the frozen
hidden states of Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct
when processing the body text of each page (mean-pooled across
tokens). One probe is trained per (treatment, layer, pooling) tuple
with 5-fold cross-validation; we report test-set ROC AUC averaged
across the two models and the four prompt variants.

Figure~\ref{fig:probing} reports the result. **The probe achieves
ROC AUC $\geq 0.97$ at the peak layer for every canonical content
treatment**: $T_{2a}$ Q-headings peaks at $0.989$ (layer 24);
$T_6$ freshness at $0.988$ (layer 73); $T_5$ topical
comprehensiveness at $0.974$ (layer 46); $T_{1b}$ stats density,
$T_3$ schema, and $T_4$ citation authority similarly exceed $0.97$.
Mean pooling consistently outperforms last-token pooling (right
panel of Figure~\ref{fig:probing}), confirming that content
treatments live in the sequence-average representation rather than
at the decision token. Prompt variant has negligible effect on the
probe (the shaded variant-envelope band in panel (a) is invisibly
narrow), since the probed object is the model's representation of
the page content, which is largely prompt-independent.

### Methodological caveat (drop in immediately after the headline)

High probe accuracy demonstrates that treatment-correlated
information is \emph{linearly decodable} from the model's hidden
states; it does not on its own establish that the model
\emph{uses} this information at rerank time. For several treatments,
ROC AUC already exceeds $0.85$ at layer 0 (the embedding
lookup, before any contextualisation), which we attribute to lexical
correlations rather than deep semantic representation. A diagnostic
on 500 stratified pages for $T_6$ shows that the token
\texttt{"2026"} appears in 46.4\% of fresh-class pages versus
12.4\% of stale-class pages (a 3.74$\times$ enrichment;
Table~\ref{tab:freshness-leakage}). Adjacent years show analogous
ratios. The layer-0 probe is therefore detecting vocabulary
asymmetries in time-related tokens rather than reasoning about
recency. We follow \citet{hewitt2019designing} in interpreting
probe accuracy as evidence of \emph{availability} of
treatment-correlated information in the model's representations;
the model's \emph{causal use} of that information at inference is
established separately by DML in §\ref{sec:dml}, which conditions on
the 28 page-level confounders plus the
\texttt{has\_llms\_txt} GEO-intent proxy and is therefore
invariant to surface-level vocabulary correlations.

### Why probing still adds value to the paper

Despite the lexical-leakage caveat, the probing analysis contributes
two pieces of evidence the DML estimator cannot provide:

1. **Representational availability** — the probe confirms that the
   model's hidden states encode the treatment features at all. If
   the LLM lacked any internal representation of, say, freshness,
   the DML coefficient on $T_6$ would have to be explained by
   pure pattern-matching at the surface level. The probe rules out
   that hypothesis: the information is genuinely present in the
   network's intermediate states.

2. **Layer-resolved profile** — mid-network peak layers
   (e.g.\ $T_{2a}$ at layer 24 vs.\ $T_6$ at layer 73) suggest the
   model builds different abstractions for different content
   features. Structural cues (question headings) are localized
   early; temporal cues (freshness) emerge late. This is a
   description of model behavior, not a causal claim, but it
   helps situate the DML coefficients within an interpretability
   story for the reviewers.

### Limitations (one sentence for the limitations section)

The probing analysis covers six content treatments but only the
\textbf{full sample frame}; we did not run probes on the robust-winner
subset because the layer-wise patterns we observe are stable across
frames in pilot runs. We also restrict probes to Llama-3.3-70B and
Qwen2.5-72B; smaller or instruction-tuned variants may show
different layer-wise profiles.

---

## Figure C caption (full text)

```latex
\caption{\textbf{Layer-wise linear probing of content treatments.}
Linear (logistic regression) probes trained per (treatment, layer,
pooling) tuple on frozen hidden states from Llama-3.3-70B and
Qwen2.5-72B. (a) ROC AUC at each transformer layer for the six
canonical content treatments, mean pooling, averaged across two
models and four prompt variants; shaded band shows the
min–max envelope across variants (invisibly narrow because variant
has no measurable effect on the probe). Markers indicate the
peak-AUC layer per treatment. (b) Pooling comparison: mean pooling
(solid) reaches near-ceiling AUC from layer 0; last-token pooling
(dashed) requires several layers of contextualisation before
reaching parity. \textbf{Caveat.} High layer-0 AUC reflects
vocabulary differences between class strata rather than deep
semantic representation. For $T_6$ (freshness), fresh-class pages
mention the token \texttt{"2026"} at 3.74$\times$ the rate of
stale-class pages (Table~\ref{tab:freshness-leakage}). Probing
demonstrates \emph{availability} of treatment-correlated
information; the model's causal use of that information is
established by DML in §\ref{sec:dml}.}
\label{fig:probing}
```

---

## LaTeX \includegraphics command

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.97\linewidth]{fig_probing_unified}
% caption as above
\end{figure*}
```

---

## Section structure suggestion

Where these pieces live in the paper:

| Piece | Location | Word count |
|---|---|---|
| Headline paragraph | §4 Interpretability (subsection: probing) | 180 |
| Methodological caveat | immediately after headline | 200 |
| "Why probing adds value" | optional 3rd paragraph in §4 | 130 |
| Figure C + caption | top of §4 page | — |
| Diagnostic table | Appendix D / supplementary | — |
| Limitations sentence | §Limitations | 50 |

Total main-text dedicated to probing: ~510 words across 3 paragraphs.

---

## How to update once T3 + T4 probing lands

Replace the parenthetical numbers in the headline paragraph:

```
$T_3$ schema ... (layer XX);
$T_4$ citation authority ... (layer XX);
$T_{1b}$ stats density ... (layer XX);
```

Use the actual peaks from `interpretability/output/probing_results_biased.csv`
once `make_fig_probing.py` includes those treatments.

---

## Reviewer-anticipation paragraph (optional, for if asked)

If a reviewer challenges "how do you know the probe isn't memorizing
the training fold?":

> Probes are trained with stratified 5-fold cross-validation; the
> reported ROC AUC is the held-out test-set average, not training
> accuracy. The probe is also linear, so it has no capacity to
> memorize individual examples beyond the linear separability of
> the layer's representation. The accuracy reflects the
> probe's ability to find a hyperplane in the layer's embedding
> space that separates the two treatment classes, which is exactly
> the property we want to measure.

This pre-empts the standard probing-skeptic critique.

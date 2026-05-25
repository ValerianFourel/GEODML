# Probing — methodological caveat (paper-ready text)

Drop-in for the §interpretability / §probing section of the EMNLP submission.
Two versions: a tight version for main text, and an expanded one for the
appendix if you have room. All numbers are drawn directly from the
diagnostic `docs/2026-05-24/freshness_leakage_diagnostic.md`.

---

## Main-text version (≈110 words)

A linear probe trained on frozen hidden states of Llama-3.3-70B (and
Qwen2.5-72B) achieves ROC AUC $\geq 0.85$ on every canonical content
treatment from layer 0 onward, peaking at $\geq 0.97$ in mid-network
layers (Figure C). High layer-0 accuracy, however, signals
*lexical separability* rather than deep semantic understanding: a
diagnostic on a 250 + 250 stratified sample shows that 46% of
fresh-class pages contain the literal token \texttt{"2026"} versus
12% of stale-class pages (a 3.7$\times$ concentration), and
analogous gaps for adjacent years. The token-embedding layer trivially
separates such vocabulary distributions. We follow
\citet{hewitt2019designing} in interpreting probe accuracy as
demonstrating \emph{availability} of treatment-correlated information
in the model's representations, not its causal role at inference;
the latter is established by DML (\S\ref{sec:dml}), which conditions
on 28 page-level confounders plus the \texttt{has\_llms\_txt}
GEO-intent proxy and is invariant to surface-level vocabulary
correlations of this kind.

---

## Appendix version (≈260 words, with full diagnostic)

**Lexical-leakage diagnostic for layer-0 probe accuracy.**  The linear
probes reported in Figure C reach ROC AUC $\geq 0.85$ on every
canonical content treatment from layer 0 onward—that is, from the
mean-pooled token-embedding output, before any attention or
feed-forward computation. This is a striking number, but it is not
on its own evidence that the model has rich semantic representations
of the treatments. A linear classifier on mean-pooled embeddings can
exploit any feature that systematically differs in the vocabulary
distribution of positive vs. negative pages.

To make this concrete for T6 (freshness), we drew a stratified sample
of 250 fresh-class and 250 stale-class pages (median split on the
\texttt{treat\_freshness} score), loaded each page's body text via
the same digest pipeline used by the probe, and counted
year-token presence. Table~\ref{tab:freshness-leakage} reports the
results. The token \texttt{"2026"} (the scrape year) appears in 46.4\%
of fresh-class pages but only 12.4\% of stale-class pages, a
3.74$\times$ concentration. Adjacent years show analogous gaps in
both directions (\texttt{"2025"}: 1.67$\times$ fresh-enriched;
\texttt{"2015"}: 0.33$\times$, i.e.\ stale-enriched). The
median-split label is therefore largely a lexical detector for
recent-year tokens, and a layer-0 probe can recover it without any
contextualisation.

We adopt the framing of \citet{hewitt2019designing}: probe accuracy
demonstrates the \emph{availability} of treatment-correlated
information in the model's representations, but does not establish
that this information causally drives the model's reranker behavior.
The latter is the object of our DML estimator (\S\ref{sec:dml}),
which is unaffected by surface-level vocabulary correlations because
it conditions on a large covariate set $X$ (28 page-level confounders
plus the \texttt{has\_llms\_txt} GEO-intent proxy) before isolating
the treatment effect.

---

## Diagnostic table (LaTeX-ready)

```latex
\begin{table}[h]
\centering
\small
\begin{tabular}{rrrr}
\toprule
year & stale rate & fresh rate & ratio fresh/stale \\
\midrule
2015 & 2.4\% & 0.8\%  & 0.33$\times$ \\
2016 & 1.6\% & 0.4\%  & 0.25$\times$ \\
2017 & 0.4\% & 1.6\%  & 4.00$\times$ \\
2018 & 2.4\% & 2.0\%  & 0.83$\times$ \\
2019 & 2.0\% & 1.6\%  & 0.80$\times$ \\
2020 & 3.2\% & 2.8\%  & 0.88$\times$ \\
2021 & 2.8\% & 2.8\%  & 1.00$\times$ \\
2022 & 2.0\% & 4.8\%  & 2.40$\times$ \\
2023 & 3.6\% & 5.2\%  & 1.44$\times$ \\
2024 & 5.6\% & 11.2\% & 2.00$\times$ \\
2025 & 13.2\% & 22.0\% & 1.67$\times$ \\
2026 & 12.4\% & 46.4\% & \textbf{3.74}$\bm\times$ \\
\bottomrule
\end{tabular}
\caption{Year-token presence rate in 250 fresh-class + 250 stale-class
pages (median split on \texttt{treat\_freshness}). Fresh-class pages
mention the current year (\texttt{2026}) at 3.74$\times$ the rate of
stale-class pages; older years show the inverse pattern. The
layer-0 probe can exploit this vocabulary asymmetry directly without
contextualisation.}
\label{tab:freshness-leakage}
\end{table}
```

---

## BibTeX entry for Hewitt \& Liang

```bibtex
@inproceedings{hewitt2019designing,
  title     = {Designing and Interpreting Probes with Control Tasks},
  author    = {Hewitt, John and Liang, Percy},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages     = {2733--2743},
  year      = {2019},
  publisher = {Association for Computational Linguistics},
  address   = {Hong Kong, China}
}
```

---

## Caption update for Figure C

Replace the current caption (or append):

> **Caveat on layer-0 accuracy.** High probe ROC AUC at layer 0
> reflects vocabulary differences between class strata rather than
> deep semantic representation. For T6 specifically, fresh-class
> pages mention the token \texttt{"2026"} at 3.74$\times$ the rate
> of stale-class pages
> (Table~\ref{tab:freshness-leakage}). Probing demonstrates
> \emph{availability} of treatment-correlated information in the
> embeddings; the model's causal use of that information at rerank
> time is established by DML in \S\ref{sec:dml}.

---

## Where this caveat should live in the paper

- **One sentence in the main results paragraph** acknowledging layer-0
  is "lexically explainable" (so the reviewer sees you noticed)
- **The appendix version + diagnostic table** in the methodology
  appendix as Section X.Y
- **The caption addition** on Figure C itself so a skimmer doesn't
  misread the layer-0 number

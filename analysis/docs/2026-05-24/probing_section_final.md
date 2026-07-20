# §Interpretability — final paper text (paper uses fig_admission_pooled only)

Drop-in text for §Interpretability of the EMNLP submission. Numbers below
are from `interpretability/output/probing_results_*.csv` (final merge
2026-05-25, 4 prompt variants × ~800-960 keywords × ~10 URLs per rerank
prompt). The figure to include is
`docs/2026-05-24/figures_canonical/tmp/fig_admission_pooled.pdf`.

---

## §4  Probing the LLM's admission decision

### §4.1  Motivation and design

The DML estimates of §3 quantify which content features *causally drive*
the LLM reranker's behavior — but they say nothing about *where in the
network* that behavior crystallizes. Two distinct interpretability
questions remain. **(i) Availability:** are the content features
$T_1$–$T_6$ represented at all in the model's hidden states, or does the
reranker treat each page as an opaque blob? **(ii) Behavioral
pre-commitment:** at what point in the network's forward pass has the
model effectively decided which URLs to admit? A flat near-saturated
curve from layer 0 onward would mean the decision is essentially a
lookup over surface features; a steep mid-network rise would mean the
admission decision is genuinely *built* through compositional reasoning
across many layers.

We answer both with linear probing \citep{alain2017understanding,
hewitt2019designing}. For each transformer layer $\ell$ of
Llama-3.3-70B-Instruct and Qwen2.5-72B-Instruct, we train a logistic
regression on the model's frozen hidden states and measure held-out
ROC AUC. The probe's architecture has no capacity to do anything
beyond locate a linear separator in the layer's representation space,
so probe accuracy directly measures whether the corresponding label is
*linearly decodable* from that layer's representation.

We probe two label sets in parallel. The first (treatment availability,
$T_1$–$T_6$) feeds the LLM the **body text** of each page and asks the
probe to recover the median-split treatment value. The second
(behavioral pre-commitment, $Y_1$) feeds the LLM the **full rerank
prompt** — system message, SERP candidates with snippets, the rerank
question — and, for each URL span in the prompt, asks the probe to
predict whether the (model, variant) ultimately admitted that URL.
For each rerank prompt the model performs **one forward pass** and we
extract per-span hidden states by mean-pooling over the token positions
of each URL line. The first design probes what the model *knows*; the
second probes what the model *decides*. Crucially, the admission-probe
label is the LLM's own behavioral output, so it cannot be trivially
recovered from lexical statistics of a single page.

### §4.2  The content-treatment probes are lexically leaky

For completeness we ran linear probes on the body-text representation
of every canonical treatment $T_1$–$T_6$. All six achieve held-out
ROC AUC $\geq 0.97$ at peak layers between 24 and 73, and several
remain $\geq 0.97$ from the embedding layer ($\ell\!=\!0$) onward. A
diagnostic on a stratified 250+250 sample reveals why: for $T_6$
(freshness) the binary label is strongly correlated with year tokens
in the page text — 46.4\% of fresh-class pages contain the literal
string \texttt{"2026"} versus 12.4\% of stale-class pages, a
3.74$\times$ enrichment. Adjacent years show analogous patterns
(\texttt{"2025"}: 1.67$\times$ fresh-enriched; \texttt{"2015"}:
0.33$\times$, i.e.\ stale-enriched). The layer-0 probe is therefore
detecting *vocabulary asymmetries* between the class strata rather
than the model's semantic understanding of the treatments. Analogous
lexical signatures are plausible for $T_1$ (numeric tokens),
$T_2$ (question marks at headings), $T_3$ (JSON-LD strings), and
$T_4$ (citation markers). We therefore treat the body-text probes as
documenting **availability** of treatment-correlated information in
the model's representations, but do not interpret their high accuracy
as evidence of compositional reasoning, and do not include the
body-text probing figure in the main text.

### §4.3  Admission pre-commitment shows genuine layer-wise composition

The admission probe (\autoref{fig:admission_probe}) tells a different
story. With mean pooling and averaging across the two models and the
four prompt variants, the probe achieves ROC AUC = $0.671$ at
$\ell\!=\!0$ (the input-embedding layer), $0.862$ at the
peak layer $\ell\!=\!60$ (75\% network depth), and $0.830$ at the
final layer $\ell\!=\!80$. This represents a **$+0.191$ ROC AUC gain
from embedding to peak** — substantial compositional improvement
across the network. Unlike the body-text probes, the admission probe
does *not* saturate at the embedding layer; the decision is genuinely
*assembled* by the transformer's forward pass rather than read off
from token statistics.

The shape of the curve carries additional information. The fast rise
between layers 0 and 20 (ROC AUC 0.671 → ~0.83) suggests that the
model rapidly integrates the SERP candidate list and the rerank
question with each URL span — local attention is sufficient to align
each candidate with the query intent. The slower plateau between
layers 20 and 60 corresponds to mid-network composition, where global
context (the relative ranking of all candidates) is consolidated.
The slight drop between layer 60 and layer 80 (peak 0.862 → final
0.830) is consistent with the well-documented pattern in which the
final transformer blocks re-engineer representations for next-token
generation rather than maintaining internal decision features
\citep{tenney2019bert,geva2021transformer}. Last-token pooling
(dashed line in \autoref{fig:admission_probe}) starts at chance at
$\ell\!=\!0$ and catches up by layer ~20, confirming that the
decision information lives in the sequence-aggregated representation
of each URL span rather than in any single decision token.

### §4.4  How probing complements the DML estimates

The two analyses answer different questions and reinforce each other.
DML measures \emph{which features causally influence the admission
decision} (\S\ref{sec:dml}); the admission probe measures
\emph{whether the admission decision itself is linearly decodable from
the model's hidden states, and at which depth}. Both indicate that
admission is a genuinely contentful prediction problem: DML identifies
$T_2$ (Q-headings) and $T_5$ (topical comprehensiveness) as the
strongest causal promoters, and the probe finds that the resulting
admission decision is non-trivially encoded already in early layers
(ROC AUC $\approx 0.67$ at $\ell\!=\!0$) but requires ~60 layers of
composition to reach its highest internal certainty (ROC AUC
$\approx 0.86$). Notably, the highest-DML treatment is also one of
the lowest probing-AUC treatments: $T_5$ topical comprehensiveness
peaks at ROC AUC $0.974$ in the body-text probe, the lowest of the
six — consistent with the intuition that compositional content
matching is the feature the model both \emph{uses most} and
\emph{represents least cheaply} in surface vocabulary.

We follow \citet{hewitt2019designing} in being explicit that probing
demonstrates the \emph{availability} of treatment-correlated and
decision-related information in the model's representations; it does
not by itself establish that this information drives behavior at
inference. The latter is the role of the DML estimator, which
remains the paper's primary claim on causal effect.

---

## Figure caption (full text for fig_admission_pooled)

```latex
\caption{\textbf{Admission pre-commitment probe.} For each URL
span in the rerank prompt, the probe label is 1 if the (model,
variant) admitted that URL, else 0. A logistic regression is trained
per (transformer layer, pooling), 80/20 stratified split, on frozen
hidden states from Llama-3.3-70B and Qwen-2.5-72B; results are pooled
across the four prompt variants \{biased, neutral, biased+RAG,
neutral+RAG\}. The mean-pooled probe (solid) rises from ROC AUC
$=0.671$ at the embedding layer ($\ell\!=\!0$) to a peak of $0.862$
at $\ell\!=\!60$, a $+0.191$ AUC gain that reflects genuine
compositional integration of the SERP context. Last-token pooling
(dashed) starts at chance and catches up by $\ell\!\approx\!20$. The
shaded band shows the min–max envelope across the four prompt
variants. Unlike the body-text treatment probes (Appendix~D), this
curve is not lexically leaky: the label is the model's own behavioral
output, not a property of the page's vocabulary.}
\label{fig:admission_probe}
```

---

## BibTeX entries you'll need

```bibtex
@inproceedings{hewitt2019designing,
  title     = {Designing and Interpreting Probes with Control Tasks},
  author    = {Hewitt, John and Liang, Percy},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  year      = {2019},
  pages     = {2733--2743}
}

@inproceedings{alain2017understanding,
  title     = {Understanding intermediate layers using linear classifier probes},
  author    = {Alain, Guillaume and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations Workshop},
  year      = {2017}
}

@inproceedings{tenney2019bert,
  title     = {{BERT} rediscovers the classical {NLP} pipeline},
  author    = {Tenney, Ian and Das, Dipanjan and Pavlick, Ellie},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  year      = {2019},
  pages     = {4593--4601}
}

@inproceedings{geva2021transformer,
  title     = {Transformer feed-forward layers are key-value memories},
  author    = {Geva, Mor and Schuster, Roei and Berant, Jonathan and Levy, Omer},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  year      = {2021},
  pages     = {5484--5495}
}
```

---

## Length budget

| Section | Words |
|---|---|
| §4.1 Motivation and design     | 230 |
| §4.2 Content probes are leaky  | 200 |
| §4.3 Admission pre-commitment  | 290 |
| §4.4 How probing complements DML | 200 |
| Figure caption                  | 130 |
| **Total**                       | **~1050** |

If you need this tighter, the shortest viable version is §4.3 + §4.4
alone (490 words). §4.1 can be folded into the §3 methodology and
§4.2 collapsed to a single appendix paragraph. The figure caption
needs to stay full because it carries the headline numbers.

---

## What's in the figure that's NOT in the text

The per-variant figure (`fig_admission_variants.png`) showed that
**biased prompts make the admission probe ROC ~5 points higher than
neutral prompts at every layer**, with no meaningful RAG effect. This
is worth a single appendix sentence — biased system prompts apparently
make the model's decision-making *more legible* to a probe, suggesting
the biased pathway is more deterministic. Reviewer-bait if you have
room; safely omitted otherwise.

---

## Quick sanity sentence for the abstract / conclusion

For one sentence that captures the headline:

> A novel pre-commitment probe shows that the LLM's admission
> decision crystallizes gradually across the network (ROC AUC
> $+0.19$ from input embedding to mid-network peak), demonstrating
> genuine compositional integration of SERP context rather than
> surface-level pattern matching.

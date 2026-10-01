# Astra paper selection and library extension

Date: 2026-10-02

## Request and work

Valerian requested finding the old manuscript, reviewing current experiment context, and carefully selecting downloads using GPT-6 Astra at high reasoning effort. The agent read the extracted full 12-page manuscript, current experiment contracts and relevant implementation. It mapped all 36 old references and recorded 56 retrieval decisions. This is not a reproduction of scientific results or full-method review of every downloaded paper.

Old manuscript: `ARR_ACL_CycleOct2026/oldPaper/9568_What_Drives_LLM_Re_Rankin.pdf` under the main workspace. Preserved unchanged.

Eight new papers A001-A008 were downloaded: AutoGEO, CC-GSEO-Bench, recency bias in reranking, causal inference in NLP, DoubleML Python, attribution faithfulness, omitted-variable sensitivity, and prediction-powered inference. P026 Campbell/Fiske was recovered from a university-hosted PDF. P001 and P078 PDF retries failed; retain explicit access gaps.

The library is at `/Users/valerianfourel/Hamburg/GEODML_Unified/ARR_ACL_CycleOct2026/papers/`. It now has 101 records, 86 PDFs, 96 complete abstracts reviewed, one partial abstract, and four unavailable abstracts. Ten topic folders contain 205 notes and 179 relative PDF shortcuts. Original 93-record bibliography remains frozen. Sol reviewed the initial batches; Astra high selected and reviewed the eight additions. Download years identify initial preprints, not necessarily publication dates.

## Findings and limits

The review distinguishes observed prompt readiness from randomized B, textual answer support from causal reliance, and current implementation from historical pinned execution. It flags unsupported causal-identification language, recency citation reversal, outcome-scale/covariate-count discrepancies requiring archived-run reconciliation, and pre-compaction shuffling that score sorting can erase. Production exact-target ablation does exist. No historical execution outcome is inferred solely from current source.

## Verification

101 unique IDs and CSV rows; 86 distinct matching SHA-256 hashes; all PDFs parse with pdfinfo and positive page counts; all local Markdown links and PDF shortcuts resolve; original bibliography hash unchanged. No scientific code, data, cluster state, allocation or job was changed. No cluster facts were rechecked because this was a local literature task.

## Saved documentation

See `analysis/docs/literature/2026-10-02_astra-paper-selection.md`, `2026-10-02_astra-paper-decisions.json`, `2026-10-02_topic-paper-library.json`, `2026-10-02_paper-library-verification.json`, and `2026-10-02_paper-review-provenance.json`. The Oct 1 snapshots remain unchanged. PDFs stay outside the active checkout.

## Next step

Read the methods of direct competitors and write explicit V2 measurement/estimand definitions before further broad retrieval or scientific changes. No execution is authorized by this recommendation.

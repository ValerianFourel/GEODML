"""How much of the reranker's intent sensitivity is lexical? (PREREG.md robustness, objection O1)

(i) Overlap features in the shortlisting model (top-k Plackett–Luce over each reranker event, the P|C
    stage of ``funnel_study``): word overlap of the candidate's title and snippet with the prompt minus its
    keyword words, with the prompt's action words (a lexicon fixed from the confirmation-keyword prompt
    texts only), and, for the Reactive Loop, with the agent's own query of that event. Overlap is a mediator
    of prompt intent, not a confounder: the change in the intent coefficients is the lexical part.
(ii) A lexical selector: in every reranker event keep the top k candidates by BM25 against the text the
    reranker scored against (Parallel: the user prompt; Reactive: the agent's query), and compare its
    shortlist slope increment β_Plex − β_C with the reranker's β_P − β_C under shared keyword draws.
"""

from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from math import log
from pathlib import Path
import time

import numpy as np

from analysis.interpretability.pipeline import funnel_models as fm
from analysis.interpretability.pipeline import intent_stages as stages

from .chain import summarise, within_keyword_ols
from .tables import PARALLEL, Tables, prompt_metadata, strata, tokens

LEXICON_SIZE, LEXICON_MIN_PROMPTS = 50, 200
FROZEN_LEXICON = Path(__file__).with_name("lexicon.json")  # fixed by the first (Mac) run; PREREG addendum B1


def action_lexicon(prompt_rows: list[dict], keep_keyword) -> list[str]:
    """Tokens whose within-keyword presence correlates most positively with x, over the prompts whose
    keyword satisfies ``keep_keyword`` (text and x only; no outcome is used)."""
    rows = [r for r in prompt_rows if keep_keyword(r["keyword"])]
    kw = np.unique([r["keyword"] for r in rows], return_inverse=True)[1]
    x = np.asarray([r["x"] for r in rows], float)
    toks = [tokens(r["question"]) - tokens(r["keyword"]) for r in rows]
    freq = Counter(t for ts in toks for t in ts)
    vocab = [t for t, c in freq.items() if c >= LEXICON_MIN_PROMPTS]
    size = int(kw.max()) + 1

    def centred(v):
        return v - (np.bincount(kw, v, minlength=size) / np.bincount(kw, minlength=size))[kw]

    xc = centred(x)
    scores = {}
    for t in vocab:
        v = centred(np.asarray([t in ts for ts in toks], float))
        den = np.sqrt((v * v).sum() * (xc * xc).sum())
        if den > 0:
            scores[t] = float((v * xc).sum() / den)
    return [t for t, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:LEXICON_SIZE]]


def bm25_scores(query: set, doc_tf: Counter, doc_len: int, idf: dict, avgdl: float, k1=1.2, b=0.75) -> float:
    s = 0.0
    for q in query:
        tf = doc_tf.get(q, 0)
        if tf:
            s += idf.get(q, 0.0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))
    return s


def lexical_selector(event, answer, row, query_of_event: list[set], k_of_event: np.ndarray, doc_tf, doc_len, idf, avgdl,
                     position) -> np.ndarray:
    """Boolean per candidate: kept by the BM25 selector (ties by stored position, then candidate order)."""
    score = np.asarray([bm25_scores(query_of_event[e], doc_tf[r], doc_len[r], idf, avgdl) for e, r in zip(event, row)])
    order = np.lexsort((np.arange(len(event)), position[row], -score, event))
    ev_sorted = event[order]
    start = np.r_[0, np.flatnonzero(np.diff(ev_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    keep = np.zeros(len(event), bool)
    keep[order] = rank < k_of_event[ev_sorted]
    return keep


def shortlist_value(answer, row, keep, u, n_answers) -> np.ndarray:
    """Mean u of the distinct rows kept across an answer's events."""
    key = np.unique(answer[keep].astype(np.int64) * (1 << 32) + row[keep])
    a, r = key >> 32, key & ((1 << 32) - 1)
    c = np.bincount(a, minlength=n_answers).astype(float)
    return np.divide(np.bincount(a, u[r], minlength=n_answers), c, out=np.full(n_answers, np.nan), where=c > 0)


def selection_stage(t: Tables, mask: np.ndarray, extra: dict):
    """The P|C stage of funnel_study (main specification) on the candidates of ``mask``, plus extra per-candidate
    static features (name -> array over all candidates)."""
    from analysis.scripts import funnel_study as study
    c, rf = t.data.cand, t.data.rf
    idx = np.flatnonzero(mask[c["answer"]])
    sel = stages.selection_rows(c["event"][idx], c["selected"][idx], c["answer"][idx], z=np.zeros(len(idx)),
                                u=np.zeros(len(idx)), on_keyword=np.zeros(len(idx)), topic=np.arange(len(idx), dtype=float))
    source = idx[sel.topic.astype(np.int64)]
    columns = {name: rf[name][c["row"][source]] for name, *_ in study.ROW_FEATURES}
    columns.update({k: rf[k][c["row"][source]] for k in study.MISSING_INDICATORS})
    columns["u"] = rf["u"][c["row"][source]]
    columns["topic_similarity"], columns["on_keyword"] = c["topic"][source], c["on_keyword"][source]
    feats = study._features("main", "P|C")
    for name, values in extra.items():
        columns[name] = values[source]
        feats.append(fm.Feature(name, "LX"))
    set_answer = np.zeros(sel.sets, np.int64)
    set_answer[sel.set] = c["answer"][source]
    return fm.Stage("choice", fm.Design(feats, columns), sel, c["answer"][source], set_answer)


def run(t: Tables, args, *, cache_root: Path, deadline: float):
    import pandas as pd
    from analysis.scripts import funnel_study as study
    root = Path(args.input_root)
    c = t.data.cand
    rows = pd.read_parquet(Path(args.features or root / "funnel-features-v1") / "rows.parquet", columns=["row_id", "title", "snippet", "position"])
    doc_tok = [TOK(f"{a} {b}") for a, b in zip(rows["title"], rows["snippet"])]
    doc_tf = [Counter(d) for d in doc_tok]
    doc_len = np.asarray([len(d) for d in doc_tok])
    doc_set = [set(d) for d in doc_tok]
    N = len(doc_tok)
    df = Counter(w for s in doc_set for w in s)
    idf = {w: log((N - n + 0.5) / (n + 0.5) + 1.0) for w, n in df.items()}
    avgdl = float(doc_len.mean())
    position = rows["position"].to_numpy(np.int64)

    lexicon_path = Path(args.lexicon) if args.lexicon else FROZEN_LEXICON
    if lexicon_path.exists():
        lexicon = set(json.loads(lexicon_path.read_text())["lexicon"])
        lexicon_source = f"frozen list {lexicon_path.name} (confirmation-keyword prompt texts, top 50 by within-keyword correlation with x)"
    else:
        pm_raw = []
        with gzip.open(args.prompts, "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                pm_raw.append({"id": r["prompt"]["candidate_id"], "question": r["prompt"]["question"], "keyword": r["prompt"]["keyword"],
                               "x": float(r["axis"]["axis_1_percentile_0_1"])})
        lexicon = set(action_lexicon(pm_raw, lambda k: not study.exploration_keyword(k)))
        lexicon_source = "computed: confirmation-keyword prompt texts, top 50 by within-keyword correlation with x"
    pm = t.prompts
    queries = {i: q for i, q in enumerate(t.queries)} if t.queries is not None else {}
    events = np.load(Path(t.manifest["paths"]["extract"]) / "events.npz")
    ev_answer, ev_search = events["answer"], events["search"]

    natural = (t.condition == "natural") & t.split
    use = natural[c["answer"]]
    cand_idx = np.flatnonzero(use)
    prompt_tok = {}
    kw_tok = {}
    for a in np.unique(c["answer"][cand_idx]):
        q = pm[t.prompt_id[a]]["question"]
        prompt_tok[a] = tokens(q)
        kw_tok[a] = tokens(t.keyword_text[a])
    ov_prompt = np.zeros(len(c["answer"]))
    ov_action = np.zeros(len(c["answer"]))
    ov_query = np.zeros(len(c["answer"]))
    query_of_event = [set() for _ in range(len(ev_answer))]
    prompt_of_event = [set() for _ in range(len(ev_answer))]
    for e in np.unique(c["event"][cand_idx]):
        a = int(ev_answer[e])
        qs = queries.get(a, [])
        s = int(ev_search[e])
        query_of_event[e] = tokens(qs[s]) if 0 <= s < len(qs) else set()
        prompt_of_event[e] = prompt_tok[a]
    for i in cand_idx:
        a, r, e = int(c["answer"][i]), int(c["row"][i]), int(c["event"][i])
        d = doc_set[r]
        content = prompt_tok[a] - kw_tok[a]
        ov_prompt[i] = len(d & content)
        ov_action[i] = len(d & content & lexicon)
        ov_query[i] = len(d & (query_of_event[e] - kw_tok[a]))
    extra_all = {"overlap_prompt": np.log1p(ov_prompt), "overlap_action_words": np.log1p(ov_action),
                 "overlap_agent_query": np.log1p(ov_query)}

    selection_draws = int(args.selection_draws)
    draws = stages.keyword_draws(int(t.keyword.max()) + 1, args.bootstrap, args.seed)[:selection_draws]
    cache_root.mkdir(parents=True, exist_ok=True)
    common = t.common()
    out = {"selection_model_draws": selection_draws, "lexicon": sorted(lexicon), "lexicon_source": lexicon_source,
           "agent_queries": t.manifest.get("queries"), "strata": {}}
    for name, m in strata(t).items():
        if not m.any():
            continue
        parallel = "Parallel" in name
        mask = m & common
        entry = {}
        # (i) overlap features in the shortlisting model
        extra = {k: v for k, v in extra_all.items() if k != "overlap_agent_query" or (not parallel and t.queries is not None)}
        fits = {}
        for label, ex in ((("base", {}), ("lexical", extra)) if selection_draws > 0 else ()):
            stage = selection_stage(t, mask, ex)
            stage.design.fit_stats(t.x[stage.row_answer])
            digest = hashlib.sha256(f"{name}|{label}".encode()).hexdigest()[:12]
            if time.monotonic() > deadline:
                return None
            try:
                fits[label] = fm.estimate_blocks(stage, answer_x=t.x, answer_keyword=t.keyword, draws=draws, shuffles=[],
                                                 workers=args.workers, parts=cache_root / f"{digest}.parts.jsonl", deadline=deadline)
            except fm.DeadlineReached:
                return None
        keep = ("page_intent_z", "intent_x_prompt", "intent_alignment", "topic_similarity", "on_keyword",
                "overlap_prompt", "overlap_action_words", "overlap_agent_query")
        if not fits:
            entry["selection_model"] = {"skipped": "--selection-draws 0"}
        else:
            entry["selection_model"] = {label: {f: {"beta_per_sd": v["beta_per_sd"], "ci95": v["ci95"]}
                                                for f, v in fit["features"].items() if f in keep}
                                        for label, fit in fits.items()}
            entry["selection_model_intent_change"] = {
                f: fm.contrast_from_replicates(fits["lexical"], fits["base"], f) for f in ("intent_x_prompt", "intent_alignment", "page_intent_z")}
            entry["selection_model_lexical_block_fit_share"] = fits["lexical"]["blocks"].get("LX", {}).get("fit_share")
        # (ii) lexical selector against the text the reranker scored against (and the prompt, for Reactive)
        cm = mask[c["answer"]]
        idx = np.flatnonzero(cm)
        k_ev = np.where(parallel, 7, 3) * np.ones(len(ev_answer), np.int64)
        P_lex = {}
        for label, qe in (("reranker_text", prompt_of_event if parallel else query_of_event), ("user_prompt", prompt_of_event)):
            if (parallel and label == "user_prompt") or (not parallel and label == "reranker_text" and t.queries is None):
                continue
            kept = lexical_selector(c["event"][idx], c["answer"][idx], c["row"][idx], qe, k_ev, doc_tf, doc_len, idf, avgdl, position)
            P_lex[label] = shortlist_value(c["answer"][idx], c["row"][idx], kept, t.data.rf["u"], len(t.x))
        sel = {}
        x, k = t.x[mask], t.keyword[mask]
        Cv, Pv, Kv = t.values["C"][mask], t.values["P"][mask], t.values["K"][mask]
        for label, v in P_lex.items():
            pv = v[mask]
            ok = np.isfinite(pv)

            def stats_of(w):
                ww = None if w is None else w[k[ok]]
                b = {n: float(within_keyword_ols(x[ok][:, None], arr[ok], k[ok], ww)[0]) for n, arr in
                     (("C", Cv), ("P", Pv), ("Plex", pv), ("K", Kv))}
                return {"lexical_increment": b["Plex"] - b["C"], "reranker_increment": b["P"] - b["C"],
                        "reranker_minus_lexical": b["P"] - b["Plex"],
                        "lexical_share_of_reranker": (b["Plex"] - b["C"]) / (b["P"] - b["C"]) if b["P"] != b["C"] else np.nan,
                        "lexical_share_of_K": (b["Plex"] - b["C"]) / b["K"] if b["K"] else np.nan}

            obs = stats_of(None)
            reps = [stats_of(d) for d in stages.keyword_draws(int(t.keyword.max()) + 1, args.bootstrap, args.seed)]
            sel[label] = {q: summarise(obs[q], [r[q] for r in reps], None) for q in obs}
            sel[label]["answers"] = int(ok.sum())
        entry["lexical_selector"] = sel
        out["strata"][name] = entry
        print(json.dumps({"stratum": name, "lexical_selector": {l: s["lexical_share_of_reranker"]["estimate"] for l, s in sel.items()}}), flush=True)
    return out


def TOK(text: str) -> list:
    from .tables import TOKEN
    return TOKEN.findall((text or "").casefold())

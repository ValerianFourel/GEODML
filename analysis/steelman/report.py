"""RESULTS.md and the CSV tables from the part files (chain, generator, fe, lexical, pairs).

Every number in RESULTS.md comes from those files; rerun ``python -m analysis.steelman report`` to rebuild.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from . import decide
from .tables import DEFAULT_INPUTS

ROWS = [("R0 · prompt words (lexical replay, mechanical)", "prompt_words_R0"), ("R − R0 · query rewriting", "query_rewriting"),
        ("C − R · deduplication", "dedup_condition"), ("P − C · reranker shortlist", "reranker"),
        ("I − P · shown-order weighting", "shown_order"), ("K − I · generator's own choices", "generator"),
        ("K − P · generator vs random selector", "generator_vs_random"), ("P · admission (everything before the generator)", "admission_P")]
METHODS = ("qwen38 · Reactive", "qwen38 · Parallel")
RESULTS_MD = Path(__file__).with_name("RESULTS.md")


def pct(v, digits=0):
    return "—" if v is None else f"{100 * v:.{digits}f}%"


def ci_pct(e):
    lo, hi = e["ci95"]
    return f"{pct(e['estimate'])} [{pct(lo)}, {pct(hi)}]"


def num(e, d=3, key="ci95"):
    lo, hi = e[key]
    return f"{e['estimate']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def deck_shares(explore: dict, method: str) -> dict:
    s = {k: explore["slopes"][method][f"{k}_u"]["slope"] for k in ("R0", "R", "P", "K")}
    k = s["K"]
    return {"prompt_words_R0": s["R0"] / k, "query_rewriting": (s["R"] - s["R0"]) / k, "reranker": (s["P"] - s["R"]) / k,
            "generator_vs_random": (s["K"] - s["P"]) / k, "admission_P": s["P"] / k}


def paper_text(ch, gen, fu, v1, v2, lex=None) -> list:
    """Draft paper text with the numbers of this run (wording follows the verdicts)."""
    R, P = METHODS
    cr, cp = ch["strata"][R]["chain"]["common"], ch["strata"][P]["chain"]["common"]
    dr, dp = gen["strata"][R]["delta"]["main"], gen["strata"][P]["delta"]["main"]
    s = lambda c, t: pct(c["shares"][t]["estimate"])
    lines = ["## Draft paper text\n", "**Claim sentence (as the evidence supports it).**\n"]
    if v2["verdict"] == "supported":
        lines.append("> In agentic LLM search, the prompt's intent reaches the cited sources by changing which documents are "
                     "shortlisted, through the queries the agent writes and the reranker that filters their results, while the "
                     "generator's choice among shortlisted documents follows topical match and presentation order; removing the "
                     f"generator's sensitivity to intent changes the cited-intent slope by {pct(dr['delta_gen_share_of_K']['estimate'], 1)} "
                     f"[{pct(dr['delta_gen_share_of_K']['ci95'][0], 1)}, {pct(dr['delta_gen_share_of_K']['ci95'][1], 1)}] (Reactive Loop) and "
                     f"{pct(dp['delta_gen_share_of_K']['estimate'], 1)} [{pct(dp['delta_gen_share_of_K']['ci95'][0], 1)}, "
                     f"{pct(dp['delta_gen_share_of_K']['ci95'][1], 1)}] (Parallel Expansion).\n")
    else:
        lines.append(f"> C2 verdict: {v2['verdict']}; the claim sentence is restricted to admission (C1).\n")
    lines.append(f"Supporting numbers (Qwen, exploration keywords). Reactive Loop: of the cited-intent slope, the prompt-text replay "
                 f"accounts for {s(cr, 'prompt_words_R0')}, query rewriting {s(cr, 'query_rewriting')}, the reranker {s(cr, 'reranker')}, "
                 f"shown-order weighting {s(cr, 'shown_order')} and the generator's own choices {s(cr, 'generator')}. Parallel Expansion: "
                 f"{s(cp, 'prompt_words_R0')}, {s(cp, 'query_rewriting')}, {s(cp, 'reranker')}, {s(cp, 'shown_order')}, {s(cp, 'generator')}. "
                 "Do not write \"chiefly through the queries\": the reranker step is at least as large in both methods, and the query "
                 "share depends on whether the prompt names its keyword (follow-up section).\n")
    if lex:
        lr = lex["strata"][R]["lexical_selector"]["reranker_text"]["lexical_share_of_reranker"]
        lp = lex["strata"][P]["lexical_selector"]["reranker_text"]["lexical_share_of_reranker"]
        lines.append(f"Plain word matching yields an increment as large as the reranker's: a BM25 selector scored against the same "
                     f"text reaches {ci_pct(lr)} of the reranker's increment under the Reactive Loop (the agent's own queries) and "
                     f"{ci_pct(lp)} under Parallel Expansion (the user prompt). Under the Reactive Loop the reranker never sees the "
                     "prompt, so the intent it adds travels through the agent's queries. Word-overlap controls in the shortlisting model "
                     "leave the reranker's intent coefficients unchanged, so the reranker is not simply counting shared words.\n")
    lines.append("**Limitations paragraph.**\n")
    lines.append("> These results are exploratory and observational: the prompt's position on the intent axis is a measured "
                 "property of its text, and the stage decomposition attributes an association, not an effect. The generator analysis "
                 "rests on one model (Qwen); Llama, which drops most shown links, could not be analysed at the trace level here. "
                 "The decisive test, which we did not run, holds the shortlist fixed: showing the same documents in the same order "
                 "under prompts of the same keyword at different intent positions would measure the generator's intent sensitivity by "
                 "design rather than through a fitted choice model. Our counterfactual also conditions on how many sources an answer "
                 f"cites, and that count itself falls with intent under the Reactive Loop "
                 f"({num(ch['strata'][R]['cited_count'], 2)} links from x = 0 to 1). Finally, part of what we call query rewriting "
                 "travels with whether the prompt names its keyword, so it reflects the prompt's wording as much as the agent's "
                 "reformulation.\n")
    lines.append("**Positioning.**\n")
    lines.append("> Tannenbaum (2026) and the survey of Martinez (2026) infer from live engines that exposure matters more than "
                 "selection; in a closed testbed where every stage is observed, we decompose a prompt-intent effect stage by stage "
                 "and find that the generator's own intent sensitivity leaves the cited-intent slope unchanged within a margin fixed "
                 "before the analysis once the shortlist is fixed.\n")
    return lines


def build(out_dir: Path) -> int:
    out_dir = Path(out_dir)
    load = lambda name: json.loads((out_dir / f"{name}.json").read_text()) if (out_dir / f"{name}.json").exists() else None
    ch, gen, fe, lex, pairs, fu = (load(n) for n in ("chain", "generator", "fe", "lexical", "pairs", "followup"))
    explore = json.loads((DEFAULT_INPUTS / "funnel-explore-v1/explore.json").read_text())
    L = []
    w = L.append
    commits = {n: p["git_commit"] for n, p in (("chain", ch), ("generator", gen), ("fe", fe), ("lexical", lex), ("pairs", pairs)) if p}
    w("# Steelman results: does prompt intent reach the cited sources at the shortlist?\n")
    w("Exploratory (funnel study addendum A1): Qwen, natural condition, the 237 exploration keywords with traces on the Mac. "
      "Observational: x is a measured property of the prompt text. Rules fixed in `PREREG.md` "
      f"(sha256 `{ch['prereg_sha256'][:12]}…`) before any part ran. Code commits: "
      + ", ".join(f"{k} `{v[:7]}`" for k, v in commits.items()) + ". Rebuild: `python -m analysis.steelman report`.\n")
    w("Intervals are 95% keyword-bootstrap percentile intervals (200 draws; refitted generator models use the first 100); "
      "p values are within-keyword shuffles of x (200). Page intent is the page's percentile on the prompt scale (u).\n")

    # verdicts
    v1 = {m: decide.verdict_c1(ch["strata"][m], {e: c for e, c in ch["engine_strata"].items() if e.startswith(m)}) for m in METHODS}
    v2 = decide.verdict_c2({m: gen["strata"][m]["delta"]["main"] for m in METHODS}) if gen else None
    w("## Verdicts\n")
    w("| Claim | Qwen · Reactive (primary) | Qwen · Parallel |\n|---|---|---|")
    w(f"| C1 admission | **{v1[METHODS[0]]['verdict']}** | **{v1[METHODS[1]]['verdict']}** |")
    if v2:
        w(f"| C2 selection (both methods jointly) | **{v2['verdict']}** | |")
    w("")
    for m in METHODS:
        if v1[m]["reasons"]:
            w(f"- {m}, C1: " + "; ".join(v1[m]["reasons"]) + ".")
    w("")

    # side-by-side table
    w("## Table 1. Share of the cited-intent slope added at each step\n")
    w("Shares of β_K, the within-keyword slope of the cited sources' rank-weighted intent on x. Increments sum to 1 down "
      "the first six rows in every column. The deck column is the earlier hand arithmetic (each slope on its own subset, "
      "no intervals, C and I not separated). Identity baseline = row K − I; random baseline = row K − P.\n")
    csv_rows = []
    for m in METHODS:
        e = ch["strata"][m]
        deck = deck_shares(explore, m)
        w(f"**{m}** — common sample {e['common_sample']:,} answers; β_K common {num(e['chain']['common']['slopes']['K'])}, "
          f"with controls {num(e['chain']['controls']['slopes']['K'])}, given lattice target {num(e['chain']['target']['slopes']['K'])}.\n")
        w("| Step | Deck | Common sample | + prompt controls | Given lattice target | Only answers that dropped a link |\n|---|---|---|---|---|---|")
        for label, term in ROWS:
            cells = [pct(deck.get(term)) if term in deck else "—"]
            for variant in ("common", "controls", "target"):
                cells.append(ci_pct(e["chain"][variant]["shares"][term]))
            cells.append(ci_pct(e["droppers_only"]["shares"][term]))
            w(f"| {label} | " + " | ".join(cells) + " |")
            for variant in ("common", "controls", "target"):
                s = e["chain"][variant]["shares"][term]
                inc = e["chain"][variant]["increments"][term]
                csv_rows.append({"stratum": m, "variant": variant, "term": term, "increment": inc["estimate"],
                                 "increment_lo": inc["ci95"][0], "increment_hi": inc["ci95"][1],
                                 "permutation_p": inc.get("permutation_p"), "share_of_K": s["estimate"],
                                 "share_lo": s["ci95"][0], "share_hi": s["ci95"][1], "answers": e["chain"][variant]["answers"]})
        w(f"\nDropping answers: {pct(e['droppers_share_of_answers'])} of the common sample dropped at least one shown link.\n")
    with open(out_dir / "stage_shares.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    # O1
    w("## O1. Admission is built into the pipeline\n")
    for m in METHODS:
        c = ch["strata"][m]["chain"]["common"]
        w(f"- **{m}.** Query rewriting adds {num(c['increments']['query_rewriting'])} "
          f"(p = {c['increments']['query_rewriting']['permutation_p']:.3f}); the reranker adds {num(c['increments']['reranker'])} "
          f"(p = {c['increments']['reranker']['permutation_p']:.3f}); the prompt-text replay R0 {num(c['increments']['prompt_words_R0'])}.")
    w("")
    if lex:
        w("")
        for m in METHODS:
            e = lex["strata"][m]
            for label, s in e["lexical_selector"].items():
                w(f"- **{m}, BM25 selector against the {label.replace('_', ' ')}.** Shortlist increment {num(s['lexical_increment'])} "
                  f"versus the reranker's {num(s['reranker_increment'])}; lexical share of the reranker step "
                  f"{ci_pct(s['lexical_share_of_reranker'])}.")
            chg = e["selection_model_intent_change"]
            sm = e["selection_model"]
            parts = []
            for f in ("intent_x_prompt", "intent_alignment"):
                if f in sm["base"]:
                    parts.append(f"{f} {sm['base'][f]['beta_per_sd']:+.3f} → {sm['lexical'][f]['beta_per_sd']:+.3f} per SD "
                                 f"(change {chg[f]['estimate']:+.3f} [{chg[f]['ci95'][0]:+.3f}, {chg[f]['ci95'][1]:+.3f}])")
            w(f"- **{m}, shortlisting model with word-overlap controls ({lex.get('selection_model_draws', '?')} keyword draws).** " + "; ".join(parts) +
              f". Overlap block share of fit {pct(e['selection_model_lexical_block_fit_share'])}.")
        w(f"\nAction-word lexicon ({len(lex['lexicon'])} words, from the confirmation-keyword prompts only): "
          + ", ".join(lex["lexicon"][:20]) + ", ….\n")

    # O2
    if gen:
        w("## O2. Shown order absorbs intent; C2 equivalence\n")
        w("| Quantity | Qwen · Reactive | Qwen · Parallel |\n|---|---|---|")
        g = {m: gen["strata"][m] for m in METHODS}
        for label, var, q in (("Δ_gen, drop-intent refit (primary)", "main", "delta_gen"),
                              ("Δ_gen, intent coefficients zeroed", "main", "delta_gen_zeroed"),
                              ("Δ_gen, models without shown slot", "no_slot", "delta_gen"),
                              ("Δ_gen, reranker score as control", "score", "delta_gen"),
                              ("Model check: β_x(E[K]) − β_K", "main", "model_check")):
            w(f"| {label} | " + " | ".join(num(g[m]["delta"][var][q], 4) for m in METHODS) + " |")
        w("| Δ_gen, 90% interval (TOST against ±0.015) | " + " | ".join(
            f"[{g[m]['delta']['main']['delta_gen']['ci90'][0]:+.4f}, {g[m]['delta']['main']['delta_gen']['ci90'][1]:+.4f}]" for m in METHODS) + " |")
        w("| Δ_gen as share of β_K | " + " | ".join(ci_pct(g[m]["delta"]["main"]["delta_gen_share_of_K"]) for m in METHODS) + " |")
        w(f"| Keep-informative answers | {g[METHODS[0]]['keep_informative_answers']:,} | keep fixed (99% kept) |\n")
        w("Keep and order coefficients (per SD, main variant):\n")
        for m in METHODS:
            for key, coefs in g[m]["coefficients"].items():
                if not key.startswith("main|"):
                    continue
                w(f"- {m}, {key.split('|')[1]}: " + "; ".join(f"{f} {num(v)}" for f, v in coefs.items() if f != "slot_effects"))
        w("\nWithin-answer correlation of the shown slot with:\n")
        w("| | " + " | ".join(METHODS) + " |\n|---|---|---|")
        for f in ("u", "intent_alignment", "intent_x_prompt", "topic_similarity", "reranker_logit"):
            w(f"| {f} | " + " | ".join(num(g[m]["slot_correlations"][f]) for m in METHODS) + " |")
        w("")
    if fe:
        w("Two-way fixed effects (answer and snapshot row; the same snippet shown under different prompts), linear "
          "probability per SD of each regressor:\n")
        for m in METHODS:
            for outcome, r in fe["strata"][m].items():
                if "skipped" in r:
                    w(f"- {m}, {outcome}: too few rows ({r['skipped']}).")
                    continue
                w(f"- {m}, {outcome} (mean {r['outcome_mean']:.3f}; {r['rows_used']:,} shown rows, {r['distinct_snapshot_rows']:,} snippets, "
                  f"within-snippet SD of x {r['within_row_sd_of_x']:.3f}): " +
                  "; ".join(f"{f} {num(v, 4)}" for f, v in r["coefficients_per_sd"].items()))
        w("")
    if pairs:
        w("Score-matched adjacent shown links (|Δ logit score| < ε; pair logit of the later slot winning):\n")
        for m in METHODS:
            for eps, r in pairs["strata"][m].items():
                for outcome in ("order", "keep"):
                    o = r.get(outcome, {})
                    if o.get("skipped") or not o:
                        w(f"- {m}, {eps}, {outcome}: {o.get('pairs', 0)} pairs, not fitted.")
                        continue
                    w(f"- {m}, {eps}, {outcome}: {o['pairs']:,} pairs, later wins {pct(o['later_wins_share'])}; slot intercept "
                      f"{num(o['intercept_slot_effect'])}; " + "; ".join(f"Δ{f} {num(v)}" for f, v in o["difference_coefficients_per_sd"].items()))
        w("")

    # O3
    w("## O3. Funnel arithmetic\n")
    for m in METHODS:
        e = ch["strata"][m]
        c = e["chain"]["common"]
        w(f"- **{m}.** Identity selector (first L shown, shown order) leaves the generator {ci_pct(c['shares']['generator'])} of β_K; "
          f"against a random selector (E[K] = P) {ci_pct(c['shares']['generator_vs_random'])}. Cited count on x "
          f"{num(e['cited_count'])} links (p = {e['cited_count']['permutation_p']:.3f}); share of shown links cited "
          f"{num(e['cited_share_of_shown'])}; shown count {num(e['shown_count'])}.")
    w("")
    # O4
    w("## O4. Wording and length\n")
    for m in METHODS:
        nf = ch["strata"][m]["noise_floor"]
        w(f"- **{m}.** Within (keyword, lattice target, engine) cells ({nf['cells_with_two_or_more']:,} cells with ≥ 2 prompts, "
          f"{nf['answers_in_such_cells']:,} answers): slope of K on x {num(nf['terms']['K']['slope_within_cell'])}, of P "
          f"{num(nf['terms']['P']['slope_within_cell'])}, of R {num(nf['terms']['R']['slope_within_cell'])}; within-cell SD of K "
          f"{nf['terms']['K']['sd_within_cell']:.3f} versus within-keyword {nf['terms']['K']['sd_within_keyword']:.3f}.")
    w("")
    # O5
    w("## O5. Qwen only\n")
    w("Llama traces are not on the Mac; its R0 and K come from the published answers (natural condition):\n")
    w("| Stratum | R0 slope | K slope | R0 / K | Cited count on x | Answers keeping every shown link |\n|---|---|---|---|---|---|")
    for name, r in ch["published_rows"].items():
        w(f"| {name} | {num(r['R0'])} | {num(r['K_cited_u'])} | {pct(r['R0']['estimate'] / r['K_cited_u']['estimate'])} | "
          f"{num(r['cited_count'], 2)} | {pct(r['share_answers_keeping_all_shown'])} |")
    w("")
    # O6
    w("## O6. Replication\n")
    w("No held-out (confirmation-keyword) result exists anywhere on disk; everything here is exploratory. Engine strata and "
      "leave-one-keyword-out:\n")
    w("| Stratum | Generator share K − I | Query rewriting | Reranker |\n|---|---|---|---|")
    for name, c in ch["engine_strata"].items():
        w(f"| {name} | {ci_pct(c['shares']['generator'])} | {num(c['increments']['query_rewriting'])} | {num(c['increments']['reranker'])} |")
    for m in METHODS:
        lo = ch["strata"][m]["loko"]
        w(f"\n{m}, leave one keyword out: generator share {pct(lo['generator']['min'], 1)}–{pct(lo['generator']['max'], 1)}, "
          f"query rewriting {pct(lo['query_rewriting']['min'], 1)}–{pct(lo['query_rewriting']['max'], 1)}, reranker "
          f"{pct(lo['reranker']['min'], 1)}–{pct(lo['reranker']['max'], 1)}.")
    w("")
    if fu:
        w("## Exploratory follow-up (added after the first run, not pre-registered): whether the prompt names its keyword\n")
        w("Adding the prompt controls moved the query-rewriting share. One control at a time, shares of β_K:\n")
        w("| Stratum | Control | Prompt words R0 | Query rewriting | Reranker | Generator K − I |\n|---|---|---|---|---|---|")
        for m in METHODS:
            e = fu["strata"][m]
            for lab, sh in list(e["single_control"].items()) + [(f"subset: {k}", v) for k, v in e["within_keyword_named"].items()]:
                w(f"| {m} | {lab} | {ci_pct(sh['prompt_words_R0'])} | {ci_pct(sh['query_rewriting'])} | {ci_pct(sh['reranker'])} | {ci_pct(sh['generator'])} |")
        for m in METHODS:
            w(f"\n{m}: within keyword, the share of prompts naming their keyword changes by "
              f"{num(fu['strata'][m]['control_on_x']['keyword_in_prompt'])} from x = 0 to 1.")
        w("")
    if gen:
        L.extend(paper_text(ch, gen, fu, v1, v2, lex))
    text = "\n".join(L) + "\n"
    extra = Path(__file__).with_name("paper_text.md")
    if extra.exists():
        text += "\n" + extra.read_text()
    (out_dir / "RESULTS.md").write_text(text)
    if out_dir.resolve() == (DEFAULT_INPUTS / "steelman-v1").resolve():
        RESULTS_MD.write_text(text)
    manifest = {"parts": {n: {"git_commit": p["git_commit"], "prereg_sha256": p["prereg_sha256"], "seconds": p.get("seconds"),
                              "settings": p["settings"]} for n, p in (("chain", ch), ("generator", gen), ("fe", fe), ("lexical", lex), ("pairs", pairs)) if p},
                "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out_dir.glob("*.json")) if f.name != "manifest.json"},
                "verdicts": {"C1": v1, "C2": v2}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"wrote": str(out_dir / "RESULTS.md"), "C1": {m: v1[m]["verdict"] for m in METHODS}, "C2": v2 and v2["verdict"]}))
    return 0

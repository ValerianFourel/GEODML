"""Turn the full-run result bundle into the paper's number macros and appendix tables.

Reads the held-out (confirmation) steelman parts, the funnel stage models and the generator
decision models of `~/Hamburg/geodml-inputs/fullrun-run2` and writes

* `numbers-heldout.tex`: one LaTeX macro per number the body quotes (both generators, held-out
  keywords), each line carrying the file it came from;
* `A3-generated-tables.tex`: the appendix tables (stage models, keep/order models, exact
  decomposition, pre-registered verdicts, intent-in-text slopes);
* `dossier.md`: every extracted value in reading order, for checking the prose.

Usage: python -m analysis.scripts.paper_numbers_fullrun <bundle dir> <output dir> [--coordinates <answer_coordinates.jsonl.gz>]
Every value is read from a file; nothing is computed except ratios the files already define and, with
--coordinates, the first- and last-bin stage means carried onto the prompt scale (labelled [A]).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CELLS = [  # paper column order (numbers.tex convention)
    ("llama4 · Parallel", "LP", "Llama $\\cdot$ Parallel"),
    ("llama4 · Reactive", "LR", "Llama $\\cdot$ Reactive"),
    ("qwen38 · Parallel", "QP", "Qwen $\\cdot$ Parallel"),
    ("qwen38 · Reactive", "QR", "Qwen $\\cdot$ Reactive"),
]
STEPS = [  # chain increments / shares, in pipeline order
    ("prompt_words_R0", "Words", "Prompt words ($U \\to R_0$)"),
    ("query_rewriting", "Queries", "Query rewriting ($R_0 \\to R$)"),
    ("reranker", "Rerank", "Shortlist ($C \\to P$)"),
    ("shown_order", "Order", "Shown order ($P \\to I$)"),
    ("generator", "Gen", "Generator's own choices ($I \\to K$)"),
]
STAGES_TEX = [("R0", "Rz"), ("R", "R"), ("C", "C"), ("P", "P"), ("I", "I"), ("K", "K")]
FUNNEL_STAGES = [("R|U", "Retrieval $R\\mid U$"), ("R0|U", "Prompt text $R_0\\mid U$"),
                 ("P|C", "Shortlist $P\\mid C$"), ("K|P", "Ranking $K\\mid P$")]
BLOCKS = [("A1", "Intent"), ("A2", "Topic and same keyword"), ("A3", "Snippet text"),
          ("A4", "URL string"), ("B", "Engine signals"), ("C1", "Off-page SEO"),
          ("C2", "Page body (unseen)")]
# feature display names and the block they print under (order = table order)
FEATURES = [
    ("A1", "intent_alignment", "Alignment $-|u-x|$"),
    ("A1", "intent_x_prompt", "Page intent $\\times(x-\\frac12)$"),
    ("A1", "page_intent_z", "Page intent ($z$ scale)"),
    ("A2", "topic_similarity", "Topic similarity"),
    ("A2", "on_keyword", "Same keyword"),
    ("A3", "snip_text_words", "Snippet words"),
    ("A3", "snip_title_chars", "Title length"),
    ("A3", "snip_digits", "Digits"),
    ("A3", "snip_percent", "Percent sign"),
    ("A3", "snip_year", "Year"),
    ("A3", "snip_currency", "Currency sign"),
    ("A3", "snip_title_question", "Question title"),
    ("A3", "snip_title_listicle", "Listicle title"),
    ("A3", "snip_names_domain", "Names the domain"),
    ("A4", "url_https", "https"),
    ("A4", "url_path_depth", "Path depth"),
    ("A4", "url_length", "URL length"),
    ("A4", "url_has_query", "Query string"),
    ("A4", "url_subdomain", "Subdomain"),
    ("A4", "url_tld_com", ".com"),
    ("A4", "url_tld_org", ".org"),
    ("A4", "url_tld_edu_gov", ".edu or .gov"),
    ("A4", "url_user_content", "User-content platform"),
    ("A4", "url_wikipedia", "Wikipedia"),
    ("B", "stored_position", "Stored engine position"),
    ("B", "searxng_score", "SearXNG score"),
    ("B", "searxng_engine_count", "SearXNG engine count"),
    ("C1", "dfs_organic_count", "Domain size (organic keywords)"),
    ("C1", "dfs_organic_top1", "Top-1 organic keywords"),
    ("C1", "dfs_traffic_value", "Traffic value"),
    ("C1", "dfs_paid_count", "Paid keywords"),
    ("C1", "dfs_domain_age", "Domain age"),
    ("C1", "open_pagerank", "Open PageRank"),
    ("C1", "has_llms_txt", "llms.txt present"),
    ("C1", "brand_list", "On brand list"),
    ("C1", "earned_list", "On review or press list"),
    ("C1", "google_top20_url", "URL in Google top 20"),
    ("C1", "google_top20_domain", "Domain in Google top 20"),
    ("C2", "body_word_count", "Word count"),
    ("C2", "body_readability", "Readability"),
    ("C2", "body_stats_density", "Statistics density"),
    ("C2", "body_question_headings", "Question headings"),
    ("C2", "body_modularity", "Modularity"),
    ("C2", "body_structured_data", "Structured data"),
    ("C2", "body_ext_citations", "External citations"),
    ("C2", "body_auth_citations", "Authority citations"),
    ("C2", "body_internal_links", "Internal links"),
    ("C2", "body_outbound_links", "Outbound links"),
    ("C2", "body_images_alt", "Images with alt text"),
    ("C2", "body_freshness", "Freshness"),
]
SHUFFLE_FEATURES = {"intent_alignment", "intent_x_prompt"}
TEX_BLOCK = {"A1": "Aone", "A2": "Atwo", "A3": "Athree", "A4": "Afour", "B": "B", "C1": "Cone", "C2": "Ctwo"}  # macro names take no digits


# ---------------------------------------------------------------- formatting
def neg(s: str) -> str:
    return f"\\ensuremath{{{s}}}" if "-" in s else s


def slope(v: float, d: int = 3) -> str:
    return neg(f"{v:+.{d}f}")


def ci(lo: float, hi: float, d: int = 3) -> str:
    return neg(f"[{lo:+.{d}f}, {hi:+.{d}f}]")


def pct(v: float) -> str:
    s = f"{100 * v:.0f}"
    return (f"\\ensuremath{{{s}}}" if s.startswith("-") else s) + "\\%"


def pct1(v: float) -> str:
    s = f"{100 * v:.1f}"
    s = "0.0" if s == "-0.0" else s
    return (f"\\ensuremath{{{s}}}" if s.startswith("-") else s) + "\\%"


def pctci(lo: float, hi: float) -> str:
    s = f"[{100 * lo:.0f}\\%, {100 * hi:.0f}\\%]"
    return f"\\ensuremath{{{s}}}" if "-" in s else s


def orci(f: dict) -> str:
    lo, hi = (math.exp(c) for c in f["ci95"])
    return f"{f['odds_ratio_per_sd']:.2f} [{lo:.2f}, {hi:.2f}]"


def num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pval(p) -> str:
    return "--" if p is None else f"{p:.3f}"


def est(d: dict, key="estimate"):
    return d[key]


# ---------------------------------------------------------------- loading
class Bundle:
    def __init__(self, root: Path, coordinates: Path | None = None):
        self.root = root
        self.knots = None
        if coordinates is not None:
            from analysis.scripts.acl_figures import prompt_scale_map, read_coordinates
            self.knots = prompt_scale_map(read_coordinates(coordinates))
        sc = root / "steelman-confirmation"
        self.chain = json.load(open(sc / "chain.json"))
        self.gen = json.load(open(sc / "generator.json"))
        self.supply = json.load(open(sc / "supply.json"))
        self.ablation = json.load(open(sc / "ablation.json"))
        self.queries = json.load(open(sc / "queries.json"))
        self.followup = json.load(open(sc / "followup.json"))
        self.lexical = json.load(open(sc / "lexical.json"))
        self.census = json.load(open(sc / "census.json"))
        self.fe = json.load(open(sc / "fe.json"))
        self.funnel = json.load(open(root / "paper-results/funnel-confirmation/results.json"))
        self.funnel_x = json.load(open(root / "paper-results/funnel-exploration/results.json"))
        self.dec = json.load(open(root / "paper-results/decisions-confirmation/decisions.json"))
        self.verdicts = json.load(open(root / "verdicts.json"))
        self.intent = json.load(open(root / "paper-results/intent-stages/results.json"))
        self.chain_x = json.load(open(root / "steelman-exploration/chain.json"))
        self.gen_x = json.load(open(root / "steelman-exploration/generator.json"))


# ---------------------------------------------------------------- macros
def macros(b: Bundle) -> tuple[list[str], list[str]]:
    out: list[str] = []
    doc: list[str] = []

    def m(name: str, value: str, src: str = ""):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f"  % src: {src}" if src else ""))

    def h(title: str):
        out.append(f"\n% ---- {title}")
        doc.append(f"\n## {title}\n")

    SC = "fullrun-run2/steelman-confirmation"
    PR = "fullrun-run2/paper-results"

    # ---- scope
    h("Scope of the held-out run [F]")
    cen = b.census
    m("NHeldOutKw", num(cen["keywords_with_answers"]["confirmation"]), f"{SC}/census.json keywords_with_answers")
    m("NExploreKw", num(cen["keywords_with_answers"]["exploration"]), f"{SC}/census.json keywords_with_answers")
    m("NHeldOutPrompts", num(cen["prompts_by_split"]["confirmation"]), f"{SC}/census.json prompts_by_split")
    m("NExplorePrompts", num(cen["prompts_by_split"]["exploration"]), f"{SC}/census.json prompts_by_split")
    m("NTraceAnswers", num(cen["extract_counts"]["answers"]), f"{SC}/census.json extract_counts (both models, all conditions)")
    m("NTraceAnswersLlama", num(cen["extract_counts"]["answers_llama4"]), f"{SC}/census.json")
    m("NTraceAnswersQwen", num(cen["extract_counts"]["answers_qwen38"]), f"{SC}/census.json")
    nat = {}
    for r in cen["table"]:
        if r["split"] == "confirmation" and r["condition"] == "natural":
            key = (r["model"], r["method"], r["measure"])
            nat[key] = nat.get(key, 0) + r["count"]
    for s, suf, _ in CELLS:
        model, method = s.split(" · ")
        m(f"NNatural{suf}", num(nat[(model, method, "extracted")]), f"{SC}/census.json natural extracted, both engines")
        m(f"NCited{suf}", num(nat[(model, method, "cited_at_least_one")]), f"{SC}/census.json natural cited_at_least_one")
    tot_nat = sum(v for (mo, me, ms), v in nat.items() if ms == "extracted")
    m("NNaturalHeldOut", num(tot_nat), f"{SC}/census.json natural extracted, confirmation split, both models")
    doc.append(f"natural held-out answers extracted: {tot_nat}; by cell {[(k, v) for k, v in nat.items() if k[2] == 'extracted']}")

    # ---- chain
    h("Chain: stage slopes, increments and shares of beta_K, common sample [F]")
    for s, suf, _ in CELLS:
        c = b.chain["strata"][s]["chain"]["common"]
        m(f"NCommon{suf}", num(c["answers"]), f"{SC}/chain.json strata.{s}.chain.common.answers")
        for st, stex in STAGES_TEX:
            v = c["slopes"][st]
            m(f"Sl{stex}{suf}", slope(v["estimate"]), f"{SC}/chain.json slopes.{st}")
            m(f"CISl{stex}{suf}", ci(*v["ci95"]), f"{SC}/chain.json slopes.{st} ci95")
        for key, stex, _ in STEPS:
            inc = c["increments"][key]
            sh = c["shares"][key]
            m(f"Inc{stex}{suf}", slope(inc["estimate"]), f"{SC}/chain.json increments.{key}")
            m(f"CIInc{stex}{suf}", ci(*inc["ci95"]), f"{SC}/chain.json increments.{key} ci95")
            m(f"Sh{stex}{suf}", pct(sh["estimate"]), f"{SC}/chain.json shares.{key}")
            m(f"CISh{stex}{suf}", pctci(*sh["ci95"]), f"{SC}/chain.json shares.{key} ci95")
            doc.append(f"{s} {key}: inc {inc['estimate']:+.4f} {inc['ci95']} p={inc.get('permutation_p')}; share {sh['estimate']:.3f} {sh['ci95']}")
        adm = c["shares"]["admission_P"]
        m(f"Adm{suf}", pct(adm["estimate"]), f"{SC}/chain.json shares.admission_P")
        m(f"CIAdm{suf}", pctci(*adm["ci95"]), f"{SC}/chain.json shares.admission_P ci95")
        gvr = c["shares"]["generator_vs_random"]
        m(f"ShGenRand{suf}", pct(gvr["estimate"]), f"{SC}/chain.json shares.generator_vs_random")
        m(f"CIShGenRand{suf}", pctci(*gvr["ci95"]), f"{SC}/chain.json shares.generator_vs_random ci95")
        ded = c["shares"]["dedup_condition"]
        doc.append(f"{s} admission {adm['estimate']:.3f} {adm['ci95']}; dedup share {ded['estimate']:.4f}; slopes "
                   + ", ".join(f"{st} {c['slopes'][st]['estimate']:+.4f} {c['slopes'][st]['ci95']}" for st, _ in STAGES_TEX))
        # controls variant and lattice target
        for variant, vsuf in (("controls", "Ctrl"), ("target", "Tgt")):
            cv = b.chain["strata"][s]["chain"][variant]
            for key, stex, _ in STEPS:
                sh = cv["shares"][key]
                m(f"Sh{stex}{vsuf}{suf}", pct(sh["estimate"]), f"{SC}/chain.json chain.{variant}.shares.{key}")
                m(f"CISh{stex}{vsuf}{suf}", pctci(*sh["ci95"]), f"{SC}/chain.json chain.{variant}.shares.{key} ci95")
            m(f"Adm{vsuf}{suf}", pct(cv["shares"]["admission_P"]["estimate"]), f"{SC}/chain.json chain.{variant}.shares.admission_P")
            m(f"CIAdm{vsuf}{suf}", pctci(*cv["shares"]["admission_P"]["ci95"]), f"{SC}/chain.json chain.{variant}")
            m(f"SlK{vsuf}{suf}", slope(cv["slopes"]["K"]["estimate"]), f"{SC}/chain.json chain.{variant}.slopes.K")
            m(f"CISlK{vsuf}{suf}", ci(*cv["slopes"]["K"]["ci95"]), f"{SC}/chain.json chain.{variant}.slopes.K ci95")
            doc.append(f"{s} {variant}: " + ", ".join(f"{key} {cv['shares'][key]['estimate']:.3f} {cv['shares'][key]['ci95']}" for key, _, _ in STEPS)
                       + f"; admission {cv['shares']['admission_P']['estimate']:.3f}; K {cv['slopes']['K']['estimate']:+.4f}")
        st = b.chain["strata"][s]
        for key, name in (("cited_count", "CitedCount"), ("cited_share_of_shown", "CitedShare"), ("shown_count", "ShownCount")):
            v = st[key]
            d = 2 if key != "cited_share_of_shown" else 3
            m(f"{name}{suf}", slope(v["estimate"], d), f"{SC}/chain.json strata.{s}.{key}")
            m(f"CI{name}{suf}", ci(*v["ci95"], d), f"{SC}/chain.json strata.{s}.{key} ci95")
            m(f"Mean{name}{suf}", f"{v['mean']:.2f}", f"{SC}/chain.json strata.{s}.{key}.mean")
            doc.append(f"{s} {key}: {v['estimate']:+.4f} {v['ci95']} mean {v['mean']:.3f} answers {v['answers']}")
        m(f"DroppersShare{suf}", pct(st["droppers_share_of_answers"]), f"{SC}/chain.json droppers_share_of_answers")
        lk = st["loko"]
        doc.append(f"{s} loko generator {lk['generator']['min']:.4f}-{lk['generator']['max']:.4f}; admission {lk['admission_P']['min']:.4f}-{lk['admission_P']['max']:.4f}; "
                   f"query {lk['query_rewriting']['min']:.4f}-{lk['query_rewriting']['max']:.4f}; reranker {lk['reranker']['min']:.4f}-{lk['reranker']['max']:.4f}")
        nf = st["noise_floor"]
        doc.append(f"{s} noise floor: cells {nf['cells_with_two_or_more']} answers {nf['answers_in_such_cells']}; K within-cell slope {nf['terms']['K']['slope_within_cell']['estimate']:+.4f} {nf['terms']['K']['slope_within_cell']['ci95']}; sd within cell {nf['terms']['K']['sd_within_cell']:.3f} vs within keyword {nf['terms']['K']['sd_within_keyword']:.3f}")
    # engine strata generator shares (C1 narrowing for Llama Parallel)
    h("Engine strata: generator share K - I [F]")
    for es, v in sorted(b.chain["engine_strata"].items()):
        g = v["shares"]["generator"]
        model, method, engine = es.split(" · ")
        suf = {"llama4": "L", "qwen38": "Q"}[model] + method[0] + {"duckduckgo": "Ddg", "searxng": "Sx"}[engine]
        m(f"ShGen{suf}", pct(g["estimate"]), f"{SC}/chain.json engine_strata.{es}.shares.generator")
        m(f"CIShGen{suf}", pctci(*g["ci95"]), f"{SC}/chain.json engine_strata.{es} ci95")
        doc.append(f"{es}: generator share {g['estimate']:.3f} {g['ci95']}; answers {v['answers']}; query {v['increments']['query_rewriting']['estimate']:+.4f}; reranker {v['increments']['reranker']['estimate']:+.4f}")

    # ---- generator (C2)
    h("Generator: Delta_gen, keep/order coefficients, slot correlations [F]")
    m("SESOI", f"{b.gen['sesoi']:.3f}", f"{SC}/generator.json sesoi")
    for s, suf, _ in CELLS:
        g = b.gen["strata"][s]
        dmain = g["delta"]["main"]
        dg = dmain["delta_gen"]
        m(f"DeltaGen{suf}", slope(dg["estimate"], 4), f"{SC}/generator.json delta.main.delta_gen")
        m(f"CIninetyDeltaGen{suf}", ci(*dg["ci90"], 4), f"{SC}/generator.json delta.main.delta_gen ci90 (TOST)")
        m(f"CIDeltaGen{suf}", ci(*dg["ci95"], 4), f"{SC}/generator.json delta.main.delta_gen ci95")
        sh = dmain["delta_gen_share_of_K"]
        m(f"DeltaGenShare{suf}", pct(sh["estimate"]), f"{SC}/generator.json delta.main.delta_gen_share_of_K")
        m(f"CIDeltaGenShare{suf}", pctci(*sh["ci95"]), f"{SC}/generator.json delta.main.delta_gen_share_of_K ci95")
        m(f"KeepInformative{suf}", num(g["keep_informative_answers"]), f"{SC}/generator.json keep_informative_answers")
        m(f"KeepInformativeShare{suf}", pct(g["keep_informative_share"]), f"{SC}/generator.json keep_informative_share")
        doc.append(f"{s}: delta_gen {dg['estimate']:+.4f} ci90 {dg['ci90']} ci95 {dg['ci95']}; share {sh['estimate']:.3f} {sh['ci95']}; "
                   f"keep modelled {g['keep_modelled']} informative {g['keep_informative_answers']} ({g['keep_informative_share']:.3f}); failed {g['delta']['failed_replicates']}; "
                   f"variants {[k for k in g['delta'] if k != 'failed_replicates']}")
        for variant in g["delta"]:
            if variant == "main" or not isinstance(g["delta"][variant], dict):
                continue
            dv = g["delta"][variant]["delta_gen"]
            doc.append(f"   {variant}: {dv['estimate']:+.4f} ci95 {dv['ci95']}")
        for cset, cv in g["coefficients"].items():
            doc.append(f"   coef {cset}: " + ", ".join(f"{k} {v['estimate']:+.3f} {[round(x, 3) for x in v['ci95']]}" for k, v in cv.items()
                                                        if isinstance(v, dict) and "estimate" in v)
                       + "; other keys " + str([k for k, v in cv.items() if not (isinstance(v, dict) and "estimate" in v)]))
        sc_ = g["slot_correlations"]
        doc.append("   slot corr: " + ", ".join(f"{k} {v['estimate']:+.3f}" for k, v in sc_.items()))
    # fixed effects
    h("Two-way fixed effects (answer x snapshot row) [F]")
    for s, suf, _ in CELLS:
        for outcome, ov in b.fe["strata"][s].items():
            co = ov["coefficients_per_sd"]
            doc.append(f"{s} {outcome}: rows {ov['rows_used']} snippets {ov['distinct_snapshot_rows']} mean {ov['outcome_mean']:.3f}; "
                       + ", ".join(f"{k} {v['estimate']:+.4f} {[round(x, 4) for x in v['ci95']]}" for k, v in co.items()))

    # ---- decisions (Pratt shares, pseudo R2, counts, ORs)
    h("Generator keep/order decision models, held-out [F]")
    for s, suf, _ in CELLS:
        d = b.dec["strata"][s]
        c = d["counts"]
        m(f"NShown{suf}", num(c["shown"]), f"{PR}/decisions-confirmation/decisions.json counts.shown")
        m(f"NKept{suf}", num(c["kept"]), f"{PR}/decisions-confirmation/decisions.json counts.kept")
        m(f"ShareKept{suf}", pct(c["share_kept"]), f"{PR}/decisions-confirmation/decisions.json counts.share_kept")
        m(f"NKeepAnswers{suf}", num(c["keep_informative_answers"]), f"{PR}/decisions-confirmation/decisions.json counts.keep_informative_answers")
        m(f"NOrderPicks{suf}", num(c["order_choice_sets"]), f"{PR}/decisions-confirmation/decisions.json counts.order_choice_sets")
        m(f"NDecAnswers{suf}", num(c["answers"]), f"{PR}/decisions-confirmation/decisions.json counts.answers")
        doc.append(f"{s} counts: {c}")
        for dec in ("keep", "order"):
            dd = d[dec]
            D = dec.capitalize()
            m(f"Rsq{D}{suf}", f"{dd['pseudo_r2']:.2f}", f"{PR}/decisions-confirmation/decisions.json {dec}.pseudo_r2")
            bs = dd["block_shares"]
            for blk, bname in BLOCKS:
                m(f"Blk{TEX_BLOCK[blk]}{D}{suf}", pct1(bs[blk]), f"{PR}/decisions-confirmation/decisions.json {dec}.block_shares.{blk}")
            m(f"BlkSlot{D}{suf}", pct1(bs["slot"]), f"{PR}/decisions-confirmation/decisions.json {dec}.block_shares.slot")
            m(f"TopicSlot{D}{suf}", pct(bs["A2"] + bs["slot"]), f"[A] A2 + slot from {PR}/decisions-confirmation/decisions.json {dec}.block_shares")
            doc.append(f"{s} {dec}: pseudoR2 {dd['pseudo_r2']:.3f}; blocks " + ", ".join(f"{k} {v:.3f}" for k, v in bs.items())
                       + f"; failed {dd['failed_replicates']}")
            f = dd["features"]
            for feat, name in (("topic_similarity", "Topic"), ("on_keyword", "SameKw"), ("intent_alignment", "IntAlign"),
                               ("intent_x_prompt", "IntX"), ("page_intent_z", "IntZ"), ("dfs_organic_count", "Domain")):
                m(f"OR{name}{D}{suf}", orci(f[feat]), f"{PR}/decisions-confirmation/decisions.json {dec}.features.{feat}")
                doc.append(f"   {feat}: OR {orci(f[feat])} p {f[feat]['permutation_p']}")
            if dec == "keep" and "shown_slot_1" in f:
                m(f"ORSecondSlot{suf}", orci(f["shown_slot_1"]), f"{PR}/decisions-confirmation/decisions.json keep.features.shown_slot_1 (slots indexed from 0: the second shown slot)")
                doc.append("   slots: " + ", ".join(f"{k} {orci(v)}" for k, v in f.items() if k.startswith("shown_slot_")))
            if dec == "order" and dd["position_effects"]:
                pe = dd["position_effects"]
                m(f"OrderLogOddsDrop{suf}", f"{abs(pe[min(6, len(pe) - 1)]):.1f}", f"{PR}/decisions-confirmation/decisions.json order.position_effects[6] (seventh shown slot against the first)")
                doc.append(f"   order position effects: {[round(x, 2) for x in pe]}")

    # ---- funnel stage models (held-out)
    h("Funnel stage models, held-out: odds ratios per SD [F]")
    main = b.funnel["models"]["main"]
    for s, suf, _ in CELLS:
        for stage, _n in FUNNEL_STAGES:
            ssuf = {"R|U": "Ret", "R0|U": "Rz", "P|C": "Short", "K|P": "Rank"}[stage]
            f = main[s][stage]["features"]
            for feat, name in (("dfs_organic_count", "Domain"), ("stored_position", "EnginePos"), ("topic_similarity", "Topic"),
                               ("intent_alignment", "IntAlign"), ("intent_x_prompt", "IntX"), ("page_intent_z", "IntZ")):
                if feat in f:
                    m(f"OR{name}{ssuf}{suf}", orci(f[feat]), f"{PR}/funnel-confirmation/results.json models.main.{s}.{stage}.features.{feat}")
                    if feat in SHUFFLE_FEATURES:
                        m(f"P{name}{ssuf}{suf}", pval(f[feat]["permutation_p"]), f"{PR}/funnel-confirmation/results.json ... permutation_p")
            blocks = main[s][stage]["blocks"]
            for blk, _b in BLOCKS:
                m(f"Fit{TEX_BLOCK[blk]}{ssuf}{suf}", pct1(blocks[blk]["fit_share"]), f"{PR}/funnel-confirmation/results.json models.main.{s}.{stage}.blocks.{blk}.fit_share")
            q = b.funnel["negative_control_null"][s][stage]["quantile"]
            m(f"Bar{ssuf}{suf}", f"{q:.3f}", f"{PR}/funnel-confirmation/results.json negative_control_null.{s}.{stage}.quantile")
            n_sets = main[s][stage].get("choice_sets", main[s][stage].get("answers"))
            if n_sets is not None:
                m(f"NChoice{ssuf}{suf}", num(n_sets), f"{PR}/funnel-confirmation/results.json ... choice_sets")
            doc.append(f"{s} {stage}: choice sets {n_sets} alternatives {main[s][stage].get('alternatives')} keys {sorted(main[s][stage].keys())}; bar {q:.3f}; blocks "
                       + ", ".join(f"{k} {v['fit_share']:.3f}" for k, v in blocks.items()) + "; "
                       + ", ".join(f"{feat} {orci(f[feat])} p={f[feat]['permutation_p']}" for feat in ("intent_alignment", "intent_x_prompt", "page_intent_z", "topic_similarity", "dfs_organic_count", "stored_position") if feat in f))
    # specifications: complete-case and visible, headline features
    h("Other specifications (visible, complete-case) [F]")
    for spec in ("visible", "complete"):
        for s, suf, _ in CELLS:
            for stage, _n in FUNNEL_STAGES:
                f = b.funnel["models"][spec][s][stage]["features"]
                doc.append(f"{spec} {s} {stage}: " + ", ".join(f"{feat} {orci(f[feat])}" for feat in ("topic_similarity", "intent_alignment", "dfs_organic_count") if feat in f))
                ssuf = {"R|U": "Ret", "R0|U": "Rz", "P|C": "Short", "K|P": "Rank"}[stage]
                for feat, name in (("topic_similarity", "Topic"), ("dfs_organic_count", "Domain"), ("intent_alignment", "IntAlign")):
                    if feat in f:
                        m(f"OR{name}{ssuf}{spec.capitalize()}{suf}", orci(f[feat]), f"{PR}/funnel-confirmation/results.json models.{spec}.{s}.{stage}.features.{feat}")
    # exploration-split domain at retrieval (did not replicate)
    h("Exploration split, for the replication sentence [F]")
    mx = b.funnel_x["models"]["main"]
    for s, suf, _ in CELLS:
        f = mx[s]["R|U"]["features"]["dfs_organic_count"]
        m(f"ORDomainRetExplore{suf}", orci(f), f"{PR}/funnel-exploration/results.json models.main.{s}.R|U.features.dfs_organic_count")
        doc.append(f"exploration {s} domain at R|U {orci(f)}")
    # decomposition
    h("Exact decomposition of log RR(K|U), held-out [F]")
    for s, suf, _ in CELLS:
        for feat, name in (("intent_alignment", "Align"), ("topic_similarity", "Topic"), ("dfs_organic_count", "Domain")):
            dd = b.funnel["decomposition"][s][feat]
            sh = dd["share"]["shortlist|C"]
            lo, hi = dd["share_ci95"]["shortlist|C"]
            m(f"{name}ShortShare{suf}", pct(sh), f"{PR}/funnel-confirmation/results.json decomposition.{s}.{feat}.share.shortlist|C")
            m(f"CI{name}ShortShare{suf}", pctci(lo, hi), f"{PR}/funnel-confirmation/results.json decomposition ... share_ci95.shortlist|C")
            m(f"{name}RankShare{suf}", pct(dd["share"]["ranking|P"]), f"{PR}/funnel-confirmation/results.json decomposition.{s}.{feat}.share.ranking|P")
            m(f"{name}TotalLogRR{suf}", slope(dd["total_log_rr_K_given_U"]), f"{PR}/funnel-confirmation/results.json decomposition ... total_log_rr_K_given_U")
            doc.append(f"{s} {feat}: steps " + ", ".join(f"{k} {v:+.3f}" for k, v in dd["log_rr"].items())
                       + f"; total {dd['total_log_rr_K_given_U']:+.3f} {dd['ci95_total']}; shares " + ", ".join(f"{k} {v:.3f}" for k, v in dd["share"].items())
                       + f"; shortlist share ci {dd['share_ci95']['shortlist|C']}; answers {dd['answers']}")
    # confirmatory family P1-P4 and contrasts
    h("Pre-registered funnel predictions P1-P4, held-out [F]")
    for name, v in b.funnel["confirmatory"].items():
        strata = v["strata"]
        if isinstance(strata, list):
            items = [(x["stratum"], x["estimate"], x["ci95"], x.get("p")) for x in strata]
        else:
            items = [(k, x["estimate"], x["ci95"], None) for k, x in strata.items()]
        doc.append(f"{name}: replicates={v.get("replicates")}; " + "; ".join(f"{k} {e:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] p={p}" for k, e, c, p in items))
        if "c2_null" in v:
            doc.append("   c2_null K|P quantiles: " + ", ".join(f"{k} {x['K|P']['quantile']:.3f}" for k, x in v["c2_null"].items()))
    # verdicts (checker) for both splits
    h("Checker verdicts, both splits [F]")
    for split in ("confirmation", "exploration"):
        vv = b.verdicts[split]
        doc.append(f"{split}: C1 " + ", ".join(f"{k}: {x['verdict']}" for k, x in vv["C1"].items())
                   + "; C2 " + ", ".join(f"{k}: {x['verdict']}" for k, x in vv["C2"].items())
                   + f"; supply {json.dumps(vv.get('supply'))[:400]}")
        for k, x in vv["C1"].items():
            doc.append(f"   C1 {k}: upper95 s_gen {x['checks']['generator_share_upper95']:.3f}; reasons {x['reasons']}")
        for k, x in vv["C2"].items():
            doc.append(f"   C2 {k}: {json.dumps(x['checks'])}")
    for s, suf, _ in CELLS:
        m(f"VerdictCone{suf}", b.verdicts["confirmation"]["C1"][s]["verdict"], "fullrun-run2/verdicts.json confirmation.C1")
    for model, suf in (("llama4", "L"), ("qwen38", "Q")):
        m(f"VerdictCtwo{suf}", b.verdicts["confirmation"]["C2"][model]["verdict"], "fullrun-run2/verdicts.json confirmation.C2")
    sgen_up = {s: b.verdicts["confirmation"]["C1"][s]["checks"]["generator_share_upper95"] for s, _, _ in CELLS}
    for s, suf, _ in CELLS:
        m(f"SgenUpper{suf}", pct(sgen_up[s]), "fullrun-run2/verdicts.json confirmation.C1 checks.generator_share_upper95")

    # ---- supply
    h("Supply of action-ready pages, held-out [F]")
    sp = b.supply
    m("RowsUgeEight", f"{100 * sp['rows']['share_rows_u_ge_0.8']:.2f}\\%", f"{SC}/supply.json rows.share_rows_u_ge_0.8")
    m("RowsUgeSix", pct(sp["rows"]["share_rows_u_ge_0.6"]), f"{SC}/supply.json rows.share_rows_u_ge_0.6")
    m("KwAnyUgeEight", f"{100 * sp['keywords']['share_keywords_any_u_ge_0.8']:.1f}\\%", f"{SC}/supply.json keywords.share_keywords_any_u_ge_0.8")
    m("KwAnyUgeSix", pct(sp["keywords"]["share_keywords_any_u_ge_0.6"]), f"{SC}/supply.json keywords.share_keywords_any_u_ge_0.6")
    doc.append(f"rows {sp['rows']}; keywords {sp['keywords']}; tercile rule {sp['tercile_rule']}")
    gaps, utils = [], []
    for s, suf, _ in CELLS:
        st = sp["strata"][s]
        cg = st["ceiling_gap_top_x"]
        ut = st["utilisation"]["U"]
        up = st["utilisation"]["P"]
        tc = st["tercile_contrast_top_minus_bottom"]
        m(f"CeilingGap{suf}", f"{cg['estimate']:.2f}", f"{SC}/supply.json strata.{s}.ceiling_gap_top_x")
        m(f"CICeilingGap{suf}", f"[{cg['ci95'][0]:.2f}, {cg['ci95'][1]:.2f}]", f"{SC}/supply.json ceiling_gap_top_x ci95")
        m(f"UtilU{suf}", f"{ut['estimate']:.2f}", f"{SC}/supply.json strata.{s}.utilisation.U")
        m(f"CIUtilU{suf}", f"[{ut['ci95'][0]:.2f}, {ut['ci95'][1]:.2f}]", f"{SC}/supply.json utilisation.U ci95")
        m(f"UtilP{suf}", f"{up['estimate']:.2f}", f"{SC}/supply.json strata.{s}.utilisation.P")
        m(f"TercileContrast{suf}", slope(tc["estimate"]), f"{SC}/supply.json strata.{s}.tercile_contrast_top_minus_bottom")
        m(f"CITercileContrast{suf}", ci(*tc["ci95"]), f"{SC}/supply.json tercile_contrast ci95")
        m(f"OracleUSlope{suf}", slope(st["slopes"]["oracle_U"]["estimate"]), f"{SC}/supply.json strata.{s}.slopes.oracle_U")
        m(f"OraclePSlope{suf}", slope(st["slopes"]["oracle_P"]["estimate"]), f"{SC}/supply.json strata.{s}.slopes.oracle_P")
        gaps.append(cg["estimate"]); utils.append(ut["estimate"])
        doc.append(f"{s}: answers {st['answers']}; ceiling gap {cg['estimate']:.3f} {cg['ci95']} (n {cg['answers']}); util U {ut['estimate']:.3f} {ut['ci95']}; util P {up['estimate']:.3f}; "
                   f"tercile contrast {tc['estimate']:+.4f} {tc['ci95']}; slopes " + ", ".join(f"{k} {v['estimate']:+.3f}" for k, v in st["slopes"].items()))
        doc.append("   deciles: " + "; ".join(f"x {d['mean_x']:.2f} K {d['mean_K']:.3f} oP {d['mean_oracle_P']:.3f} oU {d['mean_oracle_U']:.3f}" for d in st["deciles"]))
    m("CeilingGapRange", f"{min(gaps):.2f}--{max(gaps):.2f}", f"[A] range over the four strata, {SC}/supply.json ceiling_gap_top_x")
    m("UtilURange", f"{min(utils):.2f}--{max(utils):.2f}", f"[A] range over the four strata, {SC}/supply.json utilisation.U")

    # ---- ablation
    h("Ablated condition: pass-through [F]")
    pts = []
    for s, suf, _ in CELLS:
        a = b.ablation["strata"][s]
        pt = a["pass_through_dK_on_dP"]
        m(f"PassThrough{suf}", f"{pt['estimate']:.2f}", f"{SC}/ablation.json strata.{s}.pass_through_dK_on_dP")
        m(f"CIPassThrough{suf}", f"[{pt['ci95'][0]:.2f}, {pt['ci95'][1]:.2f}]", f"{SC}/ablation.json pass_through ci95")
        m(f"NPairs{suf}", num(a["pairs"]), f"{SC}/ablation.json strata.{s}.pairs")
        m(f"ShortlistChanged{suf}", pct(a["share_shortlist_intent_changed"]), f"{SC}/ablation.json share_shortlist_intent_changed")
        pts.append(pt["estimate"])
        doc.append(f"{s}: pairs {a['pairs']}; pass-through {pt['estimate']:.3f} {pt['ci95']}; shortlist changed {a['share_shortlist_intent_changed']:.3f}; "
                   f"mean dP {a['mean_delta_P']['estimate']:+.5f}; mean dK {a['mean_delta_K']['estimate']:+.5f}; slope dK on x {a['slope_delta_K_on_x']['estimate']:+.5f} {a['slope_delta_K_on_x']['ci95']}")
    m("PassThroughRange", f"{min(pts):.2f}--{max(pts):.2f}", f"[A] range over the four strata, {SC}/ablation.json")

    # ---- queries and keyword naming
    h("Agent queries: keyword share, prompts naming the keyword [F]")
    for s, suf, _ in CELLS:
        q = b.queries["strata"][s]
        for grp, gsuf in (("all_prompts", "All"), ("prompts_naming_keyword", "Named"), ("prompts_omitting_keyword", "Omit")):
            v = q[grp]
            m(f"KwQ{gsuf}Low{suf}", f"{v['mean_x_below_0.2']:.2f}", f"{SC}/queries.json strata.{s}.{grp}.mean_x_below_0.2")
            m(f"KwQ{gsuf}High{suf}", f"{v['mean_x_at_least_0.8']:.2f}", f"{SC}/queries.json strata.{s}.{grp}.mean_x_at_least_0.8")
            m(f"KwQ{gsuf}Slope{suf}", slope(v["slope"]["estimate"], 2), f"{SC}/queries.json strata.{s}.{grp}.slope")
            m(f"CIKwQ{gsuf}Slope{suf}", ci(*v["slope"]["ci95"], 2), f"{SC}/queries.json ... slope ci95")
            m(f"NKwQ{gsuf}{suf}", num(v["answers"]), f"{SC}/queries.json strata.{s}.{grp}.answers")
            doc.append(f"{s} {grp}: n {v['answers']}; low {v['mean_x_below_0.2']:.3f} high {v['mean_x_at_least_0.8']:.3f}; slope {v['slope']['estimate']:+.3f} {v['slope']['ci95']}")
        fu = b.followup["strata"][s]
        kn = fu["control_on_x"]["keyword_in_prompt"]
        m(f"PromptNamesKwDrop{suf}", slope(kn["estimate"], 2), f"{SC}/followup.json strata.{s}.control_on_x.keyword_in_prompt (within-keyword change from x=0 to 1)")
        m(f"CIPromptNamesKwDrop{suf}", ci(*kn["ci95"], 2), f"{SC}/followup.json ... ci95")
        m(f"PromptNamesKwMean{suf}", pct(kn["mean"]), f"{SC}/followup.json ... keyword_in_prompt.mean")
        lw = fu["control_on_x"]["log_prompt_words"]
        doc.append(f"{s}: prompt names keyword: mean {kn['mean']:.3f}, change on x {kn['estimate']:+.3f} {kn['ci95']}; log words change {lw['estimate']:+.3f}")
        for sub in ("prompt_names_keyword", "prompt_omits_keyword"):
            w = fu["within_keyword_named"][sub]
            doc.append(f"   subset {sub}: " + ", ".join(f"{k} {w[k]['estimate']:.3f} {w[k]['ci95']}" for k in ("prompt_words_R0", "query_rewriting", "reranker", "generator", "admission_P")))
        for ctrl, w in fu["single_control"].items():
            doc.append(f"   control {ctrl}: query {w['query_rewriting']['estimate']:.3f} {w['query_rewriting']['ci95']}; generator {w['generator']['estimate']:.3f}")
        # lexical selector
        lx = b.lexical["strata"][s]["lexical_selector"]
        for key, lsuf in (("reranker_text", "Rtext"), ("user_prompt", "Prompt"), ("bm25_preregistered|reranker_text", "Bmtext")):
            if key in lx:
                v = lx[key]["lexical_share_of_reranker"]
                m(f"LexShare{lsuf}{suf}", pct(v["estimate"]), f"{SC}/lexical.json strata.{s}.lexical_selector.{key}.lexical_share_of_reranker")
                m(f"CILexShare{lsuf}{suf}", pctci(*v["ci95"]), f"{SC}/lexical.json ... ci95")
                doc.append(f"{s} lexical {key}: share of reranker step {v['estimate']:.3f} {v['ci95']}; increment {lx[key]['lexical_increment']['estimate']:+.4f}")
        doc.append(f"{s} lexical keys: {list(lx.keys())}")

    # ---- intent in text (all keywords; z scale)
    h("Intent in text: stage slopes on the z scale, all 1,011 keywords [F]")
    for s, v in b.intent["strata"].items():
        sl = v["slopes"]
        doc.append(f"{s}: answers {v['answers']}; " + ", ".join(f"{k} {x['slope']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for k, x in sl.items() if k in ("Q", "R0", "R", "C", "P", "K", "A", "G")))
        if "mediated_by_pool" in v["derived"]:
            mp = v["derived"]["mediated_by_pool"]
            doc.append(f"   mediated_by_pool {mp['estimate']:.3f} {mp['ci95']}; pool - reordering {v['derived']['pool_minus_reordering']['estimate']:+.3f} {v['derived']['pool_minus_reordering']['ci95']}")
    doc.append(f"intent verdicts: {b.intent['verdicts']}")
    val = b.intent["validity"]
    m("NQueriesEmbedded", num(val["queries"]["texts"]), f"{PR}/intent-stages/results.json validity.queries.texts")
    m("QueriesViewAgreement", f"{val['queries']['spearman']:.2f}", f"{PR}/intent-stages/results.json validity.queries.spearman (two LLM2Vec views)")
    m("NAnswersEmbedded", num(val["answers"]["texts"]), f"{PR}/intent-stages/results.json validity.answers.texts")
    m("AnswersViewAgreement", f"{val['answers']['spearman']:.2f}", f"{PR}/intent-stages/results.json validity.answers.spearman")
    doc.append(f"validity queries {val['queries']}; answers {val['answers']}; pages {val['pages']}")
    if b.knots is not None:
        import numpy as np
        h("Stage means in the lowest and highest bin of x, prompt scale, all keywords [A: z means mapped through the prompt map]")
        for model, msuf in (("llama4", "L"), ("qwen38", "Q")):
            group = b.intent["curves"]["groups"][f"{model} · both engines"]["natural"]
            for term, scale in (("Q", "z"), ("A", "z"), ("R0", "z"), ("R", "u"), ("P", "u"), ("K", "u")):
                means = group[scale]["terms"][term]["mean"]
                lo, hi = means[0], means[-1]
                if scale == "z":
                    lo, hi = (float(np.interp(v, *b.knots)) for v in (lo, hi))
                tsuf = {"R0": "Rz"}.get(term, term)
                src = f"{PR}/intent-stages/results.json curves.groups[{model} · both engines].natural.{scale}.terms.{term}.mean[0], [-1]" + (" mapped through the prompt map [A]" if scale == "z" else "")
                m(f"Stage{tsuf}Low{msuf}", f"{lo:.2f}", src)
                m(f"Stage{tsuf}High{msuf}", f"{hi:.2f}", src)
                doc.append(f"{model} {term}: lowest bin {lo:.3f}, highest bin {hi:.3f} ({scale}{' mapped' if scale == 'z' else ''})")
    return out, doc


# ---------------------------------------------------------------- appendix tables
def stage_tables(b: Bundle) -> str:
    """One full odds-ratio table per generator x method, held-out keywords."""
    main = b.funnel["models"]["main"]
    parts = []
    first_label = None
    for s, suf, title in CELLS:
        label = f"tab:a3-stage-{suf.lower()}"
        first_label = first_label or label
        rows = []
        rows.append("\\begin{table*}[tp]\n\\centering\\scriptsize\n\\renewcommand{\\arraystretch}{0.94}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule")
        rows.append(" & Retrieval & Literal prompt search & Shortlist & Ranking \\\\\n & $R\\mid U$ & $R_0\\mid U$ & $P\\mid C$ & $K\\mid P$ \\\\\n\\midrule")
        stages = ["R|U", "R0|U", "P|C", "K|P"]
        current_block = None
        for blk, feat, name in FEATURES:
            if blk != current_block:
                current_block = blk
                rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{dict(BLOCKS)[blk]}}}}} \\\\")
            cells = []
            pcells = []
            for stage in stages:
                f = main[s][stage]["features"]
                if feat in f:
                    cells.append(orci(f[feat]))
                    pcells.append(pval(f[feat]["permutation_p"]))
                else:
                    cells.append("--")
                    pcells.append("--")
            rows.append(f"\\quad {name} & " + " & ".join(cells) + " \\\\")
            if feat in SHUFFLE_FEATURES:
                rows.append("\\qquad shuffle $p$ & " + " & ".join(pcells) + " \\\\")
        rows.append("\\midrule\n\\multicolumn{5}{@{}l}{\\textit{Fit lost when the block is dropped}} \\\\")
        for blk, bname in BLOCKS:
            rows.append(f"\\quad {bname} & " + " & ".join(pct1(main[s][st]["blocks"][blk]["fit_share"]) for st in stages) + " \\\\")
        rows.append("\\quad Page-body bar, 95th pct.\\ of $|\\beta|$ & " + " & ".join(f"{b.funnel['negative_control_null'][s][st]['quantile']:.3f}" for st in stages) + " \\\\")
        rows.append("\\quad Choice sets & " + " & ".join(num(main[s][st].get("choice_sets", main[s][st].get("answers", 0))) for st in stages) + " \\\\")
        rows.append("\\bottomrule\n\\end{tabular}")
        if label == first_label:
            cap = (f"Stage models for {title}, held-out keywords, all blocks fitted together. Cells give the odds ratio per SD with its 95\\% keyword-bootstrap interval; "
                   "shuffle $p$ cannot fall below 0.005. $R_0\\mid U$ refits retrieval with inclusion defined as the frozen search's top \\NRowsPerQuery{} rows for the prompt text; "
                   "$K\\mid P$ treats the cited sources as ordered picks from the shortlist, mixing keep and order (Tables~\\ref{tab:a3-keep-a}--\\ref{tab:a3-keep-b} separate them). "
                   "A dash marks a feature absent from the model (every row of $U$ is the keyword's own). The lower panel gives the fit lost when each block is dropped, "
                   "the page-body bar (\\S\\ref{sec:unread}) and the number of choice sets. Missing-data indicators and two further indicators are fitted but not shown.")
        else:
            cap = f"Stage models for {title}, held-out keywords; layout and notes as in Table~\\ref{{{first_label}}}."
        rows.append(f"\\caption{{{cap}}}\n\\label{{{label}}}\n\\end{{table*}}\n")
        src = f"% src: {b.root.name}/paper-results/funnel-confirmation/results.json models.main[{s}] (features, blocks.fit_share, choice_sets); negative_control_null[{s}]"
        parts.append(src + "\n" + "\n".join(rows))
    return "\n".join(parts)


def keep_order_tables(b: Bundle) -> str:
    """Keep models (three strata) and order models (four strata), held-out keywords: one table pair per decision."""
    dec = b.dec["strata"]
    part_a, part_b = [], []
    for decision, cells in (("keep", [c for c in CELLS if c[0] != "qwen38 · Parallel"]), ("order", CELLS)):
        ncol = len(cells)
        title = "Keep" if decision == "keep" else "Order"

        def head():
            return ("\\begin{table*}[tp]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{0.94}\n"
                    f"\\begin{{tabular}}{{@{{}}l*{{{ncol}}}{{c}}@{{}}}}\n\\toprule\n"
                    f" & \\multicolumn{{{ncol}}}{{c@{{}}}}{{{title} model}} \\\\\n\\cmidrule(l){{2-{1 + ncol}}}\n"
                    " & " + " & ".join(t for _, _, t in cells) + " \\\\\n\\midrule")

        def feature_row(feat, name, shuffle=False):
            vals, ps = [], []
            for s, _, _ in cells:
                f = dec[s][decision]["features"]
                vals.append(orci(f[feat]) if feat in f else "--")
                ps.append(pval(f[feat]["permutation_p"]) if feat in f else "--")
            rows = [f"\\quad {name} & " + " & ".join(vals) + " \\\\"]
            if shuffle:
                rows.append("\\qquad shuffle $p$ & " + " & ".join(ps) + " \\\\")
            return rows

        count_key = "keep_informative_answers" if decision == "keep" else "order_choice_sets"
        rows = [head()]
        rows.append("Decisions modelled & " + " & ".join(num(dec[s]["counts"][count_key]) for s, _, _ in cells) + " \\\\")
        rows.append("Pseudo-$R^2$ & " + " & ".join(f"{dec[s][decision]['pseudo_r2']:.2f}" for s, _, _ in cells) + " \\\\")
        rows.append(f"\\midrule\n\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{Pratt share of the explained part, by block}}}} \\\\")
        for blk, bname in BLOCKS + [("slot", "Shown slot")]:
            rows.append(f"\\quad {bname} & " + " & ".join(pct1(dec[s][decision]["block_shares"][blk]) for s, _, _ in cells) + " \\\\")
        rows.append(f"\\midrule\n\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{Odds ratio per SD [95\\% interval]}}}} \\\\")
        current = None
        for blk, feat, name in FEATURES:
            if blk not in ("A1", "A2", "A3", "A4", "B"):
                continue
            if blk != current:
                current = blk
                rows.append(f"\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{{dict(BLOCKS)[blk]}}}}} \\\\")
            rows.extend(feature_row(feat, name, shuffle=feat in SHUFFLE_FEATURES))
        if decision == "keep":
            rows.append(f"\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{Shown slot: odds ratio per SD of the 0/1 indicator, against slot 1}}}} \\\\")
            max_slots = max(len([k for k in dec[s]["keep"]["features"] if k.startswith("shown_slot_")]) for s, _, _ in cells)
            for i in range(1, max_slots + 1):
                vals = [orci(dec[s]["keep"]["features"][f"shown_slot_{i}"]) if f"shown_slot_{i}" in dec[s]["keep"]["features"] else "--" for s, _, _ in cells]
                rows.append(f"\\quad Slot {i + 1} & " + " & ".join(vals) + " \\\\")
        else:
            rows.append(f"\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{Shown slot: term on the log-odds scale, against slot 1}}}} \\\\")
            max_pos = max(len(dec[s]["order"]["position_effects"]) for s, _, _ in cells)
            for i in range(1, max_pos):
                vals = []
                for s, _, _ in cells:
                    pe = dec[s]["order"]["position_effects"]
                    vals.append(neg(f"{pe[i]:+.2f}") if i < len(pe) else "--")
                rows.append(f"\\quad Slot {i + 1} & " + " & ".join(vals) + " \\\\")
        rows.append("\\bottomrule\n\\end{tabular}")
        if decision == "keep":
            cap = ("Keep models on the held-out keywords, part~1: a conditional logit over each answer's shown snippets given how many it keeps, fitted to the "
                   "informative answers; block shares are Pratt shares as in Table~\\ref{tab:shares}; odds ratios and shuffle $p$ read as in Table~\\ref{tab:a3-stage-lp}. "
                   "Qwen under Parallel Expansion keeps \\ShareKeptQP{} of shown links and has no keep model.")
        else:
            cap = ("Order models on the held-out keywords, part~1: Plackett--Luce over kept snippets, one pick per choice set; layout as in Table~\\ref{tab:a3-keep-a}.")
        rows.append(f"\\caption{{{cap}}}\n\\label{{tab:a3-{decision}-a}}\n\\end{{table*}}\n")
        part_a.append(f"% src: {b.root.name}/paper-results/decisions-confirmation/decisions.json strata.*.counts, {decision} (block_shares, features, position_effects)\n" + "\n".join(rows))
        # part b: blocks C1 and C2
        rows = [head()]
        current = None
        for blk, feat, name in FEATURES:
            if blk not in ("C1", "C2"):
                continue
            if blk != current:
                current = blk
                rows.append(f"\\multicolumn{{{ncol + 1}}}{{@{{}}l}}{{\\textit{{{dict(BLOCKS)[blk]}}}}} \\\\")
            rows.extend(feature_row(feat, name))
        rows.append("\\bottomrule\n\\end{tabular}")
        no_keep = (" Qwen under Parallel Expansion keeps \\ShareKeptQP{} of shown links, so its keep decision is held at the observed set and not modelled."
                   if decision == "keep" else "")
        rows.append(f"\\caption{{{title} models on the held-out keywords, part~2 (off-page SEO and the unseen page body), read as Table~\\ref{{tab:a3-{decision}-a}}.{no_keep} "
                    "Missing-data indicators and two further indicators are fitted but not shown.}\n" + f"\\label{{tab:a3-{decision}-b}}\n\\end{{table*}}\n")
        part_b.append(f"% src: {b.root.name}/paper-results/decisions-confirmation/decisions.json strata.*.{decision} features of blocks C1 and C2\n" + "\n".join(rows))
    return "\n".join(part_a + part_b)


def decomposition_table(b: Bundle) -> str:
    rows = ["\\begin{table*}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule",
            "Cell & Feature & Retrieval & Shortlist & Ranking & Total log RR & Shortlist share & Ranking share \\\\",
            " & & $R\\mid U$ & $P\\mid C$ & $K\\mid P$ & $K\\mid U$ & & \\\\\n\\midrule"]
    feats = (("intent_alignment", "Alignment $-|u-x|$"), ("topic_similarity", "Topic similarity"), ("dfs_organic_count", "Domain size"))
    for s, suf, title in CELLS:
        for i, (feat, name) in enumerate(feats):
            d = b.funnel["decomposition"][s][feat]
            lr = d["log_rr"]
            tot = d["total_log_rr_K_given_U"]
            lo, hi = d["ci95_total"]
            cell = title if i == 0 else ""
            rows.append(f"{cell} & {name} & {neg(f'{lr['retrieval|U']:+.3f}')} & {neg(f'{lr['shortlist|C']:+.3f}')} & {neg(f'{lr['ranking|P']:+.3f}')} & "
                        f"{neg(f'{tot:+.3f}')} {ci(lo, hi)} & {pct(d['share']['shortlist|C'])} {pctci(*d['share_ci95']['shortlist|C'])} & {pct(d['share']['ranking|P'])} \\\\")
        if s != CELLS[-1][0]:
            rows.append("\\addlinespace[2pt]")
    rows.append("\\bottomrule\n\\end{tabular}")
    rows.append("\\caption{Exact decomposition of the log risk ratio (RR) that a row of $U$ is cited, top against bottom quartile of the feature within each answer's own rows "
                "(answers weighted equally), held-out keywords. The step terms sum to the total (95\\% keyword-bootstrap interval) together with the scoring term $C\\mid R$, "
                "which lies within $\\pm 0.006$ in every row and is omitted (the Reactive Loop scores every retrieved row, so there it is zero); the last two columns are the "
                "shortlist and ranking terms' shares of the total.}\n\\label{tab:a3-decomp}\n\\end{table*}\n")
    return f"% src: {b.root.name}/paper-results/funnel-confirmation/results.json decomposition[*].{{intent_alignment,topic_similarity,dfs_organic_count}} (log_rr, total_log_rr_K_given_U, ci95_total, share, share_ci95)\n" + "\n".join(rows)


def verdict_table(b: Bundle) -> str:
    """Pre-registered claims and their held-out verdicts, with the decisive quantities."""
    v = b.verdicts["confirmation"]
    rows = ["\\begin{table*}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{1.5pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule",
            " & " + " & ".join(t for _, _, t in CELLS) + " \\\\\n\\midrule",
            "\\multicolumn{5}{@{}l}{\\textit{C1, admission (per generator and method)}} \\\\"]
    rows.append("Generator's share & " + " & ".join(f"\\ShGen{suf}{{}} \\CIShGen{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("\\dots with prompt controls & " + " & ".join(f"\\ShGenCtrl{suf}{{}} \\CIShGenCtrl{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("\\dots DuckDuckGo / SearXNG & " + " & ".join(f"\\ShGen{suf[0]}{suf[1]}Ddg{{}} / \\ShGen{suf[0]}{suf[1]}Sx{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Query rewriting adds & " + " & ".join(f"\\IncQueries{suf}{{}} \\CIIncQueries{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Shortlist adds & " + " & ".join(f"\\IncRerank{suf}{{}} \\CIIncRerank{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Verdict & " + " & ".join(f"\\textbf{{\\VerdictCone{suf}}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("\\midrule\n\\multicolumn{5}{@{}l}{\\textit{C2, selection (per generator, both methods)}} \\\\")
    rows.append("$\\Delta_{\\mathrm{gen}}$ (90\\% interval) & " + " & ".join(f"\\DeltaGen{suf}{{}} \\CIninetyDeltaGen{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("\\dots as a share of $\\beta_K$ & " + " & ".join(f"\\DeltaGenShare{suf}{{}} \\CIDeltaGenShare{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Keep decision & " + " & ".join(("modelled (\\KeepInformative" + suf + "{})") if b.gen["strata"][s]["keep_modelled"] else "held at the observed set" for s, suf, _ in CELLS) + " \\\\")
    rows.append("Verdict (per generator) & \\multicolumn{2}{c}{\\textbf{\\VerdictCtwoL}} & \\multicolumn{2}{c}{\\textbf{\\VerdictCtwoQ}} \\\\")
    rows.append("\\midrule\n\\multicolumn{5}{@{}l}{\\textit{Supply (every cell)}} \\\\")
    rows.append("Ceiling gap at $x\\geq 0.9$ & " + " & ".join(f"\\CeilingGap{suf}{{}} \\CICeilingGap{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Tercile contrast & " + " & ".join(f"\\TercileContrast{suf}{{}} \\CITercileContrast{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Utilisation & " + " & ".join(f"\\UtilU{suf}{{}} \\CIUtilU{suf}{{}}" for _, suf, _ in CELLS) + " \\\\")
    rows.append("Verdict & \\multicolumn{4}{c}{\\textbf{narrowed} (every stratum)} \\\\")
    rows.append("\\bottomrule\n\\end{tabular}")
    rows.append("\\caption{The pre-registered claims and their verdicts on the \\NHeldOutKw{} held-out keywords (natural condition; brackets are 95\\% keyword-bootstrap intervals unless stated). "
                "C1 is supported if the 95\\% upper bound of the generator's share lies below 1/3 on the common sample, with prompt controls and in each engine, and query rewriting and the shortlist both add slope; "
                "C2 if the 90\\% interval of $\\Delta_{\\mathrm{gen}}$ lies inside $\\pm$\\SESOI{} in both methods, narrowed if the 95\\% bound of $|\\Delta_{\\mathrm{gen}}|$ stays below 0.030; "
                "supply if the ceiling gap at $x\\geq 0.9$ is above 0, the top minus bottom supply-tercile contrast of $\\beta_K$ is above 0 and utilisation of the own-row oracle is at least 0.5, narrowed if only the gap holds. "
                "Generator's share is $(\\beta_K-\\beta_I)/\\beta_K$; the engine row gives point estimates; the keep decision is modelled (informative answers in parentheses) or held at the observed set; utilisation is $\\beta_K/\\beta_{\\mathrm{oracle}(U)}$. The rules were fixed before any analysis ran (Appendix~\\ref{app:protocol}); the exploration keywords give the same verdicts.}\n\\label{tab:a3-verdicts}\n\\end{table*}\n")
    return f"% src: {b.root.name}/verdicts.json (confirmation); macros from steelman-confirmation/{{chain,generator,supply}}.json\n" + "\n".join(rows)


def funnel_predictions_table(b: Bundle) -> str:
    conf = b.funnel["confirmatory"]
    names = {"P1 alignment at R|U": ("P1", "alignment at $R\\mid U$"),
             "P2 alignment R − R0": ("P2", "alignment, $R$ minus $R_0$"),
             "P3 domain authority K|P − P|C": ("P3", "domain size, $K$ minus $P$"),
             "P4 page intent × x at R|U": ("P4", "intent $\\times(x-\\frac12)$ at $R\\mid U$")}
    rows = ["\\begin{table*}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule",
            "Prediction (coefficient per SD) & " + " & ".join(t for _, _, t in CELLS) + " \\\\\n\\midrule"]
    for key, (tag, name) in names.items():
        v = conf[key]
        strata = v["strata"]
        if isinstance(strata, list):
            by = {x["stratum"]: x for x in strata}
        else:
            by = strata
        cells = []
        for s, _, _ in CELLS:
            x = by[s]
            cells.append(f"{neg(f'{x['estimate']:+.3f}')} {ci(*x['ci95'])}")
        rows.append(f"{tag}: {name} & " + " & ".join(cells) + " \\\\")
    rows.append("\\bottomrule\n\\end{tabular}")
    rows.append("\\caption{The funnel study's pre-registered predictions on the held-out keywords (coefficient per SD with its 95\\% keyword-bootstrap interval). A prediction replicates if, in all four cells, "
                "the coefficient has the predicted sign, its interval excludes 0 and its shuffle $p<0.05$; P3 also requires the generator coefficient to clear the page-body bar. None met the rule.}\n\\label{tab:a3-predictions}\n\\end{table*}\n")
    return f"% src: {b.root.name}/paper-results/funnel-confirmation/results.json confirmatory (P1-P4: estimate, ci95, replicates)\n" + "\n".join(rows)


def intent_text_table(b: Bundle) -> str:
    """Stage slopes on the axis's native z scale per model x engine x method (intent-stages results)."""
    strata = b.intent["strata"]
    order = [k for k in strata if k.count("·") == 2]
    stages = [("Q", "Agent queries $Q$"), ("R0", "Prompt-text search $R_0$"), ("R", "Retrieved $R$"), ("P", "Shortlist $P$"), ("K", "Cited sources $K$"), ("A", "Answer text $A$")]
    rows = ["\\begin{table*}[t]\n\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}l*{8}{>{\\centering\\arraybackslash}p{1.42cm}}@{}}\n\\toprule"]
    heads = []
    for k in order:
        model, engine, method = k.split(" · ")
        heads.append(f"{'Llama' if model == 'llama4' else 'Qwen'} $\\cdot$ {'DDG' if engine == 'duckduckgo' else 'SX'} $\\cdot$ {method[0]}")
    rows.append(" & " + " & ".join(heads) + " \\\\\n\\midrule")
    for st, name in stages:
        cells = []
        for k in order:
            x = strata[k]["slopes"].get(st)
            cells.append("--" if x is None else f"{neg(f'{x['slope']:+.2f}')}\\newline{{\\tiny {ci(*x['ci95'], 2)}}}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule\nAnswers & " + " & ".join(num(strata[k]["answers"]) for k in order) + " \\\\")
    rows.append("\\bottomrule\n\\end{tabular}")
    rows.append("\\caption{Intent measured in the text at each stage: within-keyword slope on $x$ of the stage's mean position on the axis's native scale (consensus value, $z$), all \\NKeywords{} keywords, natural condition, "
                "by generator, engine snapshot (DDG DuckDuckGo, SX SearXNG) and method (P Parallel Expansion, R Reactive Loop); 95\\% keyword-bootstrap intervals. "
                "Queries and answers are the agent's own texts embedded with the same two encoders; page stages are means of page positions. Figure~\\ref{fig:stages} shows the same stages on the percentile scale.}\n\\label{tab:a3-intent-text}\n\\end{table*}\n")
    return f"% src: {b.root.name}/paper-results/intent-stages/results.json strata[model · engine · method].slopes (slope, ci95), answers\n" + "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="unpacked full-run bundle (steelman-*/, paper-results/, verdicts.json)")
    parser.add_argument("output", type=Path, help="directory for numbers-heldout.tex, A3-generated-tables.tex, dossier.md")
    parser.add_argument("--coordinates", type=Path, help="answers' (z, percentile) rows for the prompt-scale map")
    args = parser.parse_args(argv)
    root = args.bundle.expanduser()
    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    b = Bundle(root, args.coordinates)
    macro_lines, doc = macros(b)
    header = ("% numbers-heldout.tex -- generated by analysis/scripts/paper_numbers_fullrun.py from the full-run bundle\n"
              f"% ({root.name}: both generators, 733 held-out keywords, natural condition). Do not edit by hand; rerun the script.\n"
              "% Cell suffixes: LP Llama/Parallel, LR Llama/Reactive, QP Qwen/Parallel, QR Qwen/Reactive. [F] unless a src says [A].\n")
    (out / "numbers-heldout.tex").write_text(header + "\n".join(macro_lines) + "\n")
    tables = "\n".join([
        "% A3-generated-tables.tex -- generated by analysis/scripts/paper_numbers_fullrun.py; do not edit by hand.\n",
        "\\newcommand{\\AthreeVerdictTable}{%\n" + verdict_table(b) + "}\n",
        "\\newcommand{\\AthreePredictionTable}{%\n" + funnel_predictions_table(b) + "}\n",
        "\\newcommand{\\AthreeDecompTable}{%\n" + decomposition_table(b) + "}\n",
        "\\newcommand{\\AthreeIntentTextTable}{%\n" + intent_text_table(b) + "}\n",
        "\\newcommand{\\AthreeStageTables}{%\n" + stage_tables(b) + "}\n",
        "\\newcommand{\\AthreeKeepOrderTables}{%\n" + keep_order_tables(b) + "}\n",
    ])
    (out / "A3-generated-tables.tex").write_text(tables)
    (out / "dossier.md").write_text("# Full-run numbers dossier\n" + "\n".join(doc) + "\n")
    print(f"wrote {len(macro_lines)} macros to {out / 'numbers-heldout.tex'}; tables and dossier alongside")


if __name__ == "__main__":
    main()

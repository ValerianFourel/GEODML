#!/usr/bin/env python3
"""One folder with every result the paper rests on, and a Markdown report that reads them.

  python3 paper_results.py --input funnel=<report folder> --input decisions=<folder> ... --output <folder>

Each --input is label=path; the path is a folder (its results.json / decisions.json / explore.json /
review.json / keywords-summary.json / contrasts.json / manifest.json are copied) or one JSON file. The
report lists, per label, what was found and the headline numbers that file format carries, every number
read from the copied file; a label whose path is missing is reported as missing, never invented. The
manifest records sha256 and bytes of every copied file and the code commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

FILES = ("results.json", "decisions.json", "explore.json", "review.json", "keywords-summary.json", "contrasts.json",
         "manifest.json", "importance.json", "report.json")
STRATA = ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive")


def ci(entry, key="slope", d=3):
    lo, hi = entry.get("ci95") or [None, None]
    v = entry.get(key)
    if v is None:
        return "—"
    return f"{v:+.{d}f}" + (f" [{lo:+.{d}f}, {hi:+.{d}f}]" if lo is not None else "")


def orf(entry):
    lo, hi = entry.get("ci95") or [None, None]
    p = f", p {entry['permutation_p']:.3f}" if entry.get("permutation_p") is not None else ""
    return f"{entry['odds_ratio_per_sd']:.2f}" + (f" [{math.exp(lo):.2f}, {math.exp(hi):.2f}{p}]" if lo is not None else "")


def section_funnel(r: dict) -> list[str]:
    s = r.get("settings", {})
    out = [f"Split **{s.get('split')}**, {s.get('bootstrap')} draws, {s.get('permutations')} shuffles, specs {s.get('specs')}, "
           f"analysis commit {str(r.get('analysis_commit', r.get('git_commit', '')))[:7]}, exploratory {s.get('exploratory')}."]
    main = r.get("models", {}).get("main", {})
    out.append("")
    out.append("| stratum | stage | intent alignment | intent × x | topic similarity | domain organic (C1) | C2 null |β| 95th |")
    out.append("|---|---|---|---|---|---|---|")
    for st in [x for x in STRATA if x in main] + [x for x in main if x not in STRATA]:
        for stage in ("R|U", "R0|U", "P|C", "K|P"):
            f = main[st].get(stage, {}).get("features") or {}
            if not f:
                continue
            null = ((r.get("negative_control_null") or {}).get(st) or {}).get(stage) or {}
            out.append(f"| {st} | {stage} | {orf(f['intent_alignment']) if 'intent_alignment' in f else '—'} | "
                       f"{orf(f['intent_x_prompt']) if 'intent_x_prompt' in f else '—'} | {orf(f['topic_similarity']) if 'topic_similarity' in f else '—'} | "
                       f"{orf(f['dfs_organic_count']) if 'dfs_organic_count' in f else '—'} | {null.get('quantile', float('nan')):.3f} |")
    out.append("")
    out.append("Confirmatory family (four-stratum rule unless stated):")
    for name, v in (r.get("confirmatory") or {}).items():
        if isinstance(v, dict) and "strata" in v:
            strata = v["strata"]
            items = strata if isinstance(strata, list) else [{"stratum": k, **e} for k, e in strata.items()]
            cells = "; ".join(f"{e.get('stratum')}: {e.get('estimate', float('nan')):+.3f} [{(e.get('ci95') or [float('nan')]*2)[0]:+.3f}, "
                              f"{(e.get('ci95') or [float('nan')]*2)[1]:+.3f}]" for e in items)
            out.append(f"- {name}: replicates = {v.get('replicates')}; {cells}")
    dec = r.get("decomposition") or {}
    if dec:
        out.append("")
        out.append("Exact decomposition, log RR(ranked | keyword rows), top vs bottom quartile:")
        out.append("| stratum | feature | retrieval | scored | shortlist | ranking | total [95% CI] |")
        out.append("|---|---|---|---|---|---|---|")
        for st, feats in dec.items():
            for feat in ("intent_alignment", "topic_similarity", "dfs_organic_count"):
                d = feats.get(feat)
                if not d or not d.get("answers"):
                    continue
                lr = d["log_rr"]; lo, hi = d.get("ci95_total") or [float("nan")] * 2
                out.append(f"| {st} | {feat} | {lr.get('retrieval|U', float('nan')):+.3f} | {lr.get('reranker_candidate|R', float('nan')):+.3f} | "
                           f"{lr.get('shortlist|C', float('nan')):+.3f} | {lr.get('ranking|P', float('nan')):+.3f} | "
                           f"{d['total_log_rr_K_given_U']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    return out


def section_decisions(r: dict) -> list[str]:
    out = [f"Split **{r.get('split')}**, {r.get('bootstrap')} draws, {r.get('permutations')} shuffles, commit {str(r.get('git_commit', ''))[:7]}."]
    out.append("")
    out.append("| stratum | shown → kept (share) | decision | pseudo-R² | topic | shown slot | snippet | URL | off-page | page body | intent |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for st, e in (r.get("strata") or {}).items():
        c = e.get("counts", {})
        kept = f"{c.get('shown', 0):,} → {c.get('kept', 0):,} ({c.get('share_kept', float('nan')):.0%})"
        for d in ("keep", "order"):
            f = e.get(d, {})
            if "pseudo_r2" not in f:
                out.append(f"| {st} | {kept} | {d} | {f.get('skipped', 'missing')} | | | | | | | |")
                continue
            b = f.get("block_shares", {})
            out.append(f"| {st} | {kept} | {d} | {f['pseudo_r2']:.3f} | {b.get('A2', 0):.1%} | {b.get('slot', 0):.1%} | {b.get('A3', 0):.1%} | "
                       f"{b.get('A4', 0):.1%} | {b.get('C1', 0):.1%} | {b.get('C2', 0):.1%} | {b.get('A1', 0):.1%} |")
    out.append("")
    for st, e in (r.get("strata") or {}).items():
        for d in ("keep", "order"):
            f = e.get(d, {}).get("features") or {}
            if f:
                out.append(f"- {st} {d}: intent alignment {orf(f['intent_alignment'])}; intent × x {orf(f['intent_x_prompt'])}; "
                           f"topic similarity {orf(f['topic_similarity'])}; domain organic {orf(f['dfs_organic_count']) if 'dfs_organic_count' in f else '—'}")
    return out


def section_slopes(r: dict, metrics) -> list[str]:
    slopes = r.get("slopes") or {}
    strata = [x for x in STRATA if x in slopes] + [x for x in slopes if x not in STRATA]
    out = [f"Answers {r.get('answers', '?'):,}; natural {r.get('natural', r.get('natural_answers', '?')):,}; keywords {r.get('keywords', '?')}."
           if isinstance(r.get("answers"), int) else ""]
    out.append("| measure | " + " | ".join(strata) + " |")
    out.append("|---|" + "---|" * len(strata))
    for m in metrics:
        if any(m in slopes[s] for s in strata):
            out.append(f"| {m} | " + " | ".join(ci(slopes[s][m]) + f" (n={slopes[s][m].get('n', '?'):,})" if m in slopes[s] else "—" for s in strata) + " |")
    return out


def section_keywords(r: dict) -> list[str]:
    lo, hi = r.get("pooled_mean_slope_ci95") or [float("nan")] * 2
    out = [f"{r.get('keywords_with_slope')} keywords; random-effects mean slope {r.get('pooled_mean_slope', float('nan')):+.4f} "
           f"[{lo:+.4f}, {hi:+.4f}]; τ² {r.get('between_keyword_variance_tau2', float('nan')):.5f}; I² {r.get('i2', float('nan')):.2f}; "
           f"Q {r.get('heterogeneity_q', float('nan')):.0f} on {r.get('heterogeneity_df')} df, permutation p {r.get('heterogeneity_permutation_p', float('nan')):.4f} "
           f"({r.get('permutations')} shuffles, null max {r.get('null_q_max', float('nan')):.0f}); shrunken interval above 0 for "
           f"{r.get('share_keywords_shrunk_interval_above_0', float('nan')):.1%}, below 0 for {r.get('share_keywords_shrunk_interval_below_0', float('nan')):.1%}."]
    for group in ("by_intent_class", "by_difficulty_tercile"):
        for k, v in (r.get(group) or {}).items():
            if v:
                out.append(f"- {group} {k}: {ci(v)} ({v.get('keywords')} keywords, {v.get('answers'):,} answers)")
    g = r.get("gemma") or {}
    out.append(f"- Gemma (development): supplied {g.get('supplied')}, joined {g.get('graded_answers_joined')}, unjoined {g.get('graded_answers_unjoined')}")
    return out


def section_contrasts(r: dict) -> list[str]:
    out = [f"{r.get('answers', '?'):,} answers, {r.get('bootstrap')} shared draws."]
    for name, ms in (r.get("contrasts") or {}).items():
        parts = [f"{m} {ci(e, 'difference')}" for m, e in ms.items() if m in ("cited_u", "alignment_gain", "cited_on_topic", "ranking_len")]
        out.append(f"- {name}: " + "; ".join(parts))
    return out


def section_generic(r: dict) -> list[str]:
    keys = list(r)[:40]
    out = ["Top-level keys: " + ", ".join(keys)]
    for k in ("counts", "settings", "summary", "scientific_result", "judge_status", "replication"):
        if k in r:
            out.append(f"- {k}: {json.dumps(r[k], default=str)[:600]}")
    return out


def describe(label: str, name: str, r: dict) -> list[str]:
    if name == "results.json" and "models" in r and "decomposition" in r:
        return section_funnel(r)
    if name == "decisions.json":
        return section_decisions(r)
    if name == "review.json":
        return section_slopes(r, ("cited_u", "cited_gap", "null_gap", "alignment_gain", "cited_on_topic", "cited_glued", "ranking_len", "shown",
                                  "answer_chars", "answer_imperatives", "cited_google_url", "cited_dfs_organic_count", "cited_body_structured_data"))
    if name == "explore.json":
        return section_slopes(r, ("R0_u", "R_u", "C_u", "P_u", "K_u", "R_offtopic", "P_offtopic", "K_offtopic", "R0_recovered",
                                  "q_keyword_inclusion", "q_action_share"))
    if name == "keywords-summary.json":
        return section_keywords(r)
    if name == "contrasts.json":
        return section_contrasts(r)
    return section_generic(r)


def build(args) -> int:
    target = Path(args.output)
    for candidate in (target, target.with_name(target.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")
    partial = readiness.new_directory(target)
    manifest = {"created_at": readiness.now(), "git_commit": readiness.git_commit(), "inputs": {}, "files": {}}
    report = [f"# Paper results bundle ({manifest['created_at'][:10]}, code {manifest['git_commit'][:7]})", "",
              "Every number below is read from the copied file named in its section. Missing inputs are listed as missing."]
    for spec in args.input:
        label, _, path = spec.partition("=")
        source = Path(path)
        found = []
        if source.is_dir():
            found = [source / f for f in FILES if (source / f).exists()]
        elif source.is_file():
            found = [source]
        manifest["inputs"][label] = {"path": str(source), "found": [f.name for f in found]}
        report += ["", f"## {label}", ""]
        if not found:
            report.append(f"**missing**: {source}")
            continue
        (partial / label).mkdir()
        for f in found:
            shutil.copy(f, partial / label / f.name)
            manifest["files"][f"{label}/{f.name}"] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size,
                                                      "source": str(f.resolve())}
            if f.name.endswith(".json") and f.name != "manifest.json":
                try:
                    r = json.loads(f.read_text())
                except json.JSONDecodeError as error:
                    report.append(f"- {f.name}: unreadable ({error})")
                    continue
                report.append(f"### {f.name}")
                report += describe(label, f.name, r)
                report.append("")
            elif f.name == "manifest.json":
                m = json.loads(f.read_text())
                report.append(f"- manifest: stage {m.get('stage')}, commit {str(m.get('git_commit', ''))[:7]}, counts {json.dumps(m.get('counts'), default=str)[:400]}")
    readiness.write_json(partial / "manifest.json", manifest)
    (partial / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    partial.rename(target.resolve())
    print(json.dumps({"output": str(target), "inputs": {k: bool(v["found"]) for k, v in manifest["inputs"].items()}}), flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

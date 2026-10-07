#!/usr/bin/env python3
"""How the answer text changes along the prompt's information-seeking -> action-readiness axis.

Stages (all read-only on the generator datasets):

  export   CPU   verified answers of the chosen conditions (default: natural) with
                 their prompt's axis position; answers.jsonl.gz is directly usable
                 as --pages for `page_readiness_ordering.py embed` (GPU).
  text     CPU   text measures and within-keyword distinctive words -> report.md.
  analyze  CPU   after both LLM2Vec views are embedded and merged: places every
                 answer on the prompt scale through the same frozen maps, z-scaling,
                 rotation and percentile rule as the prompts -> report.md.

The prompt position is a measured property of the prompt text, not a randomized
treatment, so every result is a descriptive association. Answer coordinates use a
subspace fitted on prompts: they describe answer text out of domain, they are
neither treatments nor confounders. No judge results are used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts import report_generator_outputs as report

AXIS = report.AXIS
DESIGN = ("method", "engine")
WORD = re.compile(r"[a-z][a-z'’-]*[a-z]|[a-z]")
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")
LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
URL = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|org|net|io|co|gov|edu)\b")  # one per URL
CURRENCY = re.compile(r"[$€£¥]|\b(?:usd|eur|gbp|dollars?|euros?)\b")
# Heuristic marker lexicons; rates per 100 words. Descriptive text measures, not validated scales.
LEXICONS = {
    "second_person_per_100w": {"you", "your", "yours", "yourself", "you're", "you’ll", "you'll"},
    "action_verbs_per_100w": {"buy", "order", "book", "download", "install", "apply", "call", "contact", "visit",
                              "click", "register", "subscribe", "purchase", "choose", "select", "start", "try",
                              "schedule", "compare", "pick", "sign", "check", "follow", "go", "get", "use"},
    "immediacy_per_100w": {"today", "now", "immediately", "tonight", "quickly", "asap", "instantly", "soon"},
    "hedges_per_100w": {"may", "might", "could", "generally", "typically", "usually", "often", "depends",
                        "depending", "perhaps", "likely", "possibly"},
    "explanatory_per_100w": {"because", "means", "refers", "defined", "definition", "history", "overview",
                             "concept", "example", "therefore", "known"},
}
FEATURES = ("answer_chars", "words", "sentences", "words_per_sentence", "list_lines", "questions",
            "digits_per_100w", "currency_per_100w", "url_mentions", *LEXICONS)
PLACEMENT = ("answer_axis_percentile", "answer_minus_prompt_axis", "answer_outside_prompt_range")
PRIOR_MASS = 1000.0
MIN_WORD_COUNT = 50
TOP_WORDS = 30


def read_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_output(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} already exists; use a new path")
    path.mkdir(parents=True)
    return path


def answer_id(text):
    return "answer-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------- export

def export(datasets, axis_map, output, *, conditions=("natural",), stripes=256):
    """``datasets``: {model: root} or a list of (model, root) pairs; several roots of one model (a HoreKa dataset and a
    Hub import) are read in order and a cell seen twice (model, prompt, method, engine, condition) keeps its first copy.
    Written to ``<output>.partial`` and renamed at the end, so an interrupted export never blocks its rerun."""
    positions, axis = report.load_axis(axis_map)
    pairs = list(datasets.items()) if isinstance(datasets, dict) else [(m, Path(r)) for m, r in datasets]
    answers, observations, counts, seen = {}, [], Counter(), set()
    if Path(output).exists():
        raise ValueError(f"{output} already exists; use a new path")
    for model, root in pairs:
        cells, inventory = report.load_cells(root, model, stripes=stripes, keep_answer=True)
        counts[f"{model}_verified_cells"] += inventory["verified_cells"]
        for cell in cells:
            if cell["condition"] not in conditions:
                continue
            key = (model, cell["prompt_id"], cell["method"], cell["engine"], cell["condition"])
            if key in seen:
                counts[f"{model}_duplicate_cells_dropped"] += 1
                continue
            seen.add(key)
            text = cell["answer"].strip()
            if not text:
                counts[f"{model}_empty_answer"] += 1
                continue
            if cell["prompt_id"] not in positions:
                counts[f"{model}_prompt_without_axis"] += 1
                continue
            identity = answer_id(text)
            answers.setdefault(identity, text)
            observations.append({"model": model, "prompt_id": cell["prompt_id"], "keyword_id": cell["keyword_id"],
                                 "method": cell["method"], "engine": cell["engine"], "condition": cell["condition"],
                                 AXIS: positions[cell["prompt_id"]], "answer_id": identity,
                                 "ranking_length": cell["ranking_length"]})
            counts[f"{model}_answers"] += 1
    observations.sort(key=lambda r: (r["model"], r["prompt_id"], r["method"], r["engine"], r["condition"]))
    final = Path(output)
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.move(str(partial), str(final.with_name(f"{final.name}.partial.interrupted-{int(time.time())}")))
    output = new_output(partial)
    write_jsonl(output / "answers.jsonl.gz", ({"page_id": i, "text": t,
                                               "text_sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()}
                                              for i, t in sorted(answers.items())))
    write_jsonl(output / "observations.jsonl.gz", observations)
    manifest = {"format_version": "geodml-answer-readiness-export-v1", "git_commit": report._git_commit(),
                "datasets": [[m, str(Path(r).resolve())] for m, r in pairs], "axis_map": axis,
                "conditions": sorted(conditions), "unique_answers": len(answers),
                "observations": len(observations), "counts": dict(sorted(counts.items())),
                "files": {name: sha256_file(output / name) for name in ("answers.jsonl.gz", "observations.jsonl.gz")},
                "stored_answer_note": "stored answers are capped by the dataset (about 1,200 characters)",
                "scientific_result": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    output.rename(final)
    report.log("export", **manifest["counts"], unique_answers=len(answers))
    return manifest


def load_export(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"export file changed: {name}")
    texts = {row["page_id"]: row["text"] for row in read_jsonl(directory / "answers.jsonl.gz")}
    return manifest, texts, read_jsonl(directory / "observations.jsonl.gz")


# ---------------------------------------------------------------- text

def text_features(text):
    lowered = text.lower()
    words = WORD.findall(lowered)
    n = len(words)
    # List markers such as "1." are not sentence ends.
    sentences = max(1, len(SENTENCE_END.findall(LIST_LINE.sub("", text).strip() + " "))) if words else 0
    counts = Counter(words)
    per_100 = (lambda value: 100.0 * value / n) if n else (lambda value: None)
    return {"answer_chars": len(text), "words": n, "sentences": sentences,
            "words_per_sentence": n / sentences if sentences else None,
            "list_lines": len(LIST_LINE.findall(text)), "questions": text.count("?"),
            "digits_per_100w": per_100(len(re.findall(r"\d+(?:[.,]\d+)?", text))),
            "currency_per_100w": per_100(len(CURRENCY.findall(lowered))),
            "url_mentions": len(URL.findall(lowered)),
            **{name: per_100(sum(counts[w] for w in lexicon)) for name, lexicon in LEXICONS.items()}}


def within_keyword_halves(rows):
    """Label each row high/low relative to its keyword's median prompt position (ties dropped)."""
    by_keyword = defaultdict(set)
    for row in rows:
        by_keyword[row["keyword_id"]].add((row["prompt_id"], row[AXIS]))
    side = {}
    for prompts in by_keyword.values():
        median = statistics.median(x for _, x in prompts)
        for prompt, x in prompts:
            if x != median:
                side[prompt] = "high" if x > median else "low"
    return side


def within_keyword_contrasts(rows, outcomes, strata):
    """Mean over keywords of (high-half prompt mean - low-half prompt mean): topic held fixed."""
    side = within_keyword_halves(rows)
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if row["prompt_id"] in side:
            groups[tuple(str(row[s]) for s in strata)][row["keyword_id"]][side[row["prompt_id"]]].append(row)
    result = []
    for identity, keywords in sorted(groups.items()):
        for outcome in outcomes:
            differences = []
            for halves in keywords.values():
                means = {}
                for half in ("high", "low"):
                    by_prompt = defaultdict(list)
                    for row in halves.get(half, []):
                        if row[outcome] is not None:
                            by_prompt[row["prompt_id"]].append(row[outcome])
                    if by_prompt:
                        means[half] = statistics.fmean(statistics.fmean(v) for v in by_prompt.values())
                if len(means) == 2:
                    differences.append(means["high"] - means["low"])
            result.append({**dict(zip(strata, identity)), "outcome": outcome, "keywords": len(differences),
                           "mean_high_minus_low": statistics.fmean(differences) if differences else None,
                           "share_keywords_higher": (sum(d > 0 for d in differences) / len(differences)
                                                     if differences else None)})
    return result


def distinctive_words(rows, texts, *, model):
    """Informative-Dirichlet log-odds (Monroe et al. 2008) of high- vs low-half answers within keywords."""
    side = within_keyword_halves(rows)
    counts = {"high": Counter(), "low": Counter()}
    for row in rows:
        if row["prompt_id"] in side:
            counts[side[row["prompt_id"]]].update(WORD.findall(texts[row["answer_id"]].lower()))
    total = counts["high"] + counts["low"]
    corpus = sum(total.values())
    n_high, n_low = sum(counts["high"].values()), sum(counts["low"].values())
    scored = []
    for word, count in total.items():
        if count < MIN_WORD_COUNT:
            continue
        prior = PRIOR_MASS * count / corpus
        high, low = counts["high"][word], counts["low"][word]
        delta = (math.log((high + prior) / (n_high + PRIOR_MASS - high - prior))
                 - math.log((low + prior) / (n_low + PRIOR_MASS - low - prior)))
        z = delta / math.sqrt(1.0 / (high + prior) + 1.0 / (low + prior))
        scored.append({"model": model, "word": word, "z": z, "high_count": high, "low_count": low,
                       "high_per_10k": 1e4 * high / n_high if n_high else None,
                       "low_per_10k": 1e4 * low / n_low if n_low else None})
    scored.sort(key=lambda r: (-r["z"], r["word"]))
    return scored[:TOP_WORDS], scored[::-1][:TOP_WORDS]


def text_report(export_dir, output, *, permutations=200, seed=20261004):
    manifest, texts, observations = load_export(export_dir)
    features = {}
    rows = []
    for observation in observations:
        identity = observation["answer_id"]
        if identity not in features:
            features[identity] = text_features(texts[identity])
        rows.append({**observation, **features[identity]})
    report.log("text_features", answers=len(features), observations=len(rows))
    models = sorted({r["model"] for r in rows})
    tables = {"deciles": report.deciles(rows, FEATURES, ("model",)),
              "associations": (report.associations(rows, FEATURES, ("model",), scope="all answers of the model",
                                                   permutations=permutations, seed=seed)
                               + report.associations(rows, FEATURES, ("model", *DESIGN), scope="one method/engine",
                                                     permutations=permutations, seed=seed)),
              "contrasts": (within_keyword_contrasts(rows, FEATURES, ("model",))
                            + within_keyword_contrasts(rows, FEATURES, ("model", *DESIGN))),
              "words": []}
    word_sections = []
    for model in models:
        high, low = distinctive_words([r for r in rows if r["model"] == model], texts, model=model)
        tables["words"] += [{**r, "direction": "more after higher-axis prompts"} for r in high]
        tables["words"] += [{**r, "direction": "more after lower-axis prompts"} for r in low]
        word_sections += [f"### {model}\n", "| more after higher-axis prompts (z) | more after lower-axis prompts (z) |",
                          "|---|---|"]
        word_sections += [f"| {h['word']} ({h['z']:+.1f}) | {l['word']} ({l['z']:+.1f}) |" for h, l in zip(high, low)]
        word_sections.append("")
    output = new_output(output)
    model_rows = [r for r in tables["associations"] if r["scope"] == "all answers of the model"]
    model_contrasts = {(r["model"], r["outcome"]): r for r in tables["contrasts"] if "method" not in r}
    for row in model_rows:
        contrast = model_contrasts.get((row["model"], row["outcome"]), {})  # absent if no keyword splits
        row["within_keyword_high_minus_low"] = contrast.get("mean_high_minus_low")
        row["share_keywords_higher"] = contrast.get("share_keywords_higher")
    headline = []
    for model in models:
        strongest = sorted((r for r in model_rows if r["model"] == model and r["spearman_rho"] is not None),
                           key=lambda r: -abs(r["spearman_rho"]))[:4]
        headline += [f"**{model}**, `{r['outcome']}`: ρ = {r['spearman_rho']:+.3f}; within keyword, "
                     f"higher-axis prompts differ by {report._cell(r['within_keyword_high_minus_low'])} "
                     f"({report._cell(r['share_keywords_higher'])} of keywords higher)." for r in strongest]
    text = ["# How the answer text changes along the prompt axis\n",
            f"_`analysis/scripts/answer_readiness.py text` at commit `{report._git_commit()}`; export "
            f"`{Path(export_dir).resolve()}` (conditions: {', '.join(manifest['conditions'])}; "
            f"{manifest['observations']:,} answers, {manifest['unique_answers']:,} unique texts)._\n",
            "> " + report.LIMITATION + " Text measures are heuristic counts on the stored answer, which the "
            "dataset caps at about 1,200 characters; rates per 100 words are less affected by the cap than totals. "
            "Within-keyword contrasts compare prompts above vs below their own keyword's median axis position, "
            "so the topic is held fixed.\n",
            "## Headline numbers\n", "\n".join("- " + h for h in headline) + "\n",
            "## 1. Associations with the prompt's axis position (prompt means)\n",
            report.table(model_rows, ["model", "outcome", "n", "spearman_rho", "linear_slope_per_axis_unit",
                                      "blocked_permutation_p_two_sided", "within_keyword_high_minus_low",
                                      "share_keywords_higher"]),
            "Per method/engine rows are in `associations.csv` and `contrasts.csv`.\n",
            "## 2. Text measures by prompt-axis decile\n",
            report.table(tables["deciles"], ["model", "decile", "axis_range", "prompts", *FEATURES]),
            "## 3. Words most distinctive of answers to higher- vs lower-axis prompts (within keyword)\n",
            f"Log-odds with an informative Dirichlet prior (mass {PRIOR_MASS:g}); words seen at least "
            f"{MIN_WORD_COUNT} times. z > 0: more frequent after higher-axis prompts of the same keyword.\n",
            "\n".join(word_sections),
            "## Provenance\n", "```json\n" + json.dumps({"export": manifest, "permutations": permutations,
                                                       "seed": seed}, indent=2) + "\n```\n"]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    for name, rows_ in tables.items():
        report.write_csv(output / f"{name}.csv", rows_)
    return tables


# ---------------------------------------------------------------- analyze (after GPU embedding)

def place_answers(qwen, mistral, battery, axis_map):
    """Answer consensus z and prompt-scale percentile through the archived chain, unchanged."""
    from analysis.interpretability.pipeline import page_readiness_ordering as ordering
    from analysis.scripts.page_readiness_ordering import aligned
    rows = aligned(qwen, mistral, battery)
    ids = sorted(r["candidate_id"] for r in rows)
    by_id = {r["candidate_id"]: r for r in rows}
    z = ordering.consensus([by_id[i]["reference_axis_1_z"] for i in ids],
                           [by_id[i]["candidate_aligned_axis_1_z"] for i in ids])
    with Path(axis_map).open(encoding="utf-8") as stream:
        scale = ordering.PromptScale([json.loads(line) for line in stream if line.strip()])
    percentile, outside = scale.percentile(z)
    return {i: {"consensus_axis_1_z": float(v), "answer_axis_percentile": float(p), "outside": bool(o)}
            for i, v, p, o in zip(ids, z, percentile, outside)}


def analyze(export_dir, qwen, mistral, battery, axis_map, output, *, permutations=200, seed=20261004):
    manifest, texts, observations = load_export(export_dir)
    placed = place_answers(qwen, mistral, battery, axis_map)
    missing = {o["answer_id"] for o in observations} - set(placed)
    if missing:
        raise ValueError(f"{len(missing)} answers lack projections in both views")
    rows = [{**o, "answer_axis_percentile": placed[o["answer_id"]]["answer_axis_percentile"],
             "answer_minus_prompt_axis": placed[o["answer_id"]]["answer_axis_percentile"] - o[AXIS],
             "answer_outside_prompt_range": float(placed[o["answer_id"]]["outside"])} for o in observations]
    tables = {"deciles": report.deciles(rows, PLACEMENT, ("model",)),
              "associations": (report.associations(rows, PLACEMENT, ("model",), scope="all answers of the model",
                                                   permutations=permutations, seed=seed)
                               + report.associations(rows, PLACEMENT, ("model", *DESIGN), scope="one method/engine",
                                                     permutations=permutations, seed=seed)),
              "contrasts": (within_keyword_contrasts(rows, PLACEMENT, ("model",))
                            + within_keyword_contrasts(rows, PLACEMENT, ("model", *DESIGN)))}
    output = new_output(output)
    write_jsonl(output / "answer_coordinates.jsonl.gz",
                ({"answer_id": i, **v} for i, v in sorted(placed.items())))
    text = ["# Where the answers sit on the prompt axis\n",
            f"_`analysis/scripts/answer_readiness.py analyze` at commit `{report._git_commit()}`; "
            f"{len(rows):,} answers ({len(placed):,} unique texts)._\n",
            "> " + report.LIMITATION + " Answer positions use the frozen prompt maps (both LLM2Vec views, the "
            "battery's z-scaling and rotation, the prompt percentile scale), so they are out-of-domain "
            "descriptions of answer text. A slope of 1 for `answer_axis_percentile` would mean answers move "
            "along the axis as much as their prompts.\n",
            "## 1. Answer position vs prompt position (prompt means)\n",
            report.table(tables["associations"], ["scope", "model", *DESIGN, "outcome", "n", "spearman_rho",
                                                  "linear_slope_per_axis_unit",
                                                  "blocked_permutation_p_two_sided"]),
            "## 2. Within-keyword contrasts (higher- minus lower-axis prompts of the same keyword)\n",
            report.table(tables["contrasts"], ["model", *DESIGN, "outcome", "keywords", "mean_high_minus_low",
                                               "share_keywords_higher"]),
            "## 3. By prompt-axis decile\n",
            report.table(tables["deciles"], ["model", "decile", "axis_range", "prompts", *PLACEMENT]),
            "## Provenance\n", "```json\n" + json.dumps({
                "export": manifest, "qwen": str(Path(qwen).resolve()), "mistral": str(Path(mistral).resolve()),
                "battery": str(Path(battery).resolve()), "axis_map": str(Path(axis_map).resolve()),
                "permutations": permutations, "seed": seed}, indent=2) + "\n```\n"]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    for name, rows_ in tables.items():
        report.write_csv(output / f"{name}.csv", rows_)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    p = stages.add_parser("export")
    p.add_argument("--dataset", action="append", required=True, metavar="MODEL=DATASET_ROOT")
    p.add_argument("--axis-map", type=Path, required=True)
    p.add_argument("--condition", action="append", help="default: natural only")
    p.add_argument("--stripes", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    for name in ("text", "analyze"):
        p = stages.add_parser(name)
        p.add_argument("--export", type=Path, required=True)
        p.add_argument("--permutations", type=int, default=200)
        p.add_argument("--seed", type=int, default=20261004)
        p.add_argument("--output", type=Path, required=True)
        if name == "analyze":
            p.add_argument("--qwen", type=Path, required=True, help="merged Qwen-view answer projections")
            p.add_argument("--mistral", type=Path, required=True, help="merged Mistral-view answer projections")
            p.add_argument("--battery", type=Path, required=True)
            p.add_argument("--axis-map", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage == "export":
        datasets = []
        for item in args.dataset:
            model, sep, root = item.partition("=")
            if not sep or model not in report.GENERATOR_MODELS:
                parser.error("each --dataset must be qwen38=ROOT or llama4=ROOT (several roots per model are read in order)")
            datasets.append((model, Path(root)))
        export(datasets, args.axis_map, args.output, conditions=tuple(args.condition or ("natural",)),
               stripes=args.stripes)
    elif args.stage == "text":
        text_report(args.export, args.output, permutations=args.permutations, seed=args.seed)
    else:
        analyze(args.export, args.qwen, args.mistral, args.battery, args.axis_map, args.output,
                permutations=args.permutations, seed=args.seed)
    print("OUTPUT=" + str(Path(args.output).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

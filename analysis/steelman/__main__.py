"""python -m analysis.steelman <part> --output DIR  (parts: chain, generator, fe, lexical, pairs, followup, census, supply, ablation, queries, report).

Each part writes DIR/<part>.json once (refuses to overwrite) with the git commit, the PREREG.md sha256,
the settings and the input manifest. Exit 4: deadline checkpoint (finished fits are cached; rerun).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402
from analysis.steelman import tables as T  # noqa: E402

PREREG = Path(__file__).with_name("PREREG.md")
PARTS = ("chain", "generator", "fe", "lexical", "pairs", "followup", "census", "supply", "ablation", "queries", "report")


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def clean(value):
    return json.loads(json.dumps(value, default=float).replace("NaN", "null").replace("-Infinity", "null").replace("Infinity", "null"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=PARTS)
    p.add_argument("--output", type=Path, default=T.DEFAULT_INPUTS / "steelman-v1")
    p.add_argument("--input-root", type=Path, default=T.DEFAULT_INPUTS)
    p.add_argument("--prompts", type=Path, default=T.DEFAULT_PROMPTS,
                   help="archived prompt snapshot (Mac) or the registration's population-prompts.jsonl (HoreKa)")
    p.add_argument("--assembled", type=Path, help="funnel_study assemble output (default: Mac exploration tables)")
    p.add_argument("--extract", type=Path, help="funnel_study extract output")
    p.add_argument("--replay", type=Path, help="funnel_study replay output")
    p.add_argument("--trace-extract", type=Path, help="intent_stages_study trace-extract output (the agent's queries on HoreKa)")
    p.add_argument("--features", type=Path, help="funnel feature table (rows.parquet), default <input-root>/funnel-features-v1")
    p.add_argument("--review", type=Path, help="published answer rows (funnel_review.py); skipped when absent")
    p.add_argument("--split", choices=["exploration", "confirmation", "all"], default="exploration")
    p.add_argument("--selection-draws", type=int, default=30, help="lexical: keyword draws of the shortlisting-model refits (0 skips them)")
    p.add_argument("--lexicon", type=Path, help="lexical: frozen action-word list (default analysis/steelman/lexicon.json)")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--model-draws", type=int, default=100, help="keyword draws for refitted generator models")
    p.add_argument("--mc", type=int, default=200, help="Monte-Carlo draws per answer for expected cited intent")
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--shuffle-seed", type=int, default=20261008)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stop-after-minutes", type=float, default=1e9)
    p.add_argument("--stratum", help="generator / fe: fill the cache for this stratum only (\"model · Method\"); "
                                      "no part file is written; a run without --stratum assembles from the cache")
    a = p.parse_args(argv)
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{a.part}.json"
    if a.stratum and a.part not in ("generator", "fe"):
        raise SystemExit("--stratum applies to generator and fe only")
    if a.part == "report":
        from analysis.steelman import report
        return report.build(out_dir)
    if target.exists():
        if a.stratum:
            print(json.dumps({"part": a.part, "stratum": a.stratum, "status": "assembled already"}), flush=True)
            return 0
        raise SystemExit(f"refusing to overwrite {target}")
    started = time.time()
    t = T.load(a.input_root, a.prompts, assembled=a.assembled, extract=a.extract, replay=a.replay,
               trace_extract=a.trace_extract, split=a.split)
    header = {"part": a.part, "git_commit": readiness.git_commit(), "prereg_sha256": prereg_sha(),
              "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "inputs": t.manifest, "exploratory": a.split != "confirmation", "observational": True, "scientific_result": True}
    deadline = time.monotonic() + 60 * a.stop_after_minutes
    cache_key = {"git_commit": header["git_commit"], "prereg_sha256": header["prereg_sha256"], "split": a.split,
                 "settings": [a.bootstrap, a.permutations, a.model_draws, a.mc, a.seed, a.shuffle_seed, a.selection_draws],
                 "inputs": {k: t.manifest.get(k) for k in ("assembled", "paths")}}
    fingerprint = hashlib.sha256(json.dumps(cache_key, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def cache(name):
        root = out_dir / f"{name}.cache" / fingerprint
        root.mkdir(parents=True, exist_ok=True)
        (root / "key.json").write_text(json.dumps(cache_key, indent=1, sort_keys=True, default=str))
        return root
    if a.part == "chain":
        from analysis.steelman import chain
        body = chain.run(t, bootstrap=a.bootstrap, permutations=a.permutations, seed=a.seed, shuffle_seed=a.shuffle_seed,
                         items_path=Path(t.manifest["paths"]["extract"]) / "items.npz",
                         review_parquet=a.review or a.input_root / "funnel-review-v1/answers.parquet")
    elif a.part == "generator":
        from analysis.steelman import generator
        body = generator.run(t, a, cache_root=cache("generator"), deadline=deadline)
        if body is None:
            return 4
    elif a.part in ("census", "supply", "ablation", "queries"):
        from analysis.steelman import fullparts
        body = fullparts.run(t, a, a.part)
    elif a.part == "followup":
        from analysis.steelman import chain
        body = chain.followup(t, bootstrap=a.bootstrap, seed=a.seed)
    elif a.part == "fe":
        from analysis.steelman import generator
        body = generator.fe_run(t, a, cache_root=cache("fe"), deadline=deadline)
        if body is None:
            return 4
    elif a.part == "lexical":
        from analysis.steelman import lexical
        body = lexical.run(t, a, cache_root=cache("lexical"), deadline=deadline)
        if body is None:
            return 4
    else:
        from analysis.steelman import pairs
        body = pairs.run(t, a)
    if a.stratum:
        print(json.dumps({"part": a.part, "stratum": a.stratum, "status": "stratum cached", "seconds": round(time.time() - started, 1)}),
              flush=True)
        return 0
    header["seconds"] = round(time.time() - started, 1)
    readiness.write_json(target, clean({**header, **body}))
    print(json.dumps({"wrote": str(target), "seconds": header["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

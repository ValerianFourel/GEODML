"""First look at the full run's results: completeness, provenance and every pre-registered verdict, per split.

    python3 -m analysis.steelman.check_fullrun RESULTS_DIR [--status final-status.json] [--out verdicts.json]

RESULTS_DIR is the unpacked results tarball of the full run (it holds steelman-exploration/, steelman-confirmation/ and
paper-results/). The confirmation split carries the confirmatory verdicts (PREREG addendum B3). Exit 0 when every part
is present, 1 otherwise. Computes nothing new: verdicts come from analysis/steelman/decide.py on the stored part files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.steelman import decide  # noqa: E402

PARTS = ("chain", "followup", "pairs", "census", "supply", "ablation", "queries", "lexical", "generator", "fe")
SPLITS = ("exploration", "confirmation")
STRATA = ("llama4 · Parallel", "llama4 · Reactive", "qwen38 · Parallel", "qwen38 · Reactive")


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def split_summary(folder: Path) -> dict:
    parts = {p: _load(folder / f"{p}.json") for p in PARTS}
    out = {"missing_parts": [p for p, v in parts.items() if v is None],
           "commits": sorted({v["git_commit"] for v in parts.values() if v}),
           "prereg_sha256": sorted({v["prereg_sha256"] for v in parts.values() if v}),
           "backends": {p: v.get("backend") for p, v in parts.items() if v},
           "settings": {p: {k: v["settings"].get(k) for k in ("bootstrap", "permutations", "model_draws", "mc", "seed",
                                                                "shuffle_seed", "selection_draws")} for p, v in parts.items() if v}}
    ch, gen, sup = parts["chain"], parts["generator"], parts["supply"]
    if ch:
        methods = [m for m in STRATA if m in ch["strata"]]
        out["strata_present"] = methods
        out["C1"] = {m: decide.verdict_c1(ch["strata"][m], {e: c for e, c in ch["engine_strata"].items() if e.startswith(m)})
                     for m in methods}
        out["beta_K"] = {m: ch["strata"][m]["chain"]["common"]["slopes"].get("K") for m in methods}
        out["answers"] = {m: ch["strata"][m]["chain"]["common"]["answers"] for m in methods}
    if gen:
        models = sorted({m.split(" · ")[0] for m in gen["strata"]})
        out["C2"] = {model: decide.verdict_c2({m: gen["strata"][m]["delta"]["main"] for m in gen["strata"] if m.startswith(model + " ·")})
                     for model in models}
        out["generator_keep_modelled"] = {m: e.get("keep_modelled") for m, e in gen["strata"].items()}
        out["generator_failed_replicates"] = {m: e["delta"].get("failed_replicates") for m, e in gen["strata"].items()}
    if sup:
        usable = {k: v for k, v in sup["strata"].items() if "skipped" not in v}
        out["supply"] = decide.verdict_supply(usable) if usable else {"verdict": "no strata"}
    if parts["census"]:
        out["census"] = {k: v for k, v in parts["census"].items() if k not in ("git_commit", "prereg_sha256", "settings", "inputs")}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path)
    p.add_argument("--status", type=Path, help="final-status.json from `fullrun status` (checks done == 300)")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    report, ok = {}, True
    if a.status and a.status.exists():
        s = json.loads(a.status.read_text())
        report["ledger"] = {"tasks": s["tasks"], "states": s["states"], "failed": s["failed_tasks"]}
        ok &= s["states"].get("done", 0) == s["tasks"]
    for split in SPLITS:
        folder = next((f for f in (a.results / f"steelman-{split}", a.results / "paper-results" / f"steelman-{split}") if f.is_dir()), None)
        if folder is None:
            report[split] = {"missing": True}
            ok = False
            continue
        report[split] = split_summary(folder)
        ok &= not report[split]["missing_parts"] and len(report[split]["commits"]) == 1
    bundle = a.results / "paper-results" / "manifest.json"
    report["bundle_manifest"] = bundle.exists()
    ok &= bundle.exists()
    text = json.dumps(report, indent=1, default=str, ensure_ascii=False)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    for split in SPLITS:
        r = report.get(split, {})
        label = "CONFIRMATORY" if split == "confirmation" else "exploratory"
        print(f"\n== {split} ({label}) ==")
        for m, v in (r.get("C1") or {}).items():
            print(f"  C1 {m}: {v['verdict']}")
        for m, v in (r.get("C2") or {}).items():
            print(f"  C2 {m}: {v['verdict']}")
        if r.get("supply"):
            print(f"  supply: {r['supply']['verdict']}")
        if r.get("missing_parts"):
            print(f"  MISSING PARTS: {r['missing_parts']}")
    print("\ncomplete" if ok else "\nINCOMPLETE: see missing parts, commits or ledger above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

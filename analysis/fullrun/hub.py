"""Reconcile the HoreKa dataset roots with the Hub and import what only the Hub has (login/transfer node only).

``inventory``: list every bundle descriptor on the Hub (`exchange/bundles/`), cache them locally, and record the
completed outcomes per model. ``missing``: compare with the completed cells of the local dataset roots (task ledger
snapshots) and list the bundles that hold at least one cell no local root has. ``fetch``: import those bundles, one at
a time, into a separate import root with ``Exchange.download`` (hash-checked objects, outcomes imported into that
root's ledger), optionally only shard k/n of the list. Finite and resumable: a bundle already imported is verified
from disk and not downloaded again. Compute nodes never contact the Hub.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

REPO = "ValerianFourel/geodml-experiment-v2-paper-private"
MODEL_OF = {"meta-llama/Llama-4-Scout-17B-16E-Instruct": "llama4", "Qwen/Qwen3.8-27B": "qwen38"}


def model_label(model_id: str) -> str:
    for key, label in MODEL_OF.items():
        if model_id == key or model_id.endswith(key.split("/")[-1]):
            return label
    low = model_id.lower()
    return "llama4" if "llama" in low else "qwen38" if "qwen" in low else model_id


def outcomes_by_model(descriptor: dict) -> dict:
    """{model label: set of fingerprints} of a bundle's completed outcomes (each outcome carries its own model)."""
    out: dict = {}
    for fp, event in descriptor.get("outcomes", {}).items():
        if event.get("state") != "completed":
            continue
        model = model_label(str((event.get("identity") or {}).get("model_id", "")))
        out.setdefault(model, set()).add(fp)
    return out


def inventory(cache: Path, repo: str = REPO, revision: str | None = None) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    revision = revision or api.dataset_info(repo).sha
    cache = Path(cache)
    (cache / "bundles").mkdir(parents=True, exist_ok=True)
    names = sorted(f.path for f in api.list_repo_tree(repo, path_in_repo="exchange/bundles", repo_type="dataset", revision=revision)
                   if f.path.endswith(".json"))
    per_bundle, totals = {}, Counter()
    for name in names:
        local = cache / "bundles" / Path(name).name
        if not local.exists():
            path = hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=cache / "download")
            local.write_bytes(Path(path).read_bytes())
            Path(path).unlink()
        value = json.loads(local.read_text())
        models = outcomes_by_model(value)
        per_bundle[Path(name).stem] = {m: sorted(f) for m, f in models.items()}
        for m, f in models.items():
            totals[m] += len(f)
    unique = {}
    for models in per_bundle.values():
        for m, fps in models.items():
            unique.setdefault(m, set()).update(fps)
    result = {"repo": repo, "revision": revision, "bundles": len(names),
              "completed_outcomes_listed": dict(totals), "completed_unique": {m: len(v) for m, v in unique.items()}}
    (cache / "inventory.json").write_text(json.dumps({**result, "per_bundle": per_bundle}, indent=1))
    return result


def local_completed(roots: list[Path], stripes: int = 256) -> set:
    from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger
    done = set()
    for root in roots:
        latest = StripedTaskLedger(Path(root) / "control/task-ledger", stripe_count=stripes).snapshot()["latest"]
        done |= {fp for fp, e in latest.items() if e.get("state") == "completed"}
    return done


def missing(inventory_path: Path, local: set) -> dict:
    """Bundles with at least one completed cell absent locally, the cells per model, ordered by bundle id."""
    inv = json.loads(Path(inventory_path).read_text())
    bundles, cells = [], Counter()
    for bundle, models in sorted(inv["per_bundle"].items()):
        new = {m: [f for f in fps if f not in local] for m, fps in models.items()}
        n = sum(len(v) for v in new.values())
        if n:
            bundles.append({"bundle": bundle, "new_cells": {m: len(v) for m, v in new.items() if v}})
            for m, v in new.items():
                cells[m] += len(v)
    unique_missing = {}
    for models in inv["per_bundle"].values():
        for m, fps in models.items():
            unique_missing.setdefault(m, set()).update(f for f in fps if f not in local)
    return {"bundles": bundles, "bundle_count": len(bundles), "missing_unique_cells": {m: len(v) for m, v in unique_missing.items()}}


def fetch(plan: dict, import_root: Path, shard: str | None = None, repo: str = REPO, revision: str | None = None) -> dict:
    from analysis.interpretability.pipeline.agentic_hour_sync import Exchange, HubStore
    import_root = Path(import_root)
    import_root.mkdir(parents=True, exist_ok=True)
    exchange = Exchange(HubStore(repo), import_root.parent / f"{import_root.name}.journal")
    bundles = [b["bundle"] for b in plan["bundles"]]
    if shard:
        k, n = (int(v) for v in shard.split("/"))
        bundles = bundles[k - 1::n]
    done, errors = [], {}
    for bundle in bundles:
        try:
            exchange.download(bundle, import_root, revision=revision)
            done.append(bundle)
            print(json.dumps({"imported": bundle, "done": len(done), "of": len(bundles)}), flush=True)
        except Exception as error:  # report every failing bundle; never retry in a loop
            errors[bundle] = str(error)[:300]
            print(json.dumps({"failed": bundle, "error": errors[bundle]}), flush=True)
    return {"imported": len(done), "failed": errors, "import_root": str(import_root)}

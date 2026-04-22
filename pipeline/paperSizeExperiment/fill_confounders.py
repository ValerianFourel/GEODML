"""Fill the 6 SERP/text-based confounder columns in-place.

Columns:
  - conf_serp_position
  - conf_snippet_len
  - conf_title_has_kw
  - conf_title_kw_sim     (sentence-transformers all-MiniLM-L6-v2 cosine)
  - conf_snippet_kw_sim   (same)
  - conf_bm25             (rank_bm25.BM25Okapi on page body text)

Only NaN cells are filled; existing values are preserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent / "consolidated_results"
RUN_DIRS = sorted((ROOT / "runs").iterdir())
MINILM_NAME = "all-MiniLM-L6-v2"
BODY_TOKEN_CAP = 5000  # same cap as pipeline/extract_features.py:731


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def extract_body_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.body or soup
    return body.get_text(separator=" ", strip=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load_serp(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Return (keyword, url) -> {position, title, snippet}."""
    p = run_dir / "phase2" / "keywords.jsonl"
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not p.exists():
        return lookup
    with p.open() as fh:
        for line in fh:
            obj = json.loads(line)
            kw = obj.get("query") or obj.get("keyword")
            serp = obj.get("serp") or {}
            for r in serp.get("raw_results", []) or []:
                u = r.get("url")
                if kw and u:
                    lookup[(kw, u)] = {
                        "position": r.get("position"),
                        "title": r.get("title") or "",
                        "snippet": r.get("snippet") or "",
                    }
    return lookup


def load_body_text(run_dir: Path, url: str, cache: dict[str, str]) -> str:
    if url in cache:
        return cache[url]
    hpath = run_dir / "phase2" / "html_cache" / f"{url_hash(url)}.html"
    text = ""
    if hpath.exists():
        try:
            html = hpath.read_text(encoding="utf-8", errors="ignore")
            text = extract_body_text(html)
        except Exception:
            text = ""
    cache[url] = text
    return text


def compute_per_run(run_dir: Path, st_model: SentenceTransformer) -> pd.DataFrame:
    """Return a DataFrame keyed on (keyword, url) with the 6 filled values."""
    run_id = run_dir.name
    serp = load_serp(run_dir)
    if not serp:
        print(f"  [{run_id}] no SERP data, skipping")
        return pd.DataFrame()

    rows = []
    body_cache: dict[str, str] = {}
    by_kw: dict[str, list[tuple[str, str]]] = {}  # kw -> list of (url, body_text)

    for (kw, url), meta in serp.items():
        title = meta["title"]
        snippet = meta["snippet"]
        body = load_body_text(run_dir, url, body_cache)
        by_kw.setdefault(kw, []).append((url, body))
        rows.append({
            "run_id": run_id,
            "keyword": kw,
            "url": url,
            "conf_serp_position": meta["position"],
            "conf_snippet_len": len(snippet) if snippet else 0,
            "conf_title_has_kw": int(any(t in title.lower() for t in kw.lower().split() if t)),
            "_title": title,
            "_snippet": snippet,
        })
    df = pd.DataFrame(rows)

    # Embeddings (batch across the run)
    print(f"  [{run_id}] encoding {len(df):,} rows with MiniLM…", flush=True)
    kw_unique = sorted(df["keyword"].unique())
    kw_emb = {k: v for k, v in zip(
        kw_unique,
        st_model.encode(kw_unique, batch_size=256, show_progress_bar=False, convert_to_numpy=True),
    )}
    title_emb = st_model.encode(df["_title"].fillna("").tolist(),
                                batch_size=256, show_progress_bar=False, convert_to_numpy=True)
    snippet_emb = st_model.encode(df["_snippet"].fillna("").tolist(),
                                  batch_size=256, show_progress_bar=False, convert_to_numpy=True)

    title_sims, snip_sims = [], []
    titles_list = df["_title"].tolist()
    snippets_list = df["_snippet"].tolist()
    keywords_list = df["keyword"].tolist()
    for i in range(len(df)):
        ke = kw_emb[keywords_list[i]]
        title_sims.append(round(cosine(ke, title_emb[i]), 4) if titles_list[i] else 0.0)
        snip_sims.append(round(cosine(ke, snippet_emb[i]), 4) if snippets_list[i] else 0.0)
    df["conf_title_kw_sim"] = title_sims
    df["conf_snippet_kw_sim"] = snip_sims

    # BM25 per keyword group
    print(f"  [{run_id}] BM25 over {len(by_kw):,} keyword groups…", flush=True)
    bm25_map: dict[tuple[str, str], float] = {}
    for kw, items in by_kw.items():
        texts = [t for _, t in items]
        tokenized = [
            (t.lower().split()[:BODY_TOKEN_CAP] if t and t.strip() else [""])
            for t in texts
        ]
        try:
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(kw.lower().split())
        except Exception:
            scores = [0.0] * len(items)
        for (url, _), s in zip(items, scores):
            bm25_map[(kw, url)] = round(float(s), 4)
    df["conf_bm25"] = df.apply(lambda r: bm25_map.get((r["keyword"], r["url"])), axis=1)

    return df.drop(columns=["_title", "_snippet"])


def fill_csv(path: Path, filled: pd.DataFrame, key_cols: list[str]) -> None:
    if not path.exists():
        print(f"  skip missing: {path}"); return
    df = pd.read_csv(path, low_memory=False)
    before_nan = {c: int(df[c].isna().sum()) for c in TARGET_COLS if c in df.columns}

    m = df.merge(filled, on=key_cols, how="left", suffixes=("", "__fill"))
    for c in TARGET_COLS:
        fill_col = f"{c}__fill"
        if c in m.columns and fill_col in m.columns:
            m[c] = m[c].where(m[c].notna(), m[fill_col])
        elif fill_col in m.columns:
            m[c] = m[fill_col]
    m = m[[c for c in m.columns if not c.endswith("__fill")]]

    after_nan = {c: int(m[c].isna().sum()) for c in TARGET_COLS if c in m.columns}
    m.to_csv(path, index=False)
    delta = {c: before_nan[c] - after_nan[c] for c in before_nan}
    print(f"  {path.relative_to(ROOT.parent)}  rows={len(m):,}  filled={delta}")


TARGET_COLS = [
    "conf_serp_position",
    "conf_snippet_len",
    "conf_title_has_kw",
    "conf_title_kw_sim",
    "conf_snippet_kw_sim",
    "conf_bm25",
]


def main() -> None:
    print(f"Loading {MINILM_NAME}…", flush=True)
    st_model = SentenceTransformer(MINILM_NAME)

    # Compute per-run tables, keyed on (run_id, keyword, url)
    all_tables: list[pd.DataFrame] = []
    for run_dir in RUN_DIRS:
        if not run_dir.is_dir():
            continue
        print(f"\n→ {run_dir.name}")
        t = compute_per_run(run_dir, st_model)
        if not t.empty:
            all_tables.append(t)

    combined = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    print(f"\nBuilt fill table: {len(combined):,} (run_id, keyword, url) rows")

    # Fill per-run CSVs (key on keyword + url)
    print("\n── filling per-run CSVs ──")
    for run_dir in RUN_DIRS:
        csv = run_dir / "geodml_dataset.csv"
        run_slice = combined[combined["run_id"] == run_dir.name].drop(columns=["run_id"])
        fill_csv(csv, run_slice, key_cols=["keyword", "url"])

    # Fill merged / regression CSVs (key on run_id + keyword + url)
    print("\n── filling merged / regression CSVs ──")
    for fname in ["regression_dataset.csv",
                  "merged/merged_all_runs.csv",
                  "merged/merged_all_8runs.csv"]:
        fill_csv(ROOT / fname, combined, key_cols=["run_id", "keyword", "url"])


if __name__ == "__main__":
    main()

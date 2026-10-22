"""Precomputed cross-encoder scores in front of the live scorer.

The reranker pairs of Parallel Expansion are fixed by the logged queries, so their scores can be computed before
generation: logged scores (the original CPU run), the rebuttal's GPU scores and new GPU scores (score_pairs.py). A
pair is keyed by sha256 of (query, url, title, text), the same four fields MemoizingSnippetScorer keys on; a score
for a different text is never reused. Misses go to the live scorer (or fail when there is none).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from threading import Lock

from analysis.interpretability.pipeline.agentic_search import Snippet

KEY_BYTES = 16


def pair_key(query: str, url: str, title: str, text: str) -> bytes:
    return hashlib.sha256(json.dumps([query, url, title, text], ensure_ascii=False).encode()).digest()[:KEY_BYTES]


def load_cache(paths: Iterable[Path]) -> dict[bytes, tuple[float, str]]:
    """{key: (score, source)} from score-cache parquet files (columns key, score, source); first file wins."""
    import pyarrow.parquet as pq

    table: dict[bytes, tuple[float, str]] = {}
    for path in paths:
        t = pq.read_table(path, columns=["key", "score", "source"])
        for key, score, source in zip(t["key"].to_pylist(), t["score"].to_pylist(), t["source"].to_pylist()):
            table.setdefault(bytes(key), (float(score), str(source)))
    return table


class CachedScorer:
    """SnippetScorer that answers from a precomputed table and sends only misses to ``base``."""

    deterministic = True

    def __init__(self, base, table: dict[bytes, tuple[float, str]], *, model_id: str | None = None,
                 model_revision: str | None = None) -> None:
        if base is not None and getattr(base, "deterministic", False) is not True:
            raise ValueError("the live scorer must declare deterministic=True")
        if base is None and model_id is None:
            raise ValueError("a cache without a live scorer needs the model id it stands for")
        self.base = base
        self.table = table
        self.model_id = base.model_id if base is not None else model_id
        self.model_revision = base.model_revision if base is not None else model_revision
        self.counts: Counter[str] = Counter()
        self._lock = Lock()

    def score(self, query: str, snippets: Sequence[Snippet]) -> list[float]:
        keys = [pair_key(query, row.url, row.title, row.text) for row in snippets]
        hits = [self.table.get(k) for k in keys]
        missing = [i for i, h in enumerate(hits) if h is None]
        values = [h[0] if h is not None else 0.0 for h in hits]
        if missing:
            if self.base is None:
                raise KeyError(f"{len(missing)} pair(s) not in the score cache and no live scorer")
            fresh = list(self.base.score(query, [snippets[i] for i in missing]))
            if len(fresh) != len(missing):
                raise ValueError("snippet scorer returned the wrong number of scores")
            for i, v in zip(missing, fresh):
                values[i] = float(v)
        with self._lock:
            self.counts.update(h[1] for h in hits if h is not None)
            self.counts["live"] += len(missing)
        return values

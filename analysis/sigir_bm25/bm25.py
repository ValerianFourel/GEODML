"""Lucene-style BM25 over one frozen search snapshot, as a drop-in for the agentic pipeline's search.

Port of the ARR rebuttal's ``bm25l`` (paper_v2-rebuttal/www/analysis/t1_build.py): Lucene's BM25Similarity with
Pyserini's defaults (k1 = 0.9, b = 0.4, idf = ln(1 + (N - n + 0.5) / (n + 0.5)), document lengths through Lucene's
one-byte SmallFloat norms, N and avgdl over documents with at least one token), documents ``title + "\\n" + snippet``,
DefaultEnglishAnalyzer approximated in Python (word tokens, possessive 's removed, case-folded, Lucene's 33 English
stopwords, Porter stemmer with Martin's extensions), and the query as a bag of words (each distinct term once,
boosted by its count). Every search returns ``limit`` rows ordered by (-score, position, sha256(query NUL url),
snapshot index), the frozen search's own tie-break, so rows without a matching term fill the list in position order.

``Bm25SnapshotSearchAdapter`` reads and normalizes the snapshot exactly like ``FrozenSnapshotSearchAdapter`` and keeps
its response payload (rows with keyword and position, snapshot path and sha256), so downstream code maps rows the
same way; only ``selection`` changes, and the payload adds the scores, the index fingerprint and the analyzer.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from analysis.interpretability.pipeline.agentic_search import SearchResponse, Snippet
from analysis.scripts.run_agentic_search_integration_smoke import FrozenSnapshotSearchAdapter
from analysis.sigir_bm25.porter import PorterStemmer

K1, B = 0.9, 0.4
SELECTION = "bm25-lucene-k1-0.9-b-0.4-v1"
TOKEN = re.compile(r"\w+(?:['’.]\w+)*", re.UNICODE)
STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                 "there these they this to was will with".split())  # EnglishAnalyzer.ENGLISH_STOP_WORDS_SET
ANALYZER = {"tokens": TOKEN.pattern, "possessive": "'s and ’s removed", "case": "casefold",
            "stopwords": "lucene-english-33", "stemmer": "porter-martin-extensions (nltk 3.9.2, vendored)",
            "unidata_version": unicodedata.unidata_version}
_PORTER = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)


def lucene_tokens(text: str) -> list[str]:
    """DefaultEnglishAnalyzer, approximated: word tokens, possessive filter, lower case, stopwords, Porter stemmer."""
    out = []
    for m in TOKEN.findall(text):
        t = m[:-2] if m[-2:].casefold() in ("'s", "’s") else m
        t = t.casefold()
        if t and t not in STOP:
            out.append(_PORTER.stem(t, to_lowercase=False))
    return out


def _long_to_int4(i: int) -> int:
    bits = i.bit_length()
    if bits < 4:
        return i
    shift = bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i: int) -> int:
    bits, shift = i & 0x07, (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_FREE = 255 - _long_to_int4(2 ** 31 - 1)  # SmallFloat.NUM_FREE_VALUES = 24


def lucene_norm_length(n: int) -> int:
    """SmallFloat.byte4ToInt(SmallFloat.intToByte4(n)): the document length Lucene's BM25 sees."""
    b = n if n < _FREE else _FREE + _long_to_int4(n - _FREE)
    return b if b < _FREE else _FREE + _int4_to_long(b - _FREE)


def _digest(query: str, url: str) -> str:
    return hashlib.sha256(f"{query}\0{url}".encode()).hexdigest()


class Bm25Index:
    """Inverted index of per-document term weights, so that a query's score is sum(count * weight) per document."""

    def __init__(self, documents: Sequence[str], positions: Sequence[int], urls: Sequence[str]):
        if not (len(documents) == len(positions) == len(urls)) or not documents:
            raise ValueError("documents, positions and urls must be nonempty and aligned")
        self.position = np.asarray(positions, np.int64)
        self.url = list(urls)
        counts = [Counter(lucene_tokens(d)) for d in documents]
        lengths = np.asarray([sum(c.values()) for c in counts], np.int64)
        self.n_docs = int((lengths > 0).sum())
        if self.n_docs == 0:
            raise ValueError("no document has an indexable token")
        self.avgdl = float(lengths.sum()) / self.n_docs
        norm = np.asarray([lucene_norm_length(int(n)) for n in lengths], np.float64)
        postings: dict[str, tuple[list[int], list[float]]] = {}
        for i, c in enumerate(counts):
            for term, tf in c.items():
                postings.setdefault(term, ([], []))[0].append(i)
                postings[term][1].append(float(tf))
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, (docs, tfs) in postings.items():
            d, f = np.asarray(docs, np.int64), np.asarray(tfs, np.float64)
            idf = math.log(1 + (self.n_docs - len(d) + 0.5) / (len(d) + 0.5))
            self.postings[term] = (d, idf * f / (f + K1 * (1 - B + B * norm[d] / self.avgdl)))
        h = hashlib.sha256(f"{SELECTION}|{self.n_docs}|{self.avgdl!r}|{len(self.url)}".encode())
        for term in sorted(self.postings):
            d, w = self.postings[term]
            h.update(term.encode() + b"\0" + d.tobytes() + w.tobytes())
        self.fingerprint = h.hexdigest()

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> Bm25Index:
        return cls([f"{r['title']}\n{r['snippet']}" for r in rows], [int(r["position"]) for r in rows],
                   [str(r["url"]) for r in rows])

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.url), np.float64)
        for term, count in sorted(Counter(lucene_tokens(query)).items()):
            hit = self.postings.get(term)
            if hit is not None:
                s[hit[0]] += count * hit[1]
        return s

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """The k best documents as (index, score), ordered by (-score, position, sha256(query NUL url), index)."""
        if k < 1:
            raise ValueError("search limit must be positive")
        score = self.scores(query)
        n = len(score)
        if n > k:
            threshold = np.partition(score, n - k)[n - k]
            above = np.flatnonzero(score > threshold)
            tie = np.flatnonzero(score == threshold)
            need = k - len(above)
            if len(tie) > need:
                pos = self.position[tie]
                cut = np.partition(pos, need - 1)[need - 1]
                sure, border = tie[pos < cut], tie[pos == cut]
                rest = need - len(sure)
                if len(border) > rest:
                    border = np.asarray(sorted(border, key=lambda i: (_digest(query, self.url[i]), i))[:rest],
                                        np.int64)
                tie = np.concatenate([sure, border])
            candidates = np.concatenate([above, tie])
        else:
            candidates = np.arange(n)
        order = sorted((int(i) for i in candidates),
                       key=lambda i: (-score[i], self.position[i], _digest(query, self.url[i]), i))
        return [(i, float(score[i])) for i in order]


class Bm25SnapshotSearchAdapter(FrozenSnapshotSearchAdapter):
    """BM25 over the frozen search's own normalized snapshot rows; payload as the frozen search's."""

    def __init__(self, engine: str, path: Path, *, expected_fingerprint: str | None = None) -> None:
        super().__init__(engine, path)
        self.index = Bm25Index.from_rows(self.rows)
        if expected_fingerprint is not None and self.index.fingerprint != expected_fingerprint:
            raise ValueError(f"{engine}: BM25 index fingerprint {self.index.fingerprint} differs from "
                             f"the expected {expected_fingerprint}")

    def _select_rows(self, query: str, limit: int) -> list[dict[str, Any]]:
        return [self.rows[i] for i, _ in self.index.top(query, limit)]

    async def search(self, query: str, limit: int) -> SearchResponse:
        ranked = self.index.top(query, limit)
        selected = [self.rows[i] for i, _ in ranked]
        snippets = tuple(Snippet.from_mapping({"url": str(r["url"]), "title": str(r["title"]),
                                               "text": str(r["snippet"])}) for r in selected)
        return SearchResponse(
            engine=self.engine,
            query=query,
            snippets=snippets,
            raw_payload={
                "format_version": "frozen-search-snapshot-response-v1",
                "snapshot": str(self.path),
                "snapshot_sha256": self.snapshot_sha256,
                "snapshot_rows": self.snapshot_rows,
                "query": query,
                "selection": SELECTION,
                "rows": selected,
                "scores": [s for _, s in ranked],
                "index_fingerprint": self.index.fingerprint,
                "analyzer": ANALYZER,
            },
        )

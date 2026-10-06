"""Row bookkeeping for the funnel study (``analysis/docs/funnel_study.md``).

A *row* is one usable record of a frozen search snapshot: (engine, keyword, stored position, url,
title, snippet). Rows are read with the search adapter's own reader and usability rule, in the
adapter's order, so the row id of a record equals ``engine offset + adapter index`` with engines
in sorted order. Within one (engine, keyword) no two rows share a stored position, so
(engine, keyword, position) identifies a row exactly.

``LexicalIndex`` reproduces ``FrozenSnapshotSearchAdapter._select_rows`` with an inverted index
(same tokens, same sort key) so prompt and keyword replays of the whole population run in seconds.
``answer_items`` maps one generator trace to the exact rows each answer retrieved, scored,
presented and ranked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.interpretability.pipeline import page_readiness_ordering as ordering

PARALLEL, REACTIVE = "Parallel-Expansion-v1", "Reactive-Snippet-Loop-v1"
SEARCH_RESULT_LIMIT = 20
ROW_FIELDS = ("engine", "keyword", "position", "url", "title", "snippet")


@dataclass
class SnapshotRows:
    """All usable rows of the engine snapshots, in adapter order, engines sorted."""

    engine: list
    keyword: list
    position: np.ndarray
    url: list
    title: list
    snippet: list
    doc_id: list            # sha256(title.strip() + "\n" + snippet.strip()): the corpus page id
    engine_offset: dict     # engine -> first row id
    snapshot_sha256: dict   # engine -> sha256 of the snapshot file
    exclusions: dict        # engine -> adapter exclusion audit

    def __len__(self) -> int:
        return len(self.url)

    def lookup(self) -> dict:
        """(engine, keyword, position) -> row id; raises if the identity is not unique."""
        out = {}
        for i, key in enumerate(zip(self.engine, self.keyword, self.position.tolist())):
            if key in out:
                raise ValueError(f"snapshot row identity is not unique: {key}")
            out[key] = i
        return out


def snapshot_rows(paths: dict) -> SnapshotRows:
    """Read every engine snapshot with the adapter's reader and usability rule."""
    from analysis.scripts.run_agentic_search_integration_smoke import (
        _normalize_usable_row, _read_snapshot, _sha256_file)

    columns = {f: [] for f in ROW_FIELDS}
    offsets, digests, exclusions = {}, {}, {}
    for engine in sorted(paths):
        path = Path(paths[engine]).resolve(strict=True)
        offsets[engine] = len(columns["url"])
        digests[engine] = _sha256_file(path)
        excluded = defaultdict(int)
        for raw in _read_snapshot(path):
            row, reason = _normalize_usable_row(raw)
            if row is None:
                excluded[reason] += 1
                continue
            columns["engine"].append(engine)
            for field in ROW_FIELDS[1:]:
                columns[field].append(row[field])
        exclusions[engine] = dict(sorted(excluded.items()))
    doc_ids = [ordering.page_id(ordering.page_text(t, s)) for t, s in zip(columns["title"], columns["snippet"])]
    return SnapshotRows(engine=columns["engine"], keyword=columns["keyword"],
                        position=np.asarray(columns["position"], np.int64), url=columns["url"], title=columns["title"],
                        snippet=columns["snippet"], doc_id=doc_ids, engine_offset=offsets, snapshot_sha256=digests,
                        exclusions=exclusions)


def row_table_digest(rows: SnapshotRows) -> str:
    """Identity of the whole row table (order included)."""
    digest = hashlib.sha256()
    for record in zip(rows.engine, rows.keyword, rows.position.tolist(), rows.url, rows.title, rows.snippet):
        digest.update(json.dumps(record, ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


# ---------------------------------------------------------------- the frozen search, vectorised

class LexicalIndex:
    """``_select_rows`` of one engine: sort by (-exact, -overlap, position, sha256(query\\0url), index)
    with overlap = 4 * |query ∩ keyword tokens| + |query ∩ (title snippet) tokens|. Returns row ids."""

    def __init__(self, rows: SnapshotRows, engine: str):
        from analysis.scripts.run_agentic_search_integration_smoke import _tokens

        self._tokens = _tokens
        ids = [i for i, e in enumerate(rows.engine) if e == engine]
        self.ids = np.asarray(ids, np.int64)
        self.keyword = [rows.keyword[i] for i in ids]
        self.by_keyword = defaultdict(list)
        for local, k in enumerate(self.keyword):
            self.by_keyword[k.casefold()].append(local)
        self.url = [rows.url[i] for i in ids]
        self.position = rows.position[self.ids]
        self.keyword_postings, self.evidence_postings = defaultdict(list), defaultdict(list)
        for local, i in enumerate(ids):
            for token in _tokens(rows.keyword[i]):
                self.keyword_postings[token].append(local)
            for token in _tokens(f"{rows.title[i]} {rows.snippet[i]}"):
                self.evidence_postings[token].append(local)
        self.keyword_postings = {k: np.asarray(v, np.int64) for k, v in self.keyword_postings.items()}
        self.evidence_postings = {k: np.asarray(v, np.int64) for k, v in self.evidence_postings.items()}
        # zero-overlap rows are ordered by position, then hash, then index
        self.by_position = defaultdict(list)
        for local, p in enumerate(self.position.tolist()):
            self.by_position[p].append(local)
        self.positions = sorted(self.by_position)

    def _hash(self, query: str, local: int) -> str:
        return hashlib.sha256(f"{query}\0{self.url[local]}".encode()).hexdigest()

    def select(self, query: str, limit: int = SEARCH_RESULT_LIMIT) -> list[int]:
        terms = self._tokens(query)
        overlap = defaultdict(int)
        for token in terms:
            for local in self.keyword_postings.get(token, ()):
                overlap[int(local)] += 4
            for local in self.evidence_postings.get(token, ()):
                overlap[int(local)] += 1
        exact = set(self.by_keyword.get(query.casefold(), ()))
        hits = set(overlap) | exact
        key = lambda local: (-int(local in exact), -overlap.get(local, 0), int(self.position[local]), self._hash(query, local), local)
        chosen = sorted(hits, key=key)[:limit]
        if len(chosen) < limit:
            for p in self.positions:
                pool = sorted((local for local in self.by_position[p] if local not in hits),
                              key=lambda local: (self._hash(query, local), local))
                chosen += pool[:limit - len(chosen)]
                if len(chosen) >= limit:
                    break
        return [int(self.ids[local]) for local in chosen]


# ---------------------------------------------------------------- one trace -> rows per stage

def _events(trace: dict, kind: str) -> list[dict]:
    return [e["payload"] for e in trace.get("events", []) if isinstance(e, dict) and e.get("event_type") == kind]


def answer_items(trace: dict, method: str, engine: str, presented_urls: list, ranking_urls: list,
                 lookup: dict, rows: SnapshotRows, index: LexicalIndex | None = None) -> dict:
    """Exact snapshot rows behind one answer.

    Returns ``items`` (distinct retrieved rows in first-seen order, each with the search that first
    returned it and its rank there, whether the reranker scored it, its presented slot and its rank in
    the generator's list, −1 when absent), ``events`` (each reranker event: the search it scored or −1
    for Parallel, and its candidates as (row, score, selection order)), ``queries`` and ``counts``
    (raw rows used, replays used, unresolved items). Raises ValueError when the trace is inconsistent."""
    counts = defaultdict(int)
    searches = _events(trace, "search")
    compactions = _events(trace, "compaction")
    search_rows = []
    for s in searches:
        raw = (s.get("raw_payload") or {}).get("rows")
        snippets = s["snippets"]
        mapped = None
        if isinstance(raw, list) and len(raw) == len(snippets):
            mapped = []
            for r, snip in zip(raw, snippets):
                row = lookup.get((engine, r.get("keyword"), int(r.get("position", -1))))
                if row is None or rows.url[row] != snip["url"] or rows.title[row] != snip["title"] or rows.snippet[row] != snip["text"]:
                    mapped = None
                    break
                mapped.append(row)
            counts["searches_from_raw_rows"] += mapped is not None
        if mapped is None:
            if index is None:
                raise ValueError("search lacks usable raw rows and no replay index was given")
            mapped = index.select(s["query"])
            if [(rows.url[i], rows.title[i], rows.snippet[i]) for i in mapped] != [(x["url"], x["title"], x["text"]) for x in snippets]:
                raise ValueError("replayed search does not reproduce the recorded rows")
            counts["searches_from_replay"] += 1
        search_rows.append(mapped)

    first = {}  # row -> (search, rank)
    for s, mapped in enumerate(search_rows):
        for rank, row in enumerate(mapped):
            first.setdefault(row, (s, rank))

    events = []
    if method == PARALLEL:
        if len(compactions) != 1:
            raise ValueError("parallel trace must have exactly one compaction")
        by_url = {}
        for mapped in search_rows:
            for row in mapped:
                by_url.setdefault(rows.url[row], row)
        targets = [(c, None) for c in compactions]
    elif method == REACTIVE:
        if len(compactions) != len(searches):
            raise ValueError("reactive trace must have one compaction per search")
        targets = [(c, s) for s, c in enumerate(compactions)]
    else:
        raise ValueError(f"unknown method {method}")

    scored_rows = {}
    for c, s in targets:
        if s is not None:
            if c["query"] != searches[s]["query"]:
                raise ValueError("reactive compaction does not score its own search")
            pool = defaultdict(list)
            for row in search_rows[s]:
                pool[(rows.url[row], rows.title[row], rows.snippet[row])].append(row)
        kept = {r["source_index"]: j for j, r in enumerate(c["selected_snippets"])}
        candidates = []
        for snip in c["scored_snippets"]:
            if s is None:
                row = by_url.get(snip["url"])
                if row is not None and (rows.title[row], rows.snippet[row]) != (snip["title"], snip["text"]):
                    row = None
            else:
                stack = pool.get((snip["url"], snip["title"], snip["text"]))
                row = stack.pop(0) if stack else None
            if row is None:
                raise ValueError("scored candidate does not come from the answer's searches")
            candidates.append((row, float(snip["score"]), kept.get(snip["source_index"], -1)))
            scored_rows.setdefault(row, len(events))
        events.append({"search": -1 if s is None else s, "candidates": candidates})

    selected_by_url = {}
    for e in events:
        for row, _, order in e["candidates"]:
            if order >= 0:
                selected_by_url.setdefault(rows.url[row], row)
    presented = {}
    for slot, url in enumerate(presented_urls):
        row = selected_by_url.get(url)
        if row is None:
            raise ValueError("presented evidence was not selected by the reranker")
        presented[row] = slot
    ranked = {}
    for r, url in enumerate(ranking_urls):
        row = selected_by_url.get(url)
        if row is None or row not in presented:
            raise ValueError("ranked evidence was not presented")
        ranked[row] = r

    items = []
    for row, (s, rank) in sorted(first.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        items.append((row, s, rank, int(row in scored_rows), presented.get(row, -1), ranked.get(row, -1)))
    if any(i[3] == 0 and i[4] >= 0 for i in items):
        raise ValueError("a presented row was never scored")
    return {"items": items, "events": events, "queries": [s["query"] for s in searches], "counts": dict(counts)}

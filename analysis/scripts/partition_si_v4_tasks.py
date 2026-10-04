#!/usr/bin/env python3
"""Partition a verified SI-v4 freeze without splitting shared answer maps.

Only preparation happens here. Task and cell JSONL bytes remain unchanged;
SQLite holds the corpus on disk and gzip streams have deterministic headers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing, contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.scripts.run_source_importance_judge import file_hash

VERSION = "si-v4-map-partition-v1"
FILES = ("tasks.jsonl.gz", "cells.jsonl.gz")
ORDERING = {
    "direction": "keyword_first_forward",
    "keys": ["missing_priority", "priority_rank", "keyword_id", "generator", "prompt_id", "cell_identity"],
    "shared_map": "keep together at the earliest member's priority, including across generators",
    "missing_priority": "after ranked cells; keyword and stable identities break ties",
    "basis": "horeka_qwen_bouts.remaining_rows priority_rank/keyword_id/prompt_id order; generator groups within keyword",
}
ORDER = "missing_priority,priority_rank,keyword_id,generator,prompt_id,identity"


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _records(path):
    with gzip.open(path, "rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            if not raw.endswith(b"\n"):
                raise ValueError("frozen JSONL records must end with a newline")
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("frozen JSONL record must be an object")
            yield record, raw


@contextmanager
def _gzip(path):
    with Path(path).open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as compressed:
            yield compressed


def _verify_files(inputs, manifest, manifest_sha256):
    if file_hash(inputs / "manifest.json") != manifest_sha256:
        raise ValueError("source manifest changed during partitioning")
    for name in FILES:
        if file_hash(inputs / name) != manifest.get("files", {}).get(name):
            raise ValueError(f"frozen input checksum mismatch: {name}")


def _priority(cell):
    membership, metadata = cell.get("keyword_memberships") or {}, cell.get("task_metadata") or {}
    rank = membership.get("primary_priority_rank", metadata.get("priority_rank", metadata.get("primary_priority_rank")))
    if rank is not None and (type(rank) is not int or not 0 <= rank < 2**63):
        raise ValueError("keyword priority must be a nonnegative integer")
    keyword = membership.get("primary_keyword_id") or metadata.get("keyword_id") or metadata.get("primary_keyword_id")
    if not keyword:
        keys = membership.get("keyword_ids") or []
        keyword = min(keys) if keys else ""
    if not isinstance(keyword, str):
        raise ValueError("keyword identity must be a string")
    return int(rank is None), rank or 0, keyword


def _task_index(db, inputs):
    for record, raw in _records(inputs / "tasks.jsonl.gz"):
        kind = record.get("task")
        answer_sha = request_sha = source_sha = None
        if kind == "fulfilment" and record.get("protocol") == v3.PROTOCOL:
            v3.item_from_record(record)
            answer, request = record["answer"], record["request"]
        elif kind == "answer_map" and record.get("protocol") == v4.PROTOCOL:
            if record.get("task_version") != v4.TASK_VERSION or record.get("max_tokens") != 4096:
                raise ValueError("partition requires current v4 maps with 4096 tokens")
            if "diagnostic_seed" in record.get("inputs", {}):
                raise ValueError("diagnostic interventions are not corpus inputs")
            v4.item_from_record(record)
            answer, request = record["inputs"]["answer"], record["inputs"]["request"]
        elif kind == "source_dependency" and record.get("protocol") == v4.PROTOCOL:
            if record.get("max_tokens") != 4096:
                raise ValueError("partition requires source dependencies with 4096 tokens")
            expected = v4.source_dependency(record["map_task_id"], record["source_title"], record["source_text"], 4096)
            if expected != record:
                raise ValueError("invalid frozen source dependency")
            source_sha = v4._digest({"title": record["source_title"], "text": record["source_text"]})
            answer = request = None
        else:
            raise ValueError("unexpected task kind/protocol in v4 corpus freeze")
        if answer is not None:
            answer_sha = hashlib.sha256(answer.encode()).hexdigest()
            request_sha = hashlib.sha256(request.encode()).hexdigest()
        try:
            db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?)", (
                record["judge_task_id"], kind, record.get("map_task_id"), raw, answer_sha, request_sha, source_sha))
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate frozen task identity") from exc
    dangling = db.execute("""SELECT d.id FROM tasks d LEFT JOIN tasks m ON d.parent=m.id
        WHERE d.kind='source_dependency' AND (m.id IS NULL OR m.kind!='answer_map') LIMIT 1""").fetchone()
    if dangling:
        raise ValueError(f"dependency refers to missing/non-map task: {dangling[0]}")


def _cell_index(db, inputs, partial):
    blocked = Counter()
    with _gzip(partial / "blocked-cells.jsonl.gz") as blocked_file:
        for cell, raw in _records(inputs / "cells.jsonl.gz"):
            if "stored_answer_sensitivity" in cell:
                raise ValueError("sensitivity inputs are not supported by this single-pass partition")
            identity = cell.get("fingerprint") or f"{cell.get('model', '')}:{cell.get('cell_id', '')}"
            if not isinstance(identity, str) or not cell.get("cell_id"):
                raise ValueError("cell identity is missing")
            try:
                db.execute("INSERT INTO input_cells VALUES (?)", (identity,))
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate frozen cell identity") from exc
            if cell.get("status") != "ok":
                blocked[str(cell.get("status", "missing_status"))] += 1
                blocked_file.write(raw)
                continue
            mapper = db.execute("SELECT kind,answer_sha,request_sha FROM tasks WHERE id=?", (cell.get("map_task_id"),)).fetchone()
            if mapper is None or mapper[0] != "answer_map":
                raise ValueError("ready cell references missing/non-map task")
            if mapper[1] != cell.get("masked_answer_sha256"):
                raise ValueError("cell answer does not match its map task")
            j1 = db.execute("SELECT kind,request_sha FROM tasks WHERE id=?", (cell.get("j1_task_id"),)).fetchone()
            if j1 is None or j1[0] != "fulfilment" or j1[1] != mapper[2]:
                raise ValueError("cell J1 reference is missing, foreign, or differs from map request")
            source_count = unassessable = 0
            for source in cell.get("sources", []):
                dependency = source.get("dependency_id")
                if not dependency:
                    if source.get("status") != "unassessable_input" or source.get("judge_task_id"):
                        raise ValueError("ready cell source lacks its v4 dependency")
                    unassessable += 1
                    continue
                dep = db.execute("SELECT kind,parent,source_sha FROM tasks WHERE id=?", (dependency,)).fetchone()
                if dep is None or dep[0] != "source_dependency":
                    raise ValueError("cell references missing/non-source dependency")
                if dep[1] != cell["map_task_id"] or source.get("map_task_id", dep[1]) != dep[1]:
                    raise ValueError("source joins to a different answer map")
                if source.get("source_sha256") != dep[2]:
                    raise ValueError("source hash does not match its dependency")
                source_count += 1
                db.execute("INSERT OR IGNORE INTO refs VALUES (?,?)", (identity, dependency))
            for task_id in (cell["map_task_id"], cell["j1_task_id"]):
                db.execute("INSERT INTO refs VALUES (?,?)", (identity, task_id))
            missing, rank, keyword = _priority(cell)
            db.execute("INSERT INTO cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                identity, cell["map_task_id"], raw, missing, rank, keyword,
                cell.get("model", ""), cell.get("prompt_id", ""), cell.get("answer_source", "none"),
                source_count, unassessable, cell["cell_id"], cell["j1_task_id"]))
    foreign = db.execute("SELECT id FROM tasks WHERE NOT EXISTS (SELECT 1 FROM refs WHERE task_id=tasks.id) LIMIT 1").fetchone()
    if foreign:
        raise ValueError(f"unreferenced/foreign frozen task: {foreign[0]}")
    return dict(blocked)


def _assign(db, valid_cells, max_shards, minimum_cells_per_shard):
    db.execute(f"""CREATE TABLE components AS SELECT map_id,size,{ORDER},0 AS shard,0 AS sequence
        FROM (SELECT map_id,{ORDER},count(*) OVER (PARTITION BY map_id) AS size,
              row_number() OVER (PARTITION BY map_id ORDER BY {ORDER}) AS first
              FROM cells) WHERE first=1""")
    db.execute("CREATE UNIQUE INDEX component_map ON components(map_id)")
    components = db.execute("SELECT count(*) FROM components").fetchone()[0]
    requested = min(max_shards, max(1, (valid_cells + minimum_cells_per_shard - 1) // minimum_cells_per_shard))
    count = min(requested, components)
    if not count:
        return requested, 0
    shard, current, remaining = 1, 0, valid_cells
    for index, (map_id, size) in enumerate(db.execute(f"SELECT map_id,size FROM components ORDER BY {ORDER}"), 1):
        groups_left = components - index + 1
        target = remaining / (count - shard + 1)
        if current and shard < count and (
            groups_left == count - shard or abs(current - target) <= abs(current + size - target)
        ):
            remaining -= current
            shard += 1
            current = 0
        db.execute("UPDATE components SET shard=?,sequence=? WHERE map_id=?", (shard, index, map_id))
        current += size
    db.execute("CREATE INDEX component_shards ON components(shard,sequence)")
    return requested, count


def _write_shard(db, manifest, source_hash, partial, output, number):
    shard_id = f"shard-{number:04d}"
    destination = partial / shard_id / "inputs"
    destination.mkdir(parents=True)
    counts = Counter()
    with _gzip(destination / "cells.jsonl.gz") as stream:
        for raw, answer_source, references, unassessable in db.execute(f"""
            SELECT c.raw,c.answer_source,c.source_count,c.unassessable FROM cells c
            JOIN components g ON c.map_id=g.map_id WHERE g.shard=?
            ORDER BY g.sequence,{','.join('c.' + key for key in ORDER.split(','))}""", (number,)):
            stream.write(raw)
            counts["cells_ok"] += 1
            counts[f"answer_{answer_source}"] += 1
            counts["source_tasks_referenced"] += references
            counts["sources_unassessable"] += unassessable
    with _gzip(destination / "tasks.jsonl.gz") as stream:
        for kind, raw in db.execute("""SELECT t.kind,t.raw FROM tasks t JOIN (
            SELECT r.task_id,min(g.sequence) AS sequence FROM refs r
            JOIN cells c ON r.cell_id=c.identity JOIN components g ON c.map_id=g.map_id
            WHERE g.shard=? GROUP BY r.task_id) selected ON selected.task_id=t.id
            ORDER BY selected.sequence,CASE t.kind WHEN 'answer_map' THEN 0 WHEN 'source_dependency' THEN 1 ELSE 2 END,t.id""", (number,)):
            stream.write(raw)
            counts[f"unique_{kind}_tasks"] += 1
    counts["stored_answer_sensitivity_cells"] = 0
    shard_manifest = {**manifest, "cells": counts["cells_ok"], "counts": dict(counts),
                      "files": {name: file_hash(destination / name) for name in FILES},
                      "source_manifest_sha256": source_hash,
                      "partition": {"format_version": VERSION, "shard_id": shard_id, "ordering": ORDERING}}
    _json(destination / "manifest.json", shard_manifest)
    return {"id": shard_id, "directory": str(output / shard_id), "cells": counts["cells_ok"],
            "manifest_sha256": file_hash(destination / "manifest.json"), "counts": dict(counts)}


def partition(inputs: Path, output: Path, *, max_shards=200, minimum_cells_per_shard=1500,
              index_directory=None) -> list[dict]:
    """Return nonempty contiguous shards; shared maps override the target balance.

    Ready cells have status ``ok``. Blocked cells remain in a separate raw stream.
    If fewer map components exist than the requested shard count, emit fewer
    shards. A failed preparation preserves its .partial directory for inspection.
    ``index_directory`` (e.g. node-local $TMPDIR) holds the temporary SQLite index;
    shard contents do not depend on where it lives.
    """
    if any(type(n) is not int or n <= 0 for n in (max_shards, minimum_cells_per_shard)):
        raise ValueError("partition sizes must be positive integers")
    inputs, output = Path(inputs).resolve(), Path(output).resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise ValueError("partition output already exists; preserve it and choose a new directory")
    manifest = json.loads((inputs / "manifest.json").read_text())
    source_hash = file_hash(inputs / "manifest.json")
    if manifest.get("protocol") != v4.PROTOCOL or manifest.get("task_version") != v4.TASK_VERSION:
        raise ValueError("partition requires the current SI-v4 corpus freeze")
    if manifest.get("max_tokens") != 4096 or manifest.get("map_max_tokens") != 4096:
        raise ValueError("partition preserves 4096-token map and source budgets")
    if manifest.get("truncation_sensitivity", {}).get("fraction") != 0 or manifest.get("counts", {}).get("stored_answer_sensitivity_cells", 0):
        raise ValueError("sensitivity inputs are not supported by this single-pass partition")
    if manifest.get("constructed"):
        raise ValueError("constructed diagnostics are not corpus inputs")
    _verify_files(inputs, manifest, source_hash)
    partial.mkdir(parents=True)
    index_path = (Path(tempfile.mkdtemp(prefix="si-partition-index-", dir=index_directory)) / "partition-index.sqlite"
                  if index_directory else partial / "partition-index.sqlite")
    with closing(sqlite3.connect(index_path)) as db:
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-8192")
        db.executescript("""
            CREATE TABLE tasks (id TEXT PRIMARY KEY,kind TEXT NOT NULL,parent TEXT,raw BLOB NOT NULL,
                                answer_sha TEXT,request_sha TEXT,source_sha TEXT);
            CREATE TABLE input_cells (identity TEXT PRIMARY KEY);
            CREATE TABLE cells (identity TEXT PRIMARY KEY,map_id TEXT NOT NULL,raw BLOB NOT NULL,
                missing_priority INTEGER,priority_rank INTEGER,keyword_id TEXT,generator TEXT,prompt_id TEXT,
                answer_source TEXT,source_count INTEGER,unassessable INTEGER,cell_id TEXT,j1_id TEXT);
            CREATE TABLE refs (cell_id TEXT NOT NULL,task_id TEXT NOT NULL,PRIMARY KEY(cell_id,task_id));
            CREATE INDEX task_refs ON refs(task_id);
        """)
        _task_index(db, inputs)
        db.commit()
        blocked = _cell_index(db, inputs, partial)
        db.commit()
        db.execute(f"CREATE INDEX cells_map_order ON cells(map_id,{ORDER})")
        valid_cells = db.execute("SELECT count(*) FROM cells").fetchone()[0]
        requested, shard_count = _assign(db, valid_cells, max_shards, minimum_cells_per_shard)
        shards = [_write_shard(db, manifest, source_hash, partial, output, i) for i in range(1, shard_count + 1)]
        duplicate = db.execute("""SELECT r.task_id FROM refs r JOIN tasks t ON r.task_id=t.id
            JOIN cells c ON r.cell_id=c.identity JOIN components g ON c.map_id=g.map_id
            WHERE t.kind!='fulfilment' GROUP BY r.task_id HAVING count(DISTINCT g.shard)>1 LIMIT 1""").fetchone()
        if duplicate or sum(s["cells"] for s in shards) != valid_cells:
            raise ValueError("partition lost cells or repeated a map/source task across shards")
        task_counts = {kind: count for kind, count in db.execute("SELECT kind,count(*) FROM tasks GROUP BY kind")}
        summary = {"format_version": VERSION, "source_directory": str(inputs),
                   "source_manifest_sha256": source_hash, "source_files": manifest["files"],
                   "ordering": ORDERING, "max_shards": max_shards,
                   "minimum_cells_per_shard": minimum_cells_per_shard, "requested_shards": requested,
                   "counts": {"input_cells": valid_cells + sum(blocked.values()), "ready_cells": valid_cells,
                              "blocked_input_cells": sum(blocked.values()), "shards": len(shards),
                              "map_components": task_counts.get("answer_map", 0),
                              "unique_tasks": task_counts},
                   "blocked_input_counts": blocked, "blocked_cells_file": "blocked-cells.jsonl.gz",
                   "blocked_cells_sha256": file_hash(partial / "blocked-cells.jsonl.gz"),
                   "j1_policy": "references preserved; may repeat across shards; execution must set source_importance_only=true",
                   "shards": shards, "scientific_result": False}
        _json(partial / "partition.json", summary)
    _verify_files(inputs, manifest, source_hash)
    index_path.unlink()
    if index_directory:
        index_path.parent.rmdir()
    if output.exists():
        raise ValueError("partition output appeared during preparation; partial output preserved")
    partial.rename(output)
    return [{key: shard[key] for key in ("id", "directory", "cells")} for shard in shards]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=200)
    parser.add_argument("--minimum-cells-per-shard", type=int, default=1500)
    args = parser.parse_args(argv)
    print(json.dumps(partition(args.inputs, args.output, max_shards=args.max_shards,
                               minimum_cells_per_shard=args.minimum_cells_per_shard), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

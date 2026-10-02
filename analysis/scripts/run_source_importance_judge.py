#!/usr/bin/env python3
"""Run a finite frozen SI-v3/v4 backlog against an already running judge server.

Never allocates GPUs or starts a server. SQLite is a local index; sealed records
and the existing ledger retain task provenance. One coordinator owns an output
directory at a time. Different models/configurations/repetitions use new outputs.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline import source_importance as v3
from analysis.interpretability.pipeline import source_importance_v4 as v4
from analysis.interpretability.pipeline.agentic_judging import _digest
from analysis.interpretability.pipeline.agentic_dataset import (
    FinalDatasetWriter, initialize_dataset, verify_record_reference, recover_inprogress_writer,
)
from analysis.interpretability.pipeline.agentic_task_ledger import StripedTaskLedger, LedgerClaim
from analysis.interpretability.pipeline.agentic_storage import storage_health
from analysis.interpretability.pipeline.inference_claims import ClaimIdentity
from analysis.interpretability.pipeline.inference_budget import AllocationBudget
from analysis.scripts.run_acl_arr_vllm import VllmChatClient, _execute_one, _iter_execute, _request_sha256


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"model_id", "model_revision", "chat_template_sha256", "chat_template_kwargs",
                "precision", "serving_version", "hardware", "gpu_count", "tensor_parallel_size",
                "concurrency", "context_length", "tokenizer_path", "request_timeout", "repetition_id"}
    if not required <= config.keys():
        raise ValueError(f"execution configuration missing {sorted(required - config.keys())}")
    allowed = required | {"runtime_versions", "eager", "gpu_memory_utilization", "temperature", "seed_policy",
                          "structured_outputs_config", "source_importance_only"}
    if config.keys() - allowed:
        raise ValueError(f"unsupported execution configuration fields: {sorted(config.keys() - allowed)}")
    if "source_importance_only" in config and type(config["source_importance_only"]) is not bool:
        raise ValueError("source_importance_only must be a boolean")
    if "structured_outputs_config" in config:
        structured = config["structured_outputs_config"]
        if (structured != {"backend": "xgrammar", "disable_any_whitespace": True}
                or structured["disable_any_whitespace"] is not True):
            raise ValueError("structured outputs require xgrammar with disable_any_whitespace=true")
    if not all(isinstance(config[k], str) and config[k] for k in (
            "model_id", "model_revision", "chat_template_sha256", "precision", "serving_version",
            "tokenizer_path", "repetition_id")):
        raise ValueError("execution identity fields must be nonempty strings")
    if any(type(config[k]) is not int or config[k] <= 0 for k in (
            "gpu_count", "tensor_parallel_size", "concurrency", "context_length", "request_timeout")):
        raise ValueError("execution counts and budgets must be positive integers")
    if config["chat_template_kwargs"] != {"enable_thinking": False} or config["chat_template_kwargs"]["enable_thinking"] is not False:
        raise ValueError("first v4 configuration uses explicit thinking=false; reasoning needs separate validation")
    if config["precision"] != "bfloat16":
        raise ValueError("first comparison preserves BF16 precision")
    if config.get("temperature", 0.0) != 0.0 or config.get("seed_policy", "semantic-task-derived-v1") != "semantic-task-derived-v1":
        raise ValueError("first comparison preserves temperature zero and semantic task seeds")
    # Credentials belong in VLLM_API_KEY, never in a published configuration.
    if any(k.lower() in {"token", "hf_token", "access_token", "api_key", "secret", "password"} for k in config):
        raise ValueError("credentials are not configuration fields")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]) or not re.fullmatch(r"[0-9a-f]{64}", config["chat_template_sha256"]):
        raise ValueError("model revision and template require immutable hashes")
    return config


class ContextCheckedClient:
    """Count the actual templated request, including validation feedback, locally."""
    def __init__(self, client, tokenizer, config):
        self.client, self.tokenizer, self.config = client, tokenizer, config

    async def complete(self, **kwargs):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": kwargs["prompt"]}], tokenize=True,
            add_generation_prompt=True, **self.config["chat_template_kwargs"])
        if len(ids) + kwargs["max_tokens"] > self.config["context_length"]:
            raise ValueError("context budget exceeded; input was not truncated")
        return await self.client.complete(**kwargs)


def check_storage(workspace, quota_path, output=None):
    quota = json.loads(Path(quota_path).read_text())
    captured = quota.get("captured_at_epoch")
    quota["fresh"] = (type(captured) is int and 0 <= time.time() - captured <= 300 and
                      Path(quota.get("workspace", "")).resolve() == Path(workspace).resolve())
    status = storage_health(Path(workspace), quota_evidence=quota)
    if output is not None:
        output_status = storage_health(Path(output), quota_evidence=quota)
        status["safe_to_admit"] &= output_status["safe_to_admit"]
        status["reasons"] += output_status["reasons"]
    return status


class Coordinator:
    def __init__(self, inputs, output, config, *, recovery_evidence=None, fixed_maps=None, admission_check=None):
        self.inputs, self.output, self.config = Path(inputs), Path(output), config
        self.admission_check, self.admission_stop = admission_check, None
        if type(config.get("source_importance_only", False)) is not bool:
            raise ValueError("source_importance_only must be a boolean")
        self.manifest = json.loads((self.inputs / "manifest.json").read_text())
        if self.manifest["protocol"] not in (v3.PROTOCOL, v4.PROTOCOL, v4.V3_COMPARISON_PROTOCOL):
            raise ValueError("unsupported frozen protocol")
        for name in ("tasks.jsonl.gz", "cells.jsonl.gz"):
            if file_hash(self.inputs / name) != self.manifest["files"][name]:
                raise ValueError(f"frozen input checksum mismatch: {name}")
        self.run_identity = {"format_version": "si-execution-v1", "configuration": config,
                             "transport_maximum_attempts": 3,
                             "manifest_sha256": file_hash(self.inputs / "manifest.json"),
                             "fixed_maps_sha256": file_hash(fixed_maps) if fixed_maps else None}
        self.execution_sha256 = _digest(self.run_identity)
        initialize_dataset(self.output, population_id=self.run_identity["manifest_sha256"],
                           acceptance_policy_id=self.execution_sha256)
        self.lock = (self.output / "control/coordinator.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("another coordinator owns this output") from None
        self.writer_id = "si-" + uuid.uuid4().hex[:16]
        self.db = sqlite3.connect(self.output / "control/index.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, record TEXT NOT NULL, kind TEXT NOT NULL, parent TEXT,
            priority INTEGER NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending', result TEXT, claim TEXT, identity TEXT,
            refs TEXT, writer_id TEXT, job_id TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ready_tasks ON tasks(state,parent)")
        self.db.execute("CREATE INDEX IF NOT EXISTS ready_priority ON tasks(state,priority)")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS timing (phase TEXT, seconds REAL, gpu_count INTEGER)")
        self.db.execute("CREATE TABLE IF NOT EXISTS quarantined_maps (id TEXT PRIMARY KEY)")
        self.ledger = StripedTaskLedger(self.output / "control/task-ledger")
        try:
            self._import()
            self._recover(recovery_evidence)
            if fixed_maps:
                self._fixed_maps(fixed_maps)
            self.db.execute("""UPDATE tasks SET state='pending' WHERE state='waiting'
                AND parent IN (SELECT id FROM tasks WHERE kind='answer_map' AND state='done')""")
            self.db.commit()
            self.writer = FinalDatasetWriter(self.output, writer_id=self.writer_id)
            self.writer.append("provenance", {**self.run_identity, "execution_sha256": self.execution_sha256,
                               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "scientific_result": False},
                               transaction_id=self.writer_id)
        except BaseException:
            self.db.close()
            self.lock.close()
            raise
        self.pending_seals = 0

    def _import(self):
        if self.db.execute("SELECT value FROM metadata WHERE key='imported'").fetchone():
            return
        with self.db:
            for record in rows(self.inputs / "tasks.jsonl.gz"):
                expected_protocol = v3.PROTOCOL if record.get("task") == "fulfilment" else self.manifest["protocol"]
                if record.get("protocol") != expected_protocol:
                    raise ValueError("task protocol differs from its frozen manifest")
                if record.get("protocol") == v4.V3_COMPARISON_PROTOCOL:
                    v4.comparison_from_record(record)
                elif record.get("protocol") == v4.PROTOCOL:
                    if record["task"] == "source_dependency":
                        expected = v4.source_dependency(record["map_task_id"], record["source_title"],
                                                        record["source_text"], record["max_tokens"], record.get("diagnostic_passage_ids"),
                                                        record.get("diagnostic_seed"))
                        if expected != record:
                            raise ValueError("invalid frozen dependency")
                        if ("diagnostic_seed" in record or "diagnostic_passage_ids" in record) and not self.manifest.get("constructed"):
                            raise ValueError("diagnostic interventions require a constructed input freeze")
                    else:
                        v4.item_from_record(record)
                else:
                    v3.item_from_record(record)
                self.db.execute("INSERT INTO tasks(id,record,kind,parent,priority,state) VALUES (?,?,?,?,?,?)",
                                (record["judge_task_id"], canonical(record), record["task"], record.get("map_task_id"),
                                 1 if record["task"] == "answer_map" else 2 if record["task"] == "fulfilment" else 0,
                                 "not_requested" if record["task"] == "fulfilment" and self.config.get("source_importance_only") else
                                 "waiting" if record["task"] == "source_dependency" else "pending"))
            dangling = self.db.execute("""SELECT d.id FROM tasks d LEFT JOIN tasks m ON d.parent=m.id
                WHERE d.parent IS NOT NULL AND (m.id IS NULL OR m.kind!='answer_map') LIMIT 1""").fetchone()
            if dangling:
                raise ValueError("dependency refers to missing/non-map task")
            for cell in rows(self.inputs / "cells.jsonl.gz"):
                for bundle in (cell, cell.get("stored_answer_sensitivity", {})):
                    ids = [bundle.get("j1_task_id"), bundle.get("map_task_id")]
                    ids += [s.get("dependency_id", s.get("judge_task_id")) for s in bundle.get("sources", [])]
                    for tid in filter(None, ids):
                        if not self.db.execute("SELECT 1 FROM tasks WHERE id=?", (tid,)).fetchone():
                            raise ValueError("cell references missing frozen task")
                    if bundle.get("map_task_id"):
                        mapper = json.loads(self.db.execute("SELECT record FROM tasks WHERE id=?", (bundle["map_task_id"],)).fetchone()[0])
                        if hashlib.sha256(mapper["inputs"]["answer"].encode()).hexdigest() != bundle.get("masked_answer_sha256"):
                            raise ValueError("cell answer does not match its map task")
                        for source in bundle.get("sources", []):
                            if not source.get("dependency_id"):
                                continue
                            dep = json.loads(self.db.execute("SELECT record FROM tasks WHERE id=?", (source["dependency_id"],)).fetchone()[0])
                            if dep.get("map_task_id") != bundle["map_task_id"]:
                                raise ValueError("source joins to a different answer map")
                            if source.get("source_sha256") != _digest({"title": dep["source_title"], "text": dep["source_text"]}):
                                raise ValueError("source hash does not match its dependency")
                        j1 = json.loads(self.db.execute("SELECT record FROM tasks WHERE id=?", (bundle["j1_task_id"],)).fetchone()[0])
                        if j1["request"] != mapper["inputs"]["request"]:
                            raise ValueError("J1 and map requests differ")
            self.db.execute("INSERT INTO metadata VALUES ('imported','true')")

    def _recover(self, evidence):
        # Ordinary cancellation seals outputs and checkpoints claims. Abrupt death
        # requires explicit terminal-owner evidence, never a timeout heuristic.
        active = self.db.execute("SELECT DISTINCT writer_id,job_id FROM tasks WHERE state IN ('running','saved')").fetchall()
        evidence_rows = [] if evidence is None else json.loads(Path(evidence).read_text())
        for writer_id, job_id in active:
            confirmation = next((r for r in evidence_rows if r.get("writer_id") == writer_id and
                                 r.get("slurm_job_id") == job_id and r.get("owner_terminal") is True and
                                 r.get("evidence")), None)
            if not confirmation:
                raise ValueError(f"resume requires terminal-owner evidence for {writer_id}, job {job_id}")
            recover_inprogress_writer(self.output, writer_id=writer_id)
            recovered = self._unindexed_results(writer_id)
            for tid, state, result, claim_raw, identity_raw, refs_raw in self.db.execute(
                    "SELECT id,state,result,claim,identity,refs FROM tasks WHERE writer_id=? AND state IN ('running','saved')",
                    (writer_id,)).fetchall():
                claim = LedgerClaim(**json.loads(claim_raw)) if claim_raw else None
                if state == "running" and tid in recovered:
                    result, references = recovered[tid]
                    identity = json.loads(identity_raw)
                    if result["request_sha256"] != identity["request_sha256"]:
                        raise ValueError("recovered request differs from the claimed input")
                    refs_raw, state = canonical(references), "saved"
                    self.db.execute("UPDATE tasks SET state='saved',result=?,refs=? WHERE id=?",
                                    (canonical(result), refs_raw, tid))
                    self.db.commit()
                if state == "saved":
                    refs = json.loads(refs_raw)
                    if not all(verify_record_reference(self.output, r) for r in refs):
                        raise ValueError("saved result has unverified references")
                    latest = self.ledger.inspect(ClaimIdentity(**json.loads(identity_raw)))
                    if latest["state"] not in ("completed", "terminal_failed"):
                        payload = json.loads(result) if isinstance(result, str) else result
                        self.ledger.transition(claim, state="completed" if payload.get("ok") else "terminal_failed",
                                               record_references=refs)
                    self.db.execute("UPDATE tasks SET state='done' WHERE id=?", (tid,))
                else:
                    if claim:
                        self.ledger.release_stale(ClaimIdentity(**json.loads(identity_raw)),
                            expected_token=claim.token, scheduler_confirmation=confirmation)
                    self.db.execute("UPDATE tasks SET state='pending',claim=NULL WHERE id=?", (tid,))
            self.db.commit()
        # Validate persisted result references before permitting cache reuse.
        for result, refs in self.db.execute("SELECT result,refs FROM tasks WHERE state='done' AND refs IS NOT NULL"):
            references = json.loads(refs)
            if not all(verify_record_reference(self.output, r) for r in references):
                raise ValueError("cached result lost its sealed provenance")
            ref = next(r for r in references if r["table"] == "judgments")
            expected_id = "record-" + _digest({"table": ref["table"], "transaction_id": ref["transaction_id"],
                                               "row": json.loads(result)})
            if expected_id != ref["record_id"]:
                raise ValueError("cached result differs from sealed record")

    def _unindexed_results(self, writer_id):
        """Reconcile the fsynced-result/SQLite-commit crash window without rerunning it."""
        pending = {r[0] for r in self.db.execute("SELECT id FROM tasks WHERE writer_id=? AND state='running'", (writer_id,))}
        results, wanted_inputs, inputs = {}, set(), {}
        for table in ("judgments", "judge_inputs"):
            for path in sorted((self.output / "data" / table).glob(f"part-{writer_id}-*.jsonl")):
                sequence = int(path.stem.rsplit("-", 1)[1])
                with path.open() as stream:
                    for number, line in enumerate(stream, 1):
                        envelope = json.loads(line)
                        row = envelope["row"]
                        wanted = (row.get("logical_id") in pending if table == "judgments" else
                                  envelope["record_id"] in wanted_inputs)
                        if not wanted:
                            continue
                        ref = {"table": table, "writer_id": writer_id, "shard_sequence": sequence,
                               "line_number": number, "record_id": envelope["record_id"],
                               "transaction_id": envelope["transaction_id"]}
                        if not verify_record_reference(self.output, ref):
                            raise ValueError("unindexed result failed sealed verification")
                        if table == "judgments":
                            tid = row["logical_id"]
                            if tid in results or row.get("execution_sha256") != self.execution_sha256:
                                raise ValueError("ambiguous unindexed result")
                            results[tid] = (row, ref)
                            wanted_inputs.add(row["input_record_id"])
                        else:
                            inputs[envelope["record_id"]] = ref
        if wanted_inputs - inputs.keys():
            raise ValueError("unindexed result lacks its input record")
        return {tid: (row, [inputs[row["input_record_id"]], ref]) for tid, (row, ref) in results.items()}

    def _fixed_maps(self, path):
        saved = self.output / "artifacts/fixed-maps.jsonl"
        if not saved.exists():
            shutil.copyfile(path, saved)
        if file_hash(saved) != self.run_identity["fixed_maps_sha256"]:
            raise ValueError("saved fixed maps differ from the frozen input")
        seen = set()
        for row in rows(path):
            tid = row["judge_task_id"]
            if tid in seen:
                raise ValueError("duplicate fixed map")
            seen.add(tid)
            existing = self.db.execute("SELECT record,state,result FROM tasks WHERE id=? AND kind='answer_map'", (tid,)).fetchone()
            if not existing:
                raise ValueError("fixed map does not belong to this frozen input")
            item = v4.item_from_record(json.loads(existing[0]))
            parsed = item["validator"](row["raw_output"])
            result = {"ok": True, "parsed_output": parsed, "raw_output": row["raw_output"],
                      "judge_task_id": tid, "task": "answer_map", "fixed_map": True,
                      "fixed_maps_sha256": self.run_identity["fixed_maps_sha256"]}
            if existing[1] == "pending":
                self.db.execute("UPDATE tasks SET state='done',result=? WHERE id=?", (canonical(result), tid))
            elif json.loads(existing[2]) != result:
                raise ValueError("fixed map conflicts with existing result")
        if self.db.execute("SELECT COUNT(*) FROM tasks WHERE kind='answer_map' AND state!='done'").fetchone()[0]:
            raise ValueError("fixed map file must cover every map task")
        self.db.commit()

    def result(self, tid):
        row = self.db.execute("SELECT result FROM tasks WHERE id=?", (tid,)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def ready(self, *, j1=False):
        while True:
            if self.admission_check is not None:
                status = self.admission_check()
                if not status.get("safe_to_admit") or not status.get("quota_verified"):
                    self.admission_stop = status
                    return
            row = self.db.execute("""SELECT t.id,t.record FROM tasks t
                WHERE t.state='pending' AND (t.kind='fulfilment')=?
                ORDER BY t.priority,t.rowid LIMIT 1""", (int(j1),)).fetchone()
            if not row:
                return
            tid, raw = row
            record = json.loads(raw)
            if record["task"] == "source_dependency":
                parent_record = json.loads(self.db.execute("SELECT record FROM tasks WHERE id=?", (record["map_task_id"],)).fetchone()[0])
                parent = self.result(record["map_task_id"])
                mapped = (parent or {}).get("parsed_output", {})
                quarantined = self.db.execute("SELECT 1 FROM quarantined_maps WHERE id=?", (record["map_task_id"],)).fetchone()
                if quarantined or not parent or not parent.get("ok") or mapped.get("eligibility") != "eligible":
                    result = {"ok": False, "status": "map_quarantined" if quarantined else mapped.get("eligibility", "map_failed"), "importance": None,
                              "judge_task_id": tid, "task": "source_dependency", "map_task_id": record["map_task_id"]}
                    self.db.execute("UPDATE tasks SET state='blocked',result=? WHERE id=?", (canonical(result), tid))
                    self.db.commit()
                    continue
                item = v4.materialize_source(record, parent_record, mapped)
            elif record.get("protocol") == v4.V3_COMPARISON_PROTOCOL:
                item = v4.comparison_from_record(record)
            elif record.get("protocol") == v4.PROTOCOL:
                item = v4.item_from_record(record)
            else:
                item = v3.item_from_record(record)
            item["logical_id"] = tid
            self.db.execute("UPDATE tasks SET state='running',writer_id=?,job_id=? WHERE id=?",
                            (self.writer_id, os.environ.get("SLURM_JOB_ID"), tid))
            self.db.commit()
            yield item

    async def one(self, item, *, client, fake):
        tid = item["logical_id"]
        identity = ClaimIdentity(task_id=item["base"]["judge_task_id"], model_id=self.config["model_id"],
                                 model_revision=self.config["model_revision"],
                                 protocol=f"{item['base']['protocol']}:{self.execution_sha256}",
                                 request_sha256=_request_sha256(item))
        claimed = self.ledger.claim(identity, owner_id=self.writer_id)
        if claimed.status != "owned":
            raise RuntimeError(f"unexpected task ownership: {claimed.status}")
        claim = claimed.claim
        self.db.execute("UPDATE tasks SET claim=?,identity=? WHERE id=?",
                        (canonical(asdict(claim)), canonical(asdict(identity)), tid))
        self.db.commit()
        self.ledger.transition(claim, state="running")
        item["base"]["transaction_id"] = claim.token
        result_written = False
        try:
            input_ref = self.writer.append("judge_inputs", {
                "judge_task_id": item["base"]["judge_task_id"], "logical_id": tid,
                "rendered_prompt": item["prompt"], "schema": item["schema"],
                "record": item.get("record"), "request_sha256": _request_sha256(item),
                "execution_sha256": self.execution_sha256}, transaction_id=claim.token)
            result = await _execute_one(item, client=client, fake=False)
            result = {k: v for k, v in result.items() if k != "base"}
            result.update(judge_task_id=item["base"]["judge_task_id"], logical_id=tid,
                          task=item["base"]["task"], execution_sha256=self.execution_sha256,
                          input_record_id=input_ref["record_id"])
            ref = self.writer.append("judgments", result, transaction_id=claim.token)
            result_written = True
            refs = [input_ref, ref]
            self.db.execute("UPDATE tasks SET state='saved',result=?,refs=? WHERE id=?",
                            (canonical(result), canonical(refs), tid))
            self.db.execute("UPDATE tasks SET state='pending' WHERE parent=? AND state='waiting'", (tid,))
            if result.get("parsed_output", {}).get("status") == "map_issue":
                self.db.execute("INSERT OR IGNORE INTO quarantined_maps SELECT parent FROM tasks WHERE id=?", (tid,))
            self.db.commit()
            self.ledger.transition(claim, state="result_saved", record_references=refs)
            self.pending_seals += 1
            if self.pending_seals >= 128:
                self.checkpoint()
            return result
        except BaseException:
            state = self.db.execute("SELECT state FROM tasks WHERE id=?", (tid,)).fetchone()[0]
            if state == "running" and not result_written:
                self.ledger.transition(claim, state="checkpointed", detail={"reason": "interrupted_before_saved_result"})
                self.db.execute("UPDATE tasks SET state='pending',claim=NULL WHERE id=?", (tid,))
                self.db.commit()
            raise

    def checkpoint(self):
        self.writer.seal()
        cursor = self.db.execute("SELECT id,claim,refs,result FROM tasks WHERE state='saved'")
        for tid, claim, refs, result in cursor.fetchall():
            references = json.loads(refs)
            if not all(verify_record_reference(self.output, r) for r in references):
                raise ValueError("refusing completion without sealed result")
            state = "completed" if json.loads(result).get("ok") else "terminal_failed"
            self.ledger.transition(LedgerClaim(**json.loads(claim)), state=state, record_references=references)
            self.db.execute("UPDATE tasks SET state='done' WHERE id=?", (tid,))
        self.db.commit()
        self.pending_seals = 0

    async def execute(self, client, *, budget=None):
        try:
            # Separate measured phases so J1 cannot inflate or subsidize the SI cost ratio.
            for j1 in (False, True):
                start = time.monotonic()
                try:
                    while budget is None or budget.can_start():
                        pending = self.db.execute("SELECT 1 FROM tasks WHERE state='pending' AND (kind='fulfilment')=? LIMIT 1",
                                                  (int(j1),)).fetchone()
                        if not pending:
                            break
                        executed = 0
                        async for _ in _iter_execute(self.ready(j1=j1), client=client,
                                maximum_concurrency=self.config["concurrency"], fake=False,
                                budget=budget, execute_one=self.one):
                            executed += 1
                        if not executed:
                            break
                finally:
                    elapsed = time.monotonic() - start
                    self.db.execute("INSERT INTO timing VALUES (?,?,?)", ("j1" if j1 else "si", elapsed, self.config["gpu_count"]))
                    self.db.commit()
                    self.writer.append("diagnostics", {"kind": "coordinator_timing", "seconds": elapsed,
                        "phase": "j1" if j1 else "si", "gpu_count": self.config["gpu_count"],
                        "execution_sha256": self.execution_sha256, "budget": budget.record() if budget else None},
                        transaction_id=self.writer_id)
        finally:
            # Peers cancelled before their coroutine began have no owned claim.
            self.db.execute("UPDATE tasks SET state='pending' WHERE state='running' AND claim IS NULL AND writer_id=?",
                            (self.writer_id,))
            self.db.commit()
            self.checkpoint()

    def report(self):
        directory = self.output / "reports" / self.writer_id
        directory.mkdir()
        counts, strata, time_totals = Counter(), {}, Counter()
        self.db.execute("CREATE TABLE IF NOT EXISTS quarantined_maps (id TEXT PRIMARY KEY)")
        for parent, result in self.db.execute("SELECT parent,result FROM tasks WHERE result IS NOT NULL"):
            row = json.loads(result)
            if row.get("parsed_output", {}).get("status") == "map_issue":
                self.db.execute("INSERT OR IGNORE INTO quarantined_maps VALUES (?)", (parent,))
            if not row.get("fixed_map"):
                task = row.get("task", "unknown")
                time_totals[task + "_request_seconds"] += row.get("duration_seconds", 0)
                time_totals[task + "_requests"] += int("duration_seconds" in row)
                attempts = row.get("validation_attempts") or [{"usage": row.get("usage", {})}]
                for attempt in attempts:
                    usage = attempt.get("usage", {})
                    for key in ("prompt_tokens", "completion_tokens"):
                        time_totals[task + "_" + key] += usage.get(key, 0) or 0

        def join(bundle, cell, *, sensitivity=False):
            output, grades = [], {}
            map_id = bundle.get("map_task_id")
            for source in bundle.get("sources", []):
                tid = source.get("dependency_id", source.get("judge_task_id"))
                result = self.result(tid) if tid else None
                parsed = (result or {}).get("parsed_output", {})
                status = (parsed.get("status", "scored") if result and result.get("ok") else
                          result.get("status", "inference_failed") if result else source.get("status", "missing"))
                if self.db.execute("SELECT 1 FROM quarantined_maps WHERE id=?", (map_id,)).fetchone():
                    status = "map_quarantined"
                grade = parsed.get("importance") if status == "scored" else None
                grades[source["url"]] = grade
                frozen = self.db.execute("SELECT record FROM tasks WHERE id=?", (tid,)).fetchone() if tid else None
                original = json.loads(frozen[0]) if frozen else {}
                original = original.get("legacy_record", original)
                output.append({**source, "status": status, "importance": grade,
                               "judge_task_id": (result or {}).get("judge_task_id"),
                               "raw_output_sha256": hashlib.sha256(result["raw_output"].encode()).hexdigest()
                               if result and isinstance(result.get("raw_output"), str) else None,
                               "source_sha256": _digest({"title": original.get("source_title"),
                                                         "text": original.get("source_text")}),
                               "result_logical_id": tid})
                counts[("sensitivity_sources_" if sensitivity else "sources_") + status] += 1
                if not sensitivity and status not in ("no_substantive_content", "global_absence_only", "unassessable_input"):
                    counts["model_declared_eligible_sources"] += 1
            metrics = v4.cell_metrics(grades, generator_list=cell.get("generator_ranking", []),
                                      presented=cell.get("presented", []))
            j1 = self.result(bundle.get("j1_task_id"))
            map_result = self.result(map_id)
            return {"sources": output, "grades": grades, "metrics": metrics,
                    "map_eligibility": (map_result or {}).get("parsed_output", {}).get("eligibility"),
                    "map_sha256": (map_result or {}).get("parsed_output", {}).get("map_sha256"),
                    "j1": (j1 or {}).get("parsed_output", {}).get("request_fulfillment")}

        with gzip.open(directory / "cells.jsonl.gz", "wt", encoding="utf-8") as stream:
            for cell in rows(self.inputs / "cells.jsonl.gz"):
                if cell.get("status") == "ok":
                    joined = join(cell, cell)
                    cell.update(joined)
                    counts["cells_complete"] += int(joined["metrics"]["complete"])
                    counts["cells_provenance_verified"] += int(cell.get("provenance", {}).get("status") == "verified")
                    if "stored_answer_sensitivity" in cell:
                        cell["stored_answer_sensitivity"].update(join(cell["stored_answer_sensitivity"], cell, sensitivity=True))
                    for field in ("model", "method", "engine", "condition"):
                        key = field + ":" + str(cell.get(field))
                        group = strata.setdefault(key, Counter())
                        group["cells"] += 1
                        group["complete"] += int(joined["metrics"]["complete"])
                counts["cells"] += 1
                stream.write(canonical(cell) + "\n")
        states = dict(self.db.execute("SELECT state,COUNT(*) FROM tasks GROUP BY state"))
        self.db.commit()
        timing = {phase: {"node_hours": seconds / 3600, "gpu_hours": gpu_seconds / 3600}
                  for phase, seconds, gpu_seconds in self.db.execute(
                      "SELECT phase,SUM(seconds),SUM(seconds*gpu_count) FROM timing GROUP BY phase")}
        failures = sum(1 for (raw,) in self.db.execute("SELECT result FROM tasks WHERE state='done'")
                       if raw and not json.loads(raw).get("ok"))
        summary = {"scientific_result": False, "protocol": self.manifest["protocol"],
                   "execution_sha256": self.execution_sha256, "states": states,
                   "configuration": self.config, "manifest_sha256": self.run_identity["manifest_sha256"],
                   "fixed_maps_sha256": self.run_identity["fixed_maps_sha256"],
                   "preprocessing": self.manifest.get("preprocessing", v4.PREPROCESSING_VERSION),
                   "max_tokens": self.manifest.get("max_tokens"),
                   "counts": dict(counts), "strata": strata, "request_totals": dict(time_totals),
                   "quarantined_maps": self.db.execute("SELECT COUNT(*) FROM quarantined_maps").fetchone()[0],
                   "timing": timing, "inference_failures": failures, "warm_gpu_hour_ratio_ceiling": 3.0,
                   "cost_note": "request seconds overlap; do not convert their sum to node-hours",
                   "metrics_version": v4.METRICS_VERSION,
                   "status": "incomplete" if any(states.get(s, 0) for s in ("pending", "running", "waiting")) else
                             "finished_with_failures" if failures else "finished",
                   "semantic_acceptance": "not_established"}
        summary["admission_stop"] = self.admission_stop
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (directory / "maps.jsonl").open("w") as stream:
            for tid, raw in self.db.execute("SELECT id,result FROM tasks WHERE kind='answer_map' AND result IS NOT NULL"):
                result = json.loads(raw)
                if result.get("ok"):
                    stream.write(canonical({"judge_task_id": tid, "raw_output": result["raw_output"],
                                            "execution_sha256": self.execution_sha256}) + "\n")
        return summary

    def close(self):
        self.writer.close()
        self.db.close()
        self.lock.close()


async def run(args):
    config = load_config(args.config)
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("scientific inference requires a clean committed checkout")
    config["code_revision"] = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    budget = AllocationBudget.from_environment(require=True)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("GEODML_INFERENCE_AUTH_REQUIRED") != "1":
        raise ValueError("use the approved cluster serving boundary with authenticated inference")
    if not Path(args.output).resolve().is_relative_to(Path(args.workspace).resolve()):
        raise ValueError("output must belong to the quota-checked workspace")
    health = check_storage(args.workspace, args.quota_evidence)
    if not health["safe_to_admit"] or not health["quota_verified"]:
        raise ValueError("storage admission failed: " + ", ".join(health["reasons"]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    template = tokenizer.get_chat_template()
    if hashlib.sha256(template.encode()).hexdigest() != config["chat_template_sha256"]:
        raise ValueError("local chat template does not match the execution fingerprint")
    coordinator = Coordinator(args.inputs, args.output, config, recovery_evidence=args.recovery_evidence,
                              fixed_maps=args.fixed_maps,
                              admission_check=lambda: check_storage(args.workspace, args.quota_evidence, args.output))
    try:
        async with VllmChatClient(base_url=args.base_url, api_key=None, server_model_name=config["model_id"],
                timeout_seconds=config["request_timeout"], maximum_attempts=3,
                audit_callback=coordinator.writer.audit_callback,
                chat_template_kwargs=config["chat_template_kwargs"]) as client:
            await coordinator.execute(ContextCheckedClient(client, tokenizer, config), budget=budget)
        report = coordinator.report()
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "finished" else 2
    finally:
        coordinator.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--quota-evidence", type=Path, required=True)
    parser.add_argument("--recovery-evidence", type=Path)
    parser.add_argument("--fixed-maps", type=Path, help="frozen map JSONL for Stage B-only repeats; new output required")
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

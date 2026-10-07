"""GPU worker slot logic with fake tasks: one device per statistics task, all devices for an embedding task, no overlap."""

import json
import sys
import time

from analysis.fullrun import ledger as L

RECORD = ("import os, json, time, pathlib, sys; p = pathlib.Path(sys.argv[1]); "
          "start = time.time(); time.sleep(float(sys.argv[2])); "
          "p.write_text(json.dumps({{'dev': os.environ.get('CUDA_VISIBLE_DEVICES'), 'start': start, 'end': time.time()}}))")
# (task arguments are str.format templates: literal braces are doubled)


def task(tid, out, seconds, gpus, deps=()):
    return L.Task(tid, "gpu", [sys.executable, "-c", RECORD, str(out / f"{tid}.json"), str(seconds)], list(deps),
                  cores=1, gpu=True, gpus=gpus)


def test_slots_parallel_one_gpu_tasks_and_exclusive_embeddings(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    tasks = [task(f"s{i}", out, 0.4, 1) for i in range(6)] + [task("embed", out, 0.3, 0, deps=["s0"])]
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks(tasks)
    rc = L.work(led, job="g", cores=16, end_epoch=time.time() + 3600, margin_minutes=0, stages=None, gpu=True,
                min_start_minutes=0, poll_seconds=0.02, devices=4)
    assert rc == 0 and all(led.done(t.id) for t in tasks)
    rec = {t.id: json.loads((out / f"{t.id}.json").read_text()) for t in tasks}
    assert rec["embed"]["dev"] == "0,1,2,3"
    assert all(len(rec[f"s{i}"]["dev"].split(",")) == 1 for i in range(6))

    def overlap(a, b):
        return rec[a]["start"] < rec[b]["end"] and rec[b]["start"] < rec[a]["end"]

    ids = list(rec)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if overlap(a, b):
                assert set(rec[a]["dev"].split(",")).isdisjoint(rec[b]["dev"].split(","))   # never the same device at once
    assert any(overlap(f"s{i}", f"s{j}") for i in range(6) for j in range(i + 1, 6))       # 1-GPU tasks run in parallel


def test_cpu_worker_ignores_gpu_tasks(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    led = L.Ledger(tmp_path / "ledger")
    led.write_tasks([task("s0", out, 0.0, 1)])
    assert L.work(led, job="c", cores=4, end_epoch=time.time() + 3600, margin_minutes=0, stages=None, gpu=False,
                  min_start_minutes=0, poll_seconds=0.02) == 0
    assert not led.done("s0")

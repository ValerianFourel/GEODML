#!/bin/bash
python3 - <<'PY'
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile

root = Path("/hkfs/work/workspace/scratch/uhh_bbf7367-geodml-qwen/reviews/gemma-map-retry-job5175156-spans-v1")
run = root / "run/attempts/job5175229/trial/v4-pass1"
database = run / "control/index.sqlite"

with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
    db.execute("PRAGMA query_only=ON")
    rows = db.execute(
        "SELECT id,kind,state,record,result FROM tasks ORDER BY kind,id"
    ).fetchall()

tasks = [
    dict(id=i, kind=k, state=s,
         input=json.loads(rec) if rec else {},
         result=json.loads(res) if res else {})
    for i, k, s, rec, res in rows
]
summaries = {
    str(p.relative_to(run)): json.loads(p.read_text())
    for p in sorted((run / "reports").glob("*/summary.json"))
}
output = Path(tempfile.mkdtemp(prefix="diagnostic-job5175229-", dir=root))
report = output / "diagnostic.json"
report.write_text(json.dumps(
    {"run": str(run), "summaries": summaries, "tasks": tasks},
    indent=2, ensure_ascii=False
) + "\n")

counts = Counter(
    (t["kind"], t["state"], str(t["result"].get("ok")))
    for t in tasks
)
print("TASK_COUNTS")
for key, count in sorted(counts.items()):
    print(*key, count)

for task in tasks:
    if task["kind"] == "answer_map":
        print("\nANSWER_MAP_INPUT_AND_RESULT")
        print(json.dumps(task, indent=2, ensure_ascii=False))
    elif task["kind"] == "source_dependency":
        result = task["result"]
        print("\nSOURCE_RESULT", task["id"], task["state"])
        print("TITLE", task["input"].get("source_title"))
        print(json.dumps(
            result.get("parsed_output", result),
            indent=2, ensure_ascii=False
        ))

print("\nFULL_REPORT", report)
PY

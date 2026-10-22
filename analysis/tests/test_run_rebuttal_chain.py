"""ARR rebuttal batch job (analysis/www_robustness/run_rebuttal.sbatch): chain guard, side check and stop marker.

The job runs from a throwaway git checkout holding the real script and score_pairs.py, with a stub run_horeka.sh
(exit code from STUB_RC), a stub srun that runs its command locally, and a workspace whose python is this one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def scored(run: Path, shard: int, score: float) -> None:
    d = run / "results" / "scores"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prompt_id": ["p0", "p1"], "row_id": [1, 2], "purpose": "validation", "score": [score, 1.0],
                  "fake": False}).to_parquet(d / f"shard-{shard:05d}.parquet", index=False)
    (d / f"shard-{shard:05d}.json").write_text(json.dumps({"pairs": 2, "seconds": 10.0, "worker": shard}))


class RebuttalChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        code = self.tmp / "code"
        (code / "analysis" / "www_robustness").mkdir(parents=True)
        (code / "analysis" / "__init__.py").write_text("")
        (code / "analysis" / "www_robustness" / "__init__.py").write_text("")
        for f in ("run_rebuttal.sbatch", "score_pairs.py"):
            shutil.copy(ROOT / "analysis" / "www_robustness" / f, code / "analysis" / "www_robustness" / f)
        (code / "analysis" / "www_robustness" / "run_horeka.sh").write_text(
            '#!/bin/bash\necho "stub runner"; exit "${STUB_RC:-0}"\n')
        git = ["git", "-C", str(code), "-c", "user.name=t", "-c", "user.email=t@example.org"]
        subprocess.run([*git, "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-q", "-m", "t"], check=True)
        self.code = code
        self.ws = self.tmp / "ws"
        (self.ws / "environment" / "qwen-runtime" / "bin").mkdir(parents=True)
        os.symlink(sys.executable, self.ws / "environment" / "qwen-runtime" / "bin" / "python")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        (self.bin / "srun").write_text('#!/bin/bash\nwhile [ $# -gt 0 ] && [ "${1#--}" != "$1" ]; do shift; done\n'
                                       'exec "$@"\n')
        (self.bin / "srun").chmod(0o755)
        self.run_dir = self.ws / "runs" / "www-robustness" / "r1"
        (self.run_dir / "results").mkdir(parents=True)
        pd.DataFrame({"prompt_id": ["p0"], "row_id": [1], "purpose": ["validation"], "shard": [0]}).to_parquet(
            self.run_dir / "results" / "shards.parquet", index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def job(self, rc=0, **extra):
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "GEODML_WORKSPACE": str(self.ws),
               "CODE": str(self.code), "BUNDLE": str(self.tmp / "no-bundle"), "RUN_ID": "r1", "SLURM_JOB_ID": "77",
               "STUB_RC": str(rc), **extra}
        return subprocess.run(["bash", str(self.code / "analysis" / "www_robustness" / "run_rebuttal.sbatch")],
                              env=env, capture_output=True, text=True)

    @property
    def stop_file(self) -> Path:
        return self.run_dir / "CHAIN_STOP"

    def test_checkpoint_and_done_leave_the_chain_open(self):
        for rc in (0, 4):
            p = self.job(rc)  # no scored shards yet: the side check is skipped
            self.assertEqual(p.returncode, rc, p.stderr)
            self.assertIn("sortcheck: skipped", p.stdout)
            self.assertFalse(self.stop_file.exists())

    def test_runner_failure_stops_the_chain_and_the_next_job_runs_nothing(self):
        p = self.job(1, SORTCHECK_SHARDS="none")
        self.assertEqual(p.returncode, 1)
        self.assertIn("run_horeka.sh exit 1", self.stop_file.read_text())
        p = self.job(4)
        self.assertEqual(p.returncode, 0)
        self.assertIn("stopped by an earlier job", p.stdout)
        self.assertNotIn("stub runner", p.stdout)

    def test_side_check_failure_stops_before_scoring(self):
        scored(self.run_dir, 0, 0.5)
        scored(self.run_dir, 1, 0.5)
        p = self.job(0)  # the rescoring fails (no bundle, no model), so compare finds nothing to match
        self.assertEqual(p.returncode, 3)
        self.assertIn("sortcheck failed", self.stop_file.read_text())
        self.assertNotIn("stub runner", p.stdout)

    def test_side_check_pass_then_scoring_and_no_recheck(self):
        side = self.run_dir.parent / "r1-sortcheck"
        for s in (0, 1):
            scored(self.run_dir, s, 0.5)
            scored(side, s, 0.5 + 5e-5)  # already rescored, within 1e-4
        p = self.job(4)
        self.assertEqual(p.returncode, 4, p.stderr)
        self.assertTrue(json.loads((side / "results" / "compare.json").read_text())["passed"])
        self.assertIn("stub runner", p.stdout)
        self.assertIn("sortcheck: passed before", self.job(4).stdout)
        self.assertFalse(self.stop_file.exists())


class CompareTest(unittest.TestCase):
    def test_pass_and_fail(self):
        sys.path.insert(0, str(ROOT))
        from analysis.www_robustness import score_pairs as sp
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            for s in (0, 1):
                scored(old, s, 0.5)
                scored(new, s, 0.5 + 2e-5)
            args = ["compare", "--run-dir", str(new), "--against", str(old), "--only-shards", "0,1"]
            self.assertEqual(sp.main(args), 0)
            scored(new, 1, 0.6)
            self.assertEqual(sp.main(args), 3)
            out = json.loads((new / "results" / "compare.json").read_text())
            self.assertEqual([r["passed"] for r in out["shards"]], [True, False])


if __name__ == "__main__":
    unittest.main()

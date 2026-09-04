from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "analysis/scripts/prepare_acl_arr_pilot_inputs.py"


class PrepareAclArrPilotInputsTests(unittest.TestCase):
    def test_selects_only_prompts_with_complete_cached_document_sets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audit = root / "audit"
            output = root / "output"
            html_root = (
                root
                / "data/runs/searxng_test_serp20_top10/phase2/html_cache"
            )
            audit.mkdir()
            html_root.mkdir(parents=True)

            prompts = []
            axis = []
            serp = []
            for keyword_index, keyword in enumerate(("alpha", "beta")):
                candidate_id = f"candidate-{keyword_index}"
                prompts.append(
                    {
                        "candidate_id": candidate_id,
                        "keyword": keyword,
                        "question": f"Question about {keyword}?",
                    }
                )
                axis.append(
                    {
                        "candidate_id": candidate_id,
                        "axis_1_rank": keyword_index,
                        "consensus_axis_1_z": float(keyword_index),
                    }
                )
                for position in range(1, 13):
                    url = f"https://example.com/{keyword}/{position}"
                    serp.append(
                        {
                            "keyword": keyword,
                            "position": position,
                            "title": f"Document {position}",
                            "url": url,
                            "snippet": "snippet",
                        }
                    )
                    filename = hashlib.sha256(url.encode()).hexdigest()[:16] + ".html"
                    (html_root / filename).write_text(
                        "<html><body><p>" + (f"{keyword} evidence " * 20) + "</p></body></html>",
                        encoding="utf-8",
                    )

            for name, rows in (
                ("compliant-candidates.jsonl", prompts),
                ("final-axis-map.jsonl", axis),
            ):
                (audit / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            serp_path = root / "serp.jsonl"
            serp_path.write_text(
                "".join(json.dumps(row) + "\n" for row in serp),
                encoding="utf-8",
            )
            test_modules = root / "test-modules"
            test_modules.mkdir()
            (test_modules / "dotenv.py").write_text(
                "def load_dotenv(*args, **kwargs):\n    return False\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(test_modules), str(REPOSITORY_ROOT / "analysis"))
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--audit-root",
                    str(audit),
                    "--serp",
                    str(serp_path),
                    "--data-root",
                    str(root),
                    "--pilot-size",
                    "2",
                    "--minimum-documents",
                    "11",
                    "--output-dir",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PILOT_PROMPTS=2", result.stdout)
            self.assertEqual(
                len((output / "pilot-prompts.jsonl").read_text().splitlines()), 2
            )
            self.assertEqual(
                len((output / "pilot-axis.jsonl").read_text().splitlines()), 2
            )
            manifest = json.loads(
                (output / "pilot-input-manifest.json").read_text()
            )
            self.assertEqual(manifest["selected_keyword_count"], 2)
            self.assertEqual(manifest["selected_serp_row_count"], 24)
            self.assertEqual(manifest["selected_page_count"], 24)


if __name__ == "__main__":
    unittest.main()

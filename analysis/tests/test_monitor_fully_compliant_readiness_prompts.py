from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.scripts.monitor_fully_compliant_readiness_prompts import (
    audit_new_verified_rounds,
    verified_rounds,
)
from analysis.tests.test_audit_fully_compliant_readiness_prompts import (
    _convert_to_verified_round,
    _fixture,
)


class LiveFullyCompliantMonitorTests(unittest.TestCase):
    def test_audits_each_verified_round_exactly_once(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline = Path(directory) / "pipeline"
            round_root = pipeline / "round-03"
            _fixture(round_root)
            _convert_to_verified_round(round_root)

            self.assertEqual(verified_rounds(pipeline), [round_root.resolve()])
            payloads = audit_new_verified_rounds(pipeline)

            self.assertEqual(len(payloads), 1)
            self.assertTrue(payloads[0]["audit_passed"])
            self.assertEqual(payloads[0]["fully_compliant_prompts"], 2)
            self.assertEqual(payloads[0]["ready_to_export"], 2)
            self.assertEqual(payloads[0]["missing_from_30330"], 30328)
            self.assertEqual(
                audit_new_verified_rounds(
                    pipeline,
                    already_audited=[str(round_root.resolve())],
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()

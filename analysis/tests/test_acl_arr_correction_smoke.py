import unittest
from analysis.scripts.run_acl_arr_correction_smoke import select_failures, correction_prompt


class CorrectionSmokeTests(unittest.TestCase):
    def test_selection_is_bounded_deterministic_and_spans_conditions(self):
        rows = [{'task_id': str(i), 'condition': ('natural', 'ablated', 'shuffled')[i % 3],
                 'error': 'invalid', 'raw_output': 'bad'} for i in range(73)]
        selected = select_failures(rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected, select_failures(list(reversed(rows))))
        self.assertEqual({r['condition'] for r in selected}, {'natural', 'ablated', 'shuffled'})

    def test_correction_keeps_original_evidence_and_failed_response(self):
        prompt = correction_prompt('ORIGINAL EVIDENCE', {'raw_output': 'FAILED ANSWER', 'error': 'unknown ID'})
        self.assertTrue(prompt.startswith('ORIGINAL EVIDENCE'))
        self.assertIn('FAILED ANSWER', prompt)
        self.assertIn('unknown ID', prompt)

    def test_duplicate_ids_refused(self):
        with self.assertRaises(ValueError):
            select_failures([{'task_id': 'x'}] * 10)

import json
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis.scripts.repair_acl_arr_pilot_citation_order import repair_output, audit_and_repair


class CitationOrderRepairTests(unittest.TestCase):
    def test_only_order_changes_and_answer_is_preserved(self):
        answer = 'Steps [C008] [C001] [C007]. More [C008] [C003].'
        raw = json.dumps({'answer': answer, 'cited_document_ids': ['C001', 'C003', 'C007', 'C008']})
        repaired = repair_output(raw, ['C001', 'C003', 'C007', 'C008'])
        self.assertEqual(repaired['answer'], answer)
        self.assertEqual(repaired['cited_document_ids'], ['C008', 'C001', 'C007', 'C003'])

    def test_no_other_invalid_output_is_repaired(self):
        for value in (
            {'answer': 'Use [C001].', 'cited_document_ids': ['C002']},
            {'answer': 'Use [C002] [C001].', 'cited_document_ids': ['C001', 'C002', 'C001']},
            {'answer': 'Use [C003] [C001].', 'cited_document_ids': ['C001', 'C003']},
            {'answer': 'No citations.', 'cited_document_ids': []},
            {'answer': 'Use [C001].', 'cited_document_ids': ['C001']},
            {'answer': 'Use [C002] [C001].', 'cited_document_ids': ['C001', 'C002'], 'extra': 1},
        ):
            with self.subTest(value=value):
                self.assertIsNone(repair_output(json.dumps(value), ['C001', 'C002']))
        self.assertIsNone(repair_output('{"answer":', ['C001']))

    def test_isolated_artifacts_preserve_sources_and_refuse_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            raw = json.dumps({'answer': 'Use [C002] [C001].', 'cited_document_ids': ['C001', 'C002']})
            base = {'task_id': 'one', 'pipeline': 'answer', 'input_document_ids': ['C001', 'C002']}
            row = dict(base, raw_output=raw, raw_output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                       scientific_result=False, eligible_for_analysis=False, error='citation order')
            (source / 'outcomes.jsonl').write_text('')
            for name in ('failures.jsonl', 'attempts.jsonl'):
                (source / name).write_text(json.dumps(row) + '\n')
            metadata = dict(tasks_sha256='t', model_id='m', model_revision='r', plan_source_git_commit='c')
            manifest = dict(metadata, format_version='acl-arr-pilot-answer-step-v1',
                            scientific_result=False, eligible_for_analysis=False,
                            status='complete_with_failures', source_artifacts_sha256={},
                            outcomes_sha256=hashlib.sha256(b'').hexdigest(),
                            tasks_path=str(root / 'plan/tasks/model/answer.jsonl'),
                            rerank_recovery_results=str(root / 'recovery'), completed_count=0, task_count=1)
            (source / 'answer_manifest.json').write_text(json.dumps(manifest))
            before = {p: p.read_bytes() for p in source.iterdir()}
            with patch('analysis.scripts.repair_acl_arr_pilot_citation_order.load_answers',
                       return_value=([{'base': base}], {}, metadata)):
                result = audit_and_repair(source, root / 'repaired')
                self.assertEqual(result['repaired_count'], 1)
                self.assertFalse(result['eligible_for_analysis'])
                self.assertEqual(result['inference_requests'], 0)
                self.assertFalse((root / 'repaired/outcomes.jsonl').exists())
                with self.assertRaises(ValueError):
                    audit_and_repair(source, root / 'repaired')
            self.assertEqual(before, {p: p.read_bytes() for p in source.iterdir()})


if __name__ == '__main__':
    unittest.main()

import json
import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analysis.scripts.repair_acl_arr_pilot_citation_order import repair_output, audit_and_repair


class CitationOrderRepairTests(unittest.TestCase):
    def test_formatting_only_preserves_answer_and_rejects_prose_and_bad_ids(self):
        answer = 'Use [ C002 , C001 ].'
        raw = '```json\n' + json.dumps({'answer': answer, 'cited_document_ids': ['C001', 'C002']}) + '\n```'
        self.assertEqual(repair_output(raw, ['C001', 'C002'], grouped=True, formatting=True),
                         {'answer': answer, 'cited_document_ids': ['C002', 'C001']})
        self.assertIsNone(repair_output('Here is the answer: ' + raw, ['C001', 'C002'], grouped=True, formatting=True))
        self.assertIsNone(repair_output(raw, ['C001'], grouped=True, formatting=True))

    def test_grouped_citations_preserve_answer_and_require_opt_in(self):
        answer = 'Evidence [C002]. More [C001, C003, C002].'
        raw = json.dumps({'answer': answer, 'cited_document_ids': ['C001', 'C002', 'C003']})
        self.assertIsNone(repair_output(raw, ['C001', 'C002', 'C003']))
        result = repair_output(raw, ['C001', 'C002', 'C003'], grouped=True)
        self.assertEqual(result, {'answer': answer, 'cited_document_ids': ['C002', 'C001', 'C003']})

    def test_grouped_citations_do_not_guess_or_drop_ids(self):
        for answer, declared in (
            ('Evidence [C001, C003].', ['C001', 'C003']),
            ('Evidence [C001, C002].', ['C001']),
            ('Evidence [C001, C002].', ['C001', 'C002', 'C002']),
            ('Evidence [C001, C002] [The CMO].', ['C001', 'C002']),
            ('Evidence [C001, C002,].', ['C001', 'C002']),
            ('Evidence [[C001, C002]].', ['C001', 'C002']),
            ('Evidence [C001; C002].', ['C001', 'C002']),
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(repair_output(json.dumps({'answer': answer,
                    'cited_document_ids': declared}), ['C001', 'C002'], grouped=True))

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
            group_raw = json.dumps({'answer': 'Use [C002, C001].', 'cited_document_ids': ['C001', 'C002']})
            group_base = dict(base, task_id='two')
            group_row = dict(row, **group_base)
            group_row.update(raw_output=group_raw, raw_output_sha256=hashlib.sha256(group_raw.encode()).hexdigest())
            (source / 'outcomes.jsonl').write_text('')
            for name in ('failures.jsonl', 'attempts.jsonl'):
                (source / name).write_text(json.dumps(row) + '\n' + json.dumps(group_row) + '\n')
            metadata = dict(tasks_sha256='t', model_id='m', model_revision='r', plan_source_git_commit='c')
            manifest = dict(metadata, format_version='acl-arr-pilot-answer-step-v1',
                            scientific_result=False, eligible_for_analysis=False,
                            status='complete_with_failures', source_artifacts_sha256={},
                            outcomes_sha256=hashlib.sha256(b'').hexdigest(),
                            tasks_path=str(root / 'plan/tasks/model/answer.jsonl'),
                            rerank_recovery_results=str(root / 'recovery'), completed_count=0, task_count=2)
            (source / 'answer_manifest.json').write_text(json.dumps(manifest))
            before = {p: p.read_bytes() for p in source.iterdir()}
            with patch('analysis.scripts.repair_acl_arr_pilot_citation_order.load_answers',
                       return_value=([{'base': base}, {'base': group_base}], {}, metadata)):
                result = audit_and_repair(source, root / 'repaired')
                self.assertEqual(result['repaired_count'], 1)
                self.assertFalse(result['eligible_for_analysis'])
                self.assertEqual(result['inference_requests'], 0)
                self.assertFalse((root / 'repaired/outcomes.jsonl').exists())
                with self.assertRaises(ValueError):
                    audit_and_repair(source, root / 'repaired')
                grouped = audit_and_repair(source, root / 'grouped', grouped=True)
                self.assertEqual(grouped['repaired_count'], 2)
                self.assertEqual(grouped['order_only_repaired_count'], 1)
                self.assertEqual(grouped['grouped_citation_repaired_count'], 1)
                self.assertEqual(grouped['unresolved_count'], 0)
                records = [json.loads(line) for line in (root / 'grouped/repaired_records.jsonl').read_text().splitlines()]
                self.assertEqual(records[1]['repaired_output']['answer'], 'Use [C002, C001].')
                self.assertTrue(all(r['eligible_for_analysis'] is False for r in records))
            self.assertEqual(before, {p: p.read_bytes() for p in source.iterdir()})


if __name__ == '__main__':
    unittest.main()

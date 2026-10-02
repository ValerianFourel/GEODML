"""Absolute acceptance uses frozen denominators and independent, bound evidence.

The owner boundary is evaluate()/CLI reading saved freezes/reports. These cases
protect against favorable missing-data filters, stale/unblinded references,
repeat/cost denominator errors and HTML injection; historical comparison tests
cover its separate paired API. No test-only production hook is introduced.
"""
import asyncio
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from analysis.scripts import compare_source_importance_runs as review
from analysis.scripts import prepare_si_v4_evaluation as packets
from analysis.scripts import run_source_importance_judge as runner
from analysis.tests.test_source_importance_v4 import CONFIG, Responses, freeze as freeze_tasks


def freeze(tmp_path):
    inputs, tasks = freeze_tasks(tmp_path)
    path = inputs / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['cells'] = len(list(runner.rows(inputs / 'cells.jsonl.gz')))
    path.write_text(json.dumps(manifest))
    return inputs, tasks


def write_rows(path, values):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt') as stream:
        for row in values:
            stream.write(json.dumps(row) + '\n')


def report(tmp_path, inputs, name, *, fixed_maps=None):
    controller = runner.Coordinator(inputs, tmp_path / name, {**CONFIG, 'repetition_id': name}, fixed_maps=fixed_maps)
    asyncio.run(controller.execute(Responses()))
    controller.report()
    path = controller.output / 'reports' / controller.writer_id
    controller.close()
    return path


@pytest.fixture
def evidence(tmp_path):
    inputs, _ = freeze(tmp_path)
    candidate = report(tmp_path, inputs, 'candidate')
    policy = json.loads(review.ABSOLUTE_DESIGN.read_text())
    policy.update(historical_development_cells=1, repeat_cells=1)
    design = tmp_path / 'design.json'
    design.write_text(json.dumps(policy))
    packet_root = tmp_path / 'packets'
    packets.prepare_packets(inputs, packet_root / 'grade')
    grade_records = []
    for packet in runner.rows(packet_root / 'grade/packets.jsonl'):
        for model in policy['reference_models']:
            grade_records.append({k: packet[k] for k in ('record_type', 'cell_id', 'url', 'source_sha256', 'masked_answer_sha256', 'packet_sha256')} |
                {'reviewer_model': model, 'frozen_before_grade_review': True, 'whole_source_checked': True, 'acceptable_grade_range': [4, 5]})
    reference_path = tmp_path / 'references.jsonl'
    write_rows(reference_path, grade_records)
    packets.prepare_packets(inputs, packet_root / 'map', phase='map', candidate=candidate)
    packets.prepare_packets(inputs, packet_root / 'support', phase='support', candidate=candidate, frozen_grades=reference_path)
    references = grade_records[:]
    for packet in runner.rows(packet_root / 'map/packets.jsonl'):
        for model in policy['reference_models']:
            references.append({k: packet[k] for k in ('record_type', 'cell_id', 'masked_answer_sha256', 'map_sha256', 'packet_sha256')} |
                {'reviewer_model': model, 'assessment': {'faithful': True, 'essential_omission': False, 'meaning_reversal': False, 'importance_roles_faithful': True}})
    for packet in runner.rows(packet_root / 'support/packets.jsonl'):
        for model in policy['reference_models']:
            references.append({k: packet[k] for k in ('record_type', 'cell_id', 'url', 'source_sha256', 'masked_answer_sha256', 'candidate_raw_output_sha256', 'packet_sha256')} |
                {'reviewer_model': model, 'finding_id': 'source', 'whole_source_checked': True,
                 'missed_support': 'none', 'role_consistency': 'consistent', 'decisive_error_categories': [],
                 'explanation': 'All mapped claims and source text reviewed.'})
        for finding in packet['body']['findings']:
            for model in policy['reference_models']:
                references.append({k: packet[k] for k in ('record_type', 'cell_id', 'url', 'source_sha256', 'masked_answer_sha256', 'candidate_raw_output_sha256', 'packet_sha256')} |
                    {'reviewer_model': model, 'finding_id': finding['finding_id'], 'judgment': 'supported', 'witness_assessment': 'sufficient'})
    write_rows(reference_path, references)
    return {'candidate': candidate, 'inputs': inputs, 'design': design, 'references': reference_path,
            'reference_packets': packet_root}


def test_absolute_counts_missing_sources_and_does_not_depend_on_baseline(evidence):
    result = review.evaluate(**evidence)
    assert result['counts']['frozen_sources'] == result['counts']['scored_sources'] == 1
    assert result['metrics']['reference_grade_range_agreement']['fraction'] == 1
    assert result['metrics']['resolved_reference_pair_support'] == {'numerator': 2, 'denominator': 2, 'fraction': 1}
    assert result['gates']['warm_compute']['status'] == 'unavailable'
    assert result['decision'] != 'Ready to scale v4'
    cells = list(runner.rows(evidence['candidate'] / 'cells.jsonl.gz'))
    cells[0]['sources'] = []
    write_rows(evidence['candidate'] / 'cells.jsonl.gz', cells)
    # Candidate output references are now absent too; retain independently frozen grades/maps.
    refs = [r for r in runner.rows(evidence['references']) if r['record_type'] != 'support']
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['counts']['frozen_sources'] == result['counts']['sources_missing'] == 1
    assert result['metrics']['eligible_source_completion']['fraction'] == 0
    assert result['metrics']['reference_grade_range_agreement']['fraction'] == 0
    assert result['subgroups']['model:qwen38']['status_missing'] == 1


def test_reference_conflicts_and_missing_packets_cannot_make_a_pass(evidence):
    refs = list(runner.rows(evidence['references']))
    next(r for r in refs if r['record_type'] == 'grade' and r['reviewer_model'] == 'gpt-6-sol')['acceptable_grade_range'] = [0, 0]
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['counts']['reference_disagreements'] == 1
    assert result['counts']['references_unresolved_or_missing'] == 1
    assert result['metrics']['reference_grade_range_agreement']['fraction'] is None
    assert result['gates']['reference_grade_range_agreement']['status'] == 'unavailable'
    result = review.evaluate(**{**evidence, 'reference_packets': None})
    assert result['gates']['input_integrity']['status'] == 'fail'
    assert result['maps']['faithful'] == 0


def test_witness_insufficiency_is_separate_from_wrong_full_source_relation(evidence):
    refs = [r for r in runner.rows(evidence['references']) if r.get('finding_id') != 'source']
    for row in refs:
        if row['record_type'] == 'support':
            row['witness_assessment'] = 'insufficient'
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['metrics']['resolved_reference_pair_support']['fraction'] == 0
    assert result['metrics']['whole_source_relation_agreement']['fraction'] == 1
    assert result['gates']['resolved_reference_pair_support']['status'] == 'fail'
    assert result['decision'] == 'Needs a specific further repair'


def test_missing_source_summary_review_is_visible_even_with_valid_finding_reviews(evidence):
    refs = [r for r in runner.rows(evidence['references']) if r.get('finding_id') != 'source']
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['support']['source_reviews_expected'] == 1
    assert result['support']['source_reviews_complete'] == 0
    assert result['gates']['whole_source_review']['status'] == 'unavailable'


@pytest.mark.parametrize('fault', ['role', 'map_issue', 'decisive_support', 'single_reviewer_concern'])
def test_confirmed_contract_errors_and_reference_disagreement_are_distinct(evidence, fault):
    refs = list(runner.rows(evidence['references']))
    summaries = [r for r in refs if r.get('finding_id') == 'source']
    for row in summaries:
        if fault == 'role':
            row['role_consistency'] = 'inconsistent'
        elif fault == 'map_issue':
            row['role_consistency'] = 'map_issue'
        elif fault == 'decisive_support':
            row['decisive_error_categories'] = ['qualification']
    if fault == 'single_reviewer_concern':
        summaries[0]['decisive_error_categories'] = ['entity']
        summaries[0]['missed_support'] = 'confirmed'
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    if fault in ('role', 'map_issue'):
        assert result['gates']['fixed_map_contract']['status'] == 'fail'
        assert result['decision'] == 'Needs a specific further repair'
    elif fault == 'decisive_support':
        assert result['gates']['decisive_support_errors']['status'] == 'fail'
        assert result['decision'] == 'Judge configuration needs reconsideration'
    else:
        assert result['gates']['decisive_support_errors']['status'] == 'pass'
        assert result['support']['source_review_disagreements'] == 1
        assert result['support']['confirmed_missed_support'] == 0


def test_zero_without_findings_still_receives_complete_source_review(evidence):
    from analysis.interpretability.pipeline import source_importance_v4 as v4
    cells = list(runner.rows(evidence['candidate'] / 'cells.jsonl.gz'))
    cell, source = cells[0], cells[0]['sources'][0]
    tasks = {t['judge_task_id']: t for t in runner.rows(evidence['inputs'] / 'tasks.jsonl.gz')}
    raw = json.dumps({'status': 'scored', 'findings': [], 'importance': 0, 'note': 'No support found.'})
    parsed = v4.materialize_source(tasks[source['dependency_id']], tasks[cell['map_task_id']],
                                  cell['map_result']['parsed_output'])['validator'](raw)
    source.update(importance=0, raw_output=raw, raw_output_sha256=hashlib.sha256(raw.encode()).hexdigest(), parsed_output=parsed)
    write_rows(evidence['candidate'] / 'cells.jsonl.gz', cells)
    refs = [r for r in runner.rows(evidence['references']) if r['record_type'] != 'support']
    write_rows(evidence['references'], refs)
    grade_path = evidence['references'].with_name('independent-grades.jsonl')
    write_rows(grade_path, [r for r in refs if r['record_type'] == 'grade'])
    target = evidence['reference_packets'] / 'support-zero'
    packets.prepare_packets(evidence['inputs'], target, phase='support', candidate=evidence['candidate'], frozen_grades=grade_path)
    packet = next(runner.rows(target / 'packets.jsonl'))
    for model in ('gpt-6-astra', 'gpt-6-sol'):
        refs.append({k: packet[k] for k in ('record_type', 'cell_id', 'url', 'source_sha256', 'masked_answer_sha256', 'candidate_raw_output_sha256', 'packet_sha256')} |
                    {'reviewer_model': model, 'finding_id': 'source', 'whole_source_checked': True,
                     'missed_support': 'confirmed', 'role_consistency': 'consistent', 'decisive_error_categories': [],
                     'explanation': 'The complete source supports the exporter claims.'})
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['support']['source_reviews_expected'] == result['support']['source_reviews_complete'] == 1
    assert result['support']['confirmed_missed_support'] == 1
    assert result['counts']['missed_positive_reference'] == 1
    assert result['metrics']['whole_source_relation_agreement']['denominator'] == 0


@pytest.mark.parametrize('fault', ['raw_missing', 'raw_hash', 'parsed_tamper'])
def test_retained_structural_gate_revalidates_actual_raw_output(evidence, fault):
    cells = list(runner.rows(evidence['candidate'] / 'cells.jsonl.gz'))
    source = cells[0]['sources'][0]
    if fault == 'raw_missing':
        source.pop('raw_output')
    elif fault == 'raw_hash':
        source['raw_output_sha256'] = '0' * 64
    else:
        source['parsed_output']['findings'] = []
    write_rows(evidence['candidate'] / 'cells.jsonl.gz', cells)
    result = review.evaluate(**{**evidence, 'references': None})
    assert result['metrics']['retained_structural_validity']['fraction'] == 0
    assert result['gates']['retained_structural_validity']['status'] == 'fail'
    assert result['sources'][0]['structural_validation_error']


@pytest.mark.parametrize('fault', ['source_hash', 'answer_hash', 'output_hash', 'blinding'])
def test_references_reject_stale_and_unblinded_evidence(evidence, fault):
    refs = list(runner.rows(evidence['references']))
    if fault == 'source_hash':
        refs[0]['source_sha256'] = '0' * 64
    elif fault == 'answer_hash':
        refs[0]['masked_answer_sha256'] = '0' * 64
    elif fault == 'output_hash':
        next(r for r in refs if r['record_type'] == 'support')['candidate_raw_output_sha256'] = '0' * 64
    else:
        path = evidence['reference_packets'] / 'grade/packets.jsonl'
        packet = next(runner.rows(path))
        old_hash = packet.pop('packet_sha256')
        packet['body']['candidate_grade'] = 5
        packet['packet_sha256'] = hashlib.sha256(runner.canonical(packet).encode()).hexdigest()
        for ref in refs:
            if ref['packet_sha256'] == old_hash:
                ref['packet_sha256'] = packet['packet_sha256']
        write_rows(path, [packet])
    write_rows(evidence['references'], refs)
    with pytest.raises(ValueError, match='hash|blinded'):
        review.evaluate(**evidence)


def test_both_repeat_modes_keep_missing_runs_and_all_zero_tops_visible(evidence, tmp_path):
    first = evidence['candidate']
    end_to_end = [first, report(tmp_path, evidence['inputs'], 'e2'), report(tmp_path, evidence['inputs'], 'e3')]
    fixed = [report(tmp_path, evidence['inputs'], 'f' + str(i), fixed_maps=first / 'maps.jsonl') for i in range(3)]
    result = review.evaluate(**evidence, repeats={'fixed_map': fixed, 'end_to_end': end_to_end}, repeat_cohort=['cell1'])
    for mode in ('fixed_map', 'end_to_end'):
        assert result['repeats'][mode]['counts']['planned_source_pairs'] == 3
        assert result['gates'][mode + '_exact']['status'] == 'pass'
        assert result['gates'][mode + '_nonempty_top_group']['status'] == 'pass'
        assert len(result['repeats'][mode]['executions']) == 3
        assert result['repeats'][mode]['executions'][0]['cells'][0]['map_sha256']
    partial = review.evaluate(**evidence, repeats={'end_to_end': end_to_end[:2]}, repeat_cohort=['cell1'])
    assert partial['repeats']['end_to_end']['counts']['missing_source_pairs'] == 2
    assert partial['gates']['end_to_end_exact']['status'] == 'unavailable'
    assert partial['repeats']['end_to_end']['executions'][2]['cells'][0]['sources'][0]['status'] == 'missing'
    # All-zero agreement is explicitly different from nonempty ranking agreement.
    for path in fixed:
        cells = list(runner.rows(path / 'cells.jsonl.gz'))
        cells[0]['sources'][0]['importance'] = 0
        write_rows(path / 'cells.jsonl.gz', cells)
    result = review.evaluate(**evidence, repeats={'fixed_map': fixed}, repeat_cohort=['cell1'])
    assert result['repeats']['fixed_map']['counts']['both_all_zero_top_pairs'] == 3
    assert result['repeats']['fixed_map']['nonempty_top_group']['fraction'] is None
    with pytest.raises(ValueError, match='reused an execution'):
        review.evaluate(**evidence, repeats={'end_to_end': [first, first]}, repeat_cohort=['cell1'])
    with pytest.raises(ValueError, match='mode differs'):
        review.evaluate(**evidence, repeats={'end_to_end': fixed}, repeat_cohort=['cell1'])


def test_essential_omission_remains_in_map_denominator(evidence):
    refs = list(runner.rows(evidence['references']))
    maps = [r for r in refs if r['record_type'] == 'map']
    maps[0]['assessment']['essential_omission'] = True
    write_rows(evidence['references'], refs)
    result = review.evaluate(**evidence)
    assert result['maps']['essential_errors'] == 1
    assert result['metrics']['map_fidelity']['fraction'] == 0
    assert result['gates']['essential_map_errors']['status'] == 'fail'


def test_html_escapes_external_fields_and_cli_writes_both_outputs(evidence, tmp_path):
    output, webpage = tmp_path / 'result.json', tmp_path / 'result.html'
    code = review.main(['--absolute', '--candidate', str(evidence['candidate']), '--inputs', str(evidence['inputs']),
                       '--design', str(evidence['design']), '--output', str(output)])
    assert code == 0 and webpage.is_file()
    saved = json.loads(output.read_text())
    assert saved['counts']['frozen_sources'] == 1
    saved['sources'][0]['url'] = '</pre><script>alert(1)</script>'
    hostile = tmp_path / 'escaped.html'
    review.write_absolute_html(hostile, saved)
    html = hostile.read_text()
    assert '<script>alert(1)</script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert '<caption>Acceptance gates</caption>' in html


def test_cost_normalizes_by_completed_eligible_cells_and_rejects_640_bridge(tmp_path):
    inputs, _ = freeze(tmp_path)
    cells = list(runner.rows(inputs / 'cells.jsonl.gz'))
    second = copy.deepcopy(cells[0])
    second['cell_id'] = 'cell2'
    cells.append(second)
    write_rows(inputs / 'cells.jsonl.gz', cells)
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['cells'] = 2
    manifest['files']['cells.jsonl.gz'] = runner.file_hash(inputs / 'cells.jsonl.gz')
    manifest_path.write_text(json.dumps(manifest))
    candidate = report(tmp_path, inputs, 'cost-candidate')
    bridge = report(tmp_path, inputs, 'cost-bridge')
    candidate_cells = list(runner.rows(candidate / 'cells.jsonl.gz'))
    candidate_cells[1]['sources'][0].update(status='inference_failed', importance=None)
    write_rows(candidate / 'cells.jsonl.gz', candidate_cells)
    for path, hours, protocol in ((candidate, 2, 'agentic-source-importance-v4'),
                                  (bridge, 1, 'agentic-source-importance-v3')):
        summary = json.loads((path / 'summary.json').read_text())
        summary.update(protocol=protocol, max_tokens=4096)
        summary['timing']['si']['gpu_hours'] = hours
        (path / 'summary.json').write_text(json.dumps(summary))
    result = review.evaluate(candidate, inputs=inputs, cost_baseline=bridge)
    assert result['cost']['completed_eligible_cells'] == {'bridge': 2, 'candidate': 1}
    assert result['cost']['ratio'] == 4  # Raw total-hours ratio is only two.
    assert result['gates']['warm_compute']['status'] == 'fail'
    summary = json.loads((bridge / 'summary.json').read_text())
    summary['max_tokens'] = 640
    (bridge / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='budget differs'):
        review.evaluate(candidate, inputs=inputs, cost_baseline=bridge)


def test_controls_use_saved_case_results_and_reject_fabricated_receipts(evidence, tmp_path):
    inputs = evidence['inputs']
    cell = next(runner.rows(inputs / 'cells.jsonl.gz'))
    source = cell['sources'][0]
    expectations = [{**{k: cell[k] for k in ('cell_id', 'masked_answer_sha256')},
                     'url': source['url'], 'source_sha256': source['source_sha256'],
                     'acceptable_grade_range': [0, 0], 'expected_relation': 'unsupported'}]
    write_rows(inputs / 'expectations.jsonl.gz', expectations)
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(constructed=True, split='development')
    manifest['files']['expectations.jsonl.gz'] = runner.file_hash(inputs / 'expectations.jsonl.gz')
    manifest_path.write_text(json.dumps(manifest))
    candidate = report(tmp_path, inputs, 'overcredited-control')
    receipt = review.summarize_controls(inputs, candidate)
    assert receipt['attempted_pairs'] == receipt['evaluated_pairs'] == 1
    assert receipt['unsupported_grade4_or_grade5'] == receipt['expectation_failures'] == 1
    fabricated = {**receipt, 'unsupported_grade4_or_grade5': 0, 'expectation_failures': 0}
    with pytest.raises(ValueError, match='receipt differs'):
        review.evaluate(candidate, inputs=inputs, controls=fabricated)
    empty = review.evaluate(candidate, inputs=inputs, resource_usage={'complete': True, 'gpu_hours_total': 0})
    assert empty['gates']['finite_budget']['status'] == 'unavailable'


def test_fresh_cycle_rechecks_saved_development_supplements(evidence, tmp_path):
    from analysis.scripts import run_si_v4_cycle as cycle
    from analysis.scripts.prepare_si_v4_diagnostics import freeze_supplements
    from analysis.tests.test_prepare_si_v4_evaluation import population
    spec = {'prefix': 'Factors include: ', 'items': [f'Factor {i} preserves condition {i}.' for i in range(5)],
            'suffix': ' These factors all matter.', 'independent_items_confirmed': True}
    answer = spec['prefix'] + ' '.join(f'{i + 1}) {text}' for i, text in enumerate(spec['items'])) + spec['suffix']
    original = population(tmp_path / 'supplement-original', 1, answer=answer)
    frozen = freeze_supplements(tmp_path / 'supplements', original, 'qwen38-0', spec)
    receipt = packets.summarize_supplements(frozen['order_inputs'], None, frozen['role_inputs'], None)
    root = tmp_path / 'fresh-cycle'
    root.mkdir()
    shutil.copytree(evidence['inputs'], root / 'candidate-inputs')
    # Preserve the report pointer emitted by the real Coordinator.
    shutil.copytree(evidence['candidate'].parent, root / 'evaluation-results/candidate-e2e-1/reports')
    manifest = json.loads((root / 'candidate-inputs/manifest.json').read_text())
    (root / 'evaluation-plan.json').write_text(json.dumps({'format_version': cycle.FORMAT, 'phase': 'fresh',
        'budget': cycle.BUDGET, 'inventories': {'candidate-inputs': manifest},
        'queue': [{'name': 'candidate-e2e-1', 'inputs': 'candidate-inputs'}], 'repeat_selection': ['cell1'],
        'development_report': {'gates': {'supplementary_controls': {'observed': receipt}}}}))
    output = root / 'evaluation.json'
    assert cycle.assess(SimpleNamespace(config=root / 'config.json', output=output, references=None,
        reference_packets=None, resource_usage=None, supplement_references=None, supplement_packets=None)) == 0
    evaluated = json.loads(output.read_text())
    # The incomplete saved evidence remains visible instead of disappearing, and
    # the real summarizer independently verifies it during fresh assessment.
    assert evaluated['gates']['supplementary_controls']['observed'] == receipt
    assert evaluated['gates']['supplementary_controls']['status'] == 'unavailable'
    assert evaluated['decision'] != 'Ready to scale v4'

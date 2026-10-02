#!/usr/bin/env python3
"""Compare bounded SI review reports; reference-model agreement is not accuracy.

Reads report directories produced by run_source_importance_judge. Never changes
judgments, resolves reviewer disagreements, or promotes a protocol automatically.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.scripts.run_source_importance_judge import rows, file_hash
from analysis.interpretability.pipeline.cluster_bootstrap import cluster_bootstrap, paired_difference


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def load(directory):
    directory = Path(directory)
    cells = {}
    for cell in rows(directory / "cells.jsonl.gz"):
        if cell["cell_id"] in cells:
            raise ValueError("duplicate cell in report")
        cells[cell["cell_id"]] = cell
    return json.loads((directory / "summary.json").read_text()), cells


def comparable(a, b):
    for field in ("judged_answer_sha256", "masked_answer_sha256", "presented", "generator_ranking"):
        if field not in a or field not in b:
            raise ValueError(f"missing comparison input: {field}")
        if a.get(field) != b.get(field):
            raise ValueError(f"comparison inputs differ: {field}")
    sa, sb = ({s["url"]: s for s in c["sources"]} for c in (a, b))
    if sa.keys() != sb.keys() or any(sa[u].get("source_sha256") != sb[u].get("source_sha256") for u in sa):
        raise ValueError("comparison source inputs differ")
    return sa, sb


def compare(baseline, candidate, *, references=None, repetitions=(), repeat_kind="end_to_end", replicates=2000):
    old_summary, old = load(baseline)
    new_summary, new = load(candidate)
    config = lambda s: {k: v for k, v in s["configuration"].items() if k not in ("repetition_id", "tokenizer_path")}
    if config(old_summary) != config(new_summary):
        raise ValueError("architecture comparison requires the same judge configuration")
    if old_summary.get("preprocessing") != new_summary.get("preprocessing") or old_summary.get("max_tokens") != new_summary.get("max_tokens"):
        raise ValueError("preprocessing or output budgets differ; report as a separate configuration comparison")
    if old.keys() != new.keys():
        raise ValueError("architecture comparison requires the same frozen cell cohort")
    refs = {}
    if references:
        for ref in rows(references):
            key = (ref["cell_id"], ref["url"])
            if key in refs:
                raise ValueError("duplicate reference; retain reviewer disagreements explicitly")
            if not ref.get("reviewer_models") or not ref.get("frozen_before_grade_review"):
                raise ValueError("reference provenance and independent range freeze required")
            refs[key] = ref
    counts, comparison_rows, subgroup, maps = Counter(), [], defaultdict(Counter), {}
    for cid in old:
        if old[cid].get("status") != "ok" or new[cid].get("status") != "ok":
            counts["input_ineligible_cells"] += 1
            continue
        sa, sb = comparable(old[cid], new[cid])
        counts["common_complete_cells"] += int(all(sa[u].get("importance") is not None and
                                                   sb[u].get("importance") is not None for u in sa))
        for url in sa:
            counts["observed_sources"] += 1
            a, b = sa[url].get("importance"), sb[url].get("importance")
            counts["baseline_scored"] += a is not None
            counts["candidate_scored"] += b is not None
            reference = refs.get((cid, url))
            if not reference:
                counts["missing_references"] += 1
                continue
            if reference["source_sha256"] != sb[url]["source_sha256"] or reference["masked_answer_sha256"] != new[cid].get("masked_answer_sha256"):
                raise ValueError("reference input hash mismatch")
            if "map_assessment" in reference:
                if reference.get("map_sha256") != new[cid].get("map_sha256") or not reference.get("map_sha256"):
                    raise ValueError("map assessment refers to a different map")
                assessment = reference["map_assessment"]
                if set(assessment) != {"faithful", "essential_omission", "meaning_reversal"} or any(type(v) is not bool for v in assessment.values()):
                    raise ValueError("invalid map assessment")
                previous = maps.setdefault(reference["map_sha256"], assessment)
                if previous != assessment:
                    raise ValueError("map references disagree; retain disagreement as unresolved")
            if "candidate_pair_support" in reference:
                if not reference.get("candidate_raw_output_sha256") or reference["candidate_raw_output_sha256"] != sb[url].get("raw_output_sha256"):
                    raise ValueError("witness assessment refers to a different output")
                for label in reference["candidate_pair_support"]:
                    if label not in ("supported", "unsupported", "ambiguous"):
                        raise ValueError("invalid pair support label")
                    counts["candidate_pairs_" + label] += 1
            interval = reference.get("acceptable_grade_range")
            if interval is None:
                counts["unresolved_ranges"] += 1
                continue
            if not (isinstance(interval, list) and len(interval) == 2 and
                    all(type(n) is int for n in interval) and 0 <= interval[0] <= interval[1] <= 5):
                raise ValueError("invalid independently specified grade range")
            a_ok = interval[0] <= a <= interval[1] if a is not None else None
            b_ok = interval[0] <= b <= interval[1] if b is not None else None
            if a_ok is not None and b_ok is not None:
                counts["paired_resolved_grades"] += 1
                counts["baseline_in_range"] += a_ok
                counts["candidate_in_range"] += b_ok
                comparison_rows.append({"baseline": a_ok, "candidate": b_ok,
                                        "keyword": reference.get("keyword_id")})
                fields = {k: new[cid].get(k) for k in ("model", "method", "engine", "condition")}
                fields.update({k: reference.get(k) for k in ("style", "readiness_stratum", "source_length_stratum")})
                for field, value in fields.items():
                    group = subgroup[f"{field}:{value}"]
                    group["sources"] += 1
                    group["candidate_outside_range"] += not b_ok
                    group["baseline_outside_range"] += not a_ok
    if refs.keys() - {(cid, s["url"]) for cid, c in new.items() for s in c.get("sources", [])}:
        raise ValueError("references contain cases outside this comparison")
    interval = (cluster_bootstrap(comparison_rows, cluster="keyword",
                statistic=paired_difference("candidate", "baseline"), replicates=replicates, seed=20260930)
                if comparison_rows and all(r["keyword"] is not None for r in comparison_rows) else None)
    repeats = Counter()
    seen_executions = {new_summary["execution_sha256"]}
    all_runs = [(new_summary, new)]
    for directory in repetitions:
        summary, cells = load(directory)
        if summary["execution_sha256"] in seen_executions:
            raise ValueError("repeat reused an execution identity; use separate uncached repetition IDs")
        seen_executions.add(summary["execution_sha256"])
        if config(summary) != config(new_summary) or summary["protocol"] != new_summary["protocol"]:
            raise ValueError("repeat configuration differs")
        if not cells.keys() <= new.keys():
            raise ValueError("repeat includes unknown cells")
        all_runs.append((summary, cells))
    # Pairwise comparisons across the three executions; comparisons are correlated.
    for i, (_, first) in enumerate(all_runs):
        for _, second in all_runs[i + 1:]:
            for cid in first.keys() & second.keys():
                if first[cid].get("status") != "ok" or second[cid].get("status") != "ok":
                    continue
                a, b = comparable(first[cid], second[cid])
                if repeat_kind == "fixed_map" and first[cid].get("map_sha256") != second[cid].get("map_sha256"):
                    raise ValueError("fixed-map repetition changed the map")
                for url in a:
                    x, y = a[url].get("importance"), b[url].get("importance")
                    if x is not None and y is not None:
                        repeats["grade_pairs"] += 1
                        repeats["exact"] += x == y
                        repeats["within_one"] += abs(x-y) <= 1
                        repeats["two_or_more"] += abs(x-y) >= 2
                        repeats["grade4_crossings"] += (x >= 4) != (y >= 4)
                    else:
                        repeats["missing_pairs"] += 1
                tops = []
                for sources in (a, b):
                    grades = [s.get("importance") for s in sources.values()]
                    if any(g is None for g in grades):
                        tops.append(None)
                    else:
                        maximum = max(grades, default=0)
                        tops.append({u for u, s in sources.items() if s["importance"] == maximum} if maximum > 0 else set())
                if all(t is not None for t in tops) and any(tops):
                    repeats["nonempty_top_comparisons"] += 1
                    repeats["same_top_group"] += tops[0] == tops[1]
    hours = [s.get("timing", {}).get("si", {}).get("gpu_hours") for s in (old_summary, new_summary)]
    ratio = (hours[1] / hours[0] if counts["common_complete_cells"] and hours[0] and hours[1] is not None and
             not old_summary.get("fixed_maps_sha256") and not new_summary.get("fixed_maps_sha256") else None)
    return {"scientific_result": False, "reference_is_human_gold": False,
            "baseline_report_sha256": file_hash(Path(baseline) / "cells.jsonl.gz"),
            "candidate_report_sha256": file_hash(Path(candidate) / "cells.jsonl.gz"),
            "reference_sha256": file_hash(references) if references else None, "counts": dict(counts),
            "baseline_range_agreement": rate(counts["baseline_in_range"], counts["paired_resolved_grades"]),
            "candidate_range_agreement": rate(counts["candidate_in_range"], counts["paired_resolved_grades"]),
            "paired_keyword_interval": interval, "subgroups": dict(subgroup),
            "map_fidelity": rate(sum(m["faithful"] for m in maps.values()), len(maps)),
            "essential_map_errors": sum(m["essential_omission"] or m["meaning_reversal"] for m in maps.values()),
            "resolved_pair_support": rate(counts["candidate_pairs_supported"],
                counts["candidate_pairs_supported"] + counts["candidate_pairs_unsupported"]),
            "ambiguous_pair_fraction": rate(counts["candidate_pairs_ambiguous"], sum(
                counts["candidate_pairs_" + label] for label in ("supported", "unsupported", "ambiguous"))),
            "repeat_kind": repeat_kind, "repetition_executions": len(all_runs), "repeats": dict(repeats),
            "warm_gpu_hour_ratio": ratio, "warm_gpu_hour_ratio_ceiling": 3.0,
            "promotion": "requires_semantic_review_provenance_map_fidelity_and_fresh_evaluation",
            "limitations": ["Reference-model agreement is not accuracy.",
                            "Source judgments and pairwise repetitions are clustered, not independent.",
                            "No interval is computed without keyword IDs.",
                            "Compute ratio requires matched cohorts and end-to-end runs including mapping."]}


# Absolute evaluation deliberately does not use paired baseline availability as
# a filter. The frozen input inventory is the denominator for every coverage table.
ABSOLUTE_DESIGN = Path(__file__).resolve().parents[1] / 'config/si_v4_absolute_evaluation.json'
NA_STATUSES = {'no_substantive_content', 'global_absence_only', 'unassessable_input'}


def _digest(value):
    import hashlib
    from analysis.scripts.run_source_importance_judge import canonical
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sources(cell):
    result = {}
    for source in cell.get('sources', []):
        if source['url'] in result:
            raise ValueError('duplicate source URL in cell')
        result[source['url']] = source
    return result


def _execution_settings(summary):
    return {k: v for k, v in summary.get('configuration', {}).items()
            if k not in {'repetition_id', 'tokenizer_path', 'code_revision'}}


def _grade(source):
    value = source.get('importance')
    return value if source.get('status') == 'scored' and type(value) is int and 0 <= value <= 5 else None


def _compatible_partial(original, observed):
    for field in ('judged_answer_sha256', 'masked_answer_sha256', 'presented', 'generator_ranking'):
        if field not in original or observed.get(field) != original[field]:
            raise ValueError('comparison inputs differ: ' + field)
    expected, actual = _sources(original), _sources(observed)
    if actual.keys() - expected.keys() or any(actual[url].get('source_sha256') != expected[url].get('source_sha256') for url in actual):
        raise ValueError('comparison source inputs differ')
    if observed.get('map_task_id') != original.get('map_task_id') or any(
            actual[url].get('dependency_id') != expected[url].get('dependency_id') for url in actual):
        raise ValueError('comparison prompt/schema/task identities differ')


def _load_packets(path):
    if path is None:
        return {}
    paths = sorted(p for p in Path(path).rglob('*') if p.suffix in ('.json', '.jsonl')) if Path(path).is_dir() else [Path(path)]
    packets = {}
    for filename in paths:
        records = rows(filename) if filename.suffix == '.jsonl' else [json.loads(filename.read_text())]
        for packet in records:
            if packet.get('format_version') != 'si-v4-reference-packet-v1':
                if 'body' in packet and 'packet_sha256' in packet:
                    raise ValueError('unknown reference packet format')
                continue  # Adjacent manifests and returned reviews are not packets.
            content = {k: v for k, v in packet.items() if k != 'packet_sha256'}
            if _digest(content) != packet['packet_sha256']:
                raise ValueError('reference packet hash mismatch')
            packets[packet['packet_sha256']] = packet
    return packets


def _independent(records, reviewers):
    """Never union reviewer ranges or select a favorable duplicate."""
    by_model = defaultdict(list)
    for record in records:
        by_model[record['reviewer_model']].append(record)
    if set(by_model) != set(reviewers):
        return None
    if any(len({_digest(r) for r in group}) != 1 for group in by_model.values()):
        return None
    return [by_model[model][0] for model in reviewers]


def _ranges(records, reviewers):
    records = _independent(records, reviewers)
    if records is None:
        return None, False
    values = [r.get('acceptable_grade_range') for r in records]
    valid = lambda v: (isinstance(v, list) and len(v) == 2 and
                       all(type(n) is int for n in v) and 0 <= v[0] <= v[1] <= 5)
    if not all(valid(value) for value in values):
        return None, len({_digest(v) for v in values}) > 1
    low, high = max(v[0] for v in values), min(v[1] for v in values)
    return ([low, high] if low <= high else None), len({_digest(v) for v in values}) > 1


def assess_map_references(cell, map_record, answer_map, records, packets, reviewers, *, original_body):
    """Verify a source-blind map review against its frozen original inputs."""
    verified = []
    unavailable = []
    for record in records:
        if record.get('reviewer_model') not in reviewers:
            raise ValueError('unexpected map reference reviewer')
        packet = packets.get(record.get('packet_sha256'))
        if packet is None:
            unavailable.append('reference_packet_missing')
            continue
        if (packet.get('packet_sha256') != _digest({k: v for k, v in packet.items() if k != 'packet_sha256'}) or
            any(record.get(k) != expected or packet.get(k) != expected for k, expected in {
                'record_type': 'map', 'cell_id': cell['cell_id'],
                'masked_answer_sha256': cell['masked_answer_sha256'],
                'map_sha256': answer_map.get('map_sha256')}.items())):
            raise ValueError('map reference input/hash mismatch')
        expected_body = {**original_body, 'masked_answer': map_record['inputs']['answer'], 'answer_map': answer_map}
        if packet.get('body') != expected_body:
            raise ValueError('map reference packet differs from source-blind original/map inputs')
        verified.append(record)
    independent = _independent(verified, reviewers)
    assessments = [record.get('assessment', {}) for record in independent] if independent else []
    essential_errors = any(a.get('essential_omission') is True or a.get('meaning_reversal') is True for a in assessments)
    faithful = bool(assessments) and all(a.get('faithful') is True and
        a.get('importance_roles_faithful') is True and a.get('essential_omission') is False and
        a.get('meaning_reversal') is False for a in assessments)
    return {'status': 'pass' if faithful else 'needs_review' if independent else 'unavailable',
            'complete': bool(independent), 'assessments': assessments,
            'disagreements': len({_digest(a) for a in assessments}) > 1,
            'essential_errors': essential_errors, 'issues': unavailable}


def _repeat_metrics(candidate_summary, frozen, reports, cohort, kind, expected_executions):
    counts, issues, executions = Counter(), [], []
    seen = set()
    evidence = []
    fixed_hashes = set()
    for report in reports:
        summary, cells = load(report)
        execution = summary.get('execution_sha256')
        if not execution or execution in seen:
            raise ValueError('repeat reused an execution identity')
        seen.add(execution)
        if (_execution_settings(summary) != _execution_settings(candidate_summary) or
                summary.get('protocol') != candidate_summary.get('protocol')):
            raise ValueError('repeat configuration differs')
        if any(summary.get(field) != candidate_summary.get(field) for field in ('preprocessing', 'max_tokens')):
            raise ValueError('repeat preprocessing or token budget differs')
        if summary.get('configuration', {}).get('code_revision') != candidate_summary.get('configuration', {}).get('code_revision'):
            raise ValueError('repeat code revision differs')
        fixed_hash = summary.get('fixed_maps_sha256')
        if (kind == 'fixed_map') != bool(fixed_hash):
            raise ValueError('repeat mode differs from fixed-map provenance')
        if fixed_hash:
            fixed_hashes.add(fixed_hash)
        if len(fixed_hashes) > 1:
            raise ValueError('fixed-map repeats use different saved map freezes')
        if not set(cohort) <= set(frozen):
            raise ValueError('repeat cohort includes cells outside frozen inputs')
        executions.append((summary, cells))
        evidence.append({'execution_sha256': execution, 'report': str(report),
                         'fixed_maps_sha256': fixed_hash,
                         'cells': [{'cell_id': cid, 'reported': cid in cells,
                            'map_sha256': cells.get(cid, {}).get('map_sha256'),
                            'map_status': cells.get(cid, {}).get('map_result', {}).get('parsed_output', {}).get('status', 'missing'),
                            'map_ok': cells.get(cid, {}).get('map_result', {}).get('ok'),
                            'sources': [{'url': url, 'status': _sources(cells.get(cid, {})).get(url, {}).get('status', 'missing'),
                                         'grade': _grade(_sources(cells.get(cid, {})).get(url, {}))}
                                        for url in _sources(frozen[cid])]}
                                   for cid in cohort]})
    # Missing executions count as missing planned comparisons, rather than making
    # a two-run fragment look like a completed three-execution experiment.
    if len(executions) > expected_executions:
        raise ValueError('more repeat executions than the frozen design')
    counts['executions_available'] = len(executions)
    counts['executions_expected'] = expected_executions
    evidence += [{'execution_sha256': None, 'report': None, 'status': 'missing_planned_execution',
                  'cells': [{'cell_id': cid, 'reported': False, 'map_sha256': None, 'map_status': 'missing',
                             'sources': [{'url': url, 'status': 'missing', 'grade': None} for url in _sources(frozen[cid])]}
                            for cid in cohort]}] * (expected_executions - len(executions))
    executions += [(None, {})] * (expected_executions - len(executions))
    for i, (_, first) in enumerate(executions):
        for _, second in executions[i + 1:]:
            for cid in cohort:
                left, right = first.get(cid), second.get(cid)
                base = frozen[cid]
                expected = _sources(base)
                counts['planned_cell_pairs'] += 1
                for record in (left, right):
                    if record:
                        _compatible_partial(base, record)
                if kind == 'fixed_map' and left and right:
                    if not left.get('map_sha256') or left.get('map_sha256') != right.get('map_sha256'):
                        issues.append({'cell_id': cid, 'error': 'fixed_map_changed_or_missing'})
                vectors = []
                for record in (left, right):
                    sources = _sources(record or {})
                    vectors.append([_grade(sources.get(url, {})) for url in expected])
                for x, y in zip(*vectors):
                    counts['planned_source_pairs'] += 1
                    if x is None or y is None:
                        counts['missing_source_pairs'] += 1
                    else:
                        counts['observed_source_pairs'] += 1
                        counts['exact'] += x == y
                        counts['within_one'] += abs(x - y) <= 1
                        counts['grade4_crossings'] += (x >= 4) != (y >= 4)
                if not expected or any(value is None for vector in vectors for value in vector):
                    counts['incomplete_top_pairs'] += 1
                    continue
                tops = [{url for url, value in zip(expected, vector) if value == max(vector)}
                        if max(vector) > 0 else set() for vector in vectors]
                if not any(tops):
                    counts['both_all_zero_top_pairs'] += 1
                else:
                    counts['nonempty_top_pairs'] += 1
                    counts['same_top_group'] += tops[0] == tops[1]
    map_changes = [{'cell_id': cid, 'map_sha256_by_execution': [entry['cells'][index].get('map_sha256') for entry in evidence],
                    'changed': len({entry['cells'][index].get('map_sha256') for entry in evidence
                                    if entry['cells'][index].get('map_sha256')}) > 1}
                   for index, cid in enumerate(cohort)]
    return {'cohort': list(cohort), 'counts': dict(counts), 'issues': issues, 'executions': evidence,
            'map_changes': map_changes, 'map_fidelity_scope': 'not semantically assessed by repeat agreement',
            'exact': rate(counts['exact'], counts['planned_source_pairs']),
            'within_one': rate(counts['within_one'], counts['planned_source_pairs']),
            'exact_when_both_scored': rate(counts['exact'], counts['observed_source_pairs']),
            'nonempty_top_group': rate(counts['same_top_group'], counts['nonempty_top_pairs'])}


def _cost(candidate_summary, candidate_cells, frozen, baseline):
    if baseline is None:
        return {'status': 'unavailable', 'reason': 'matched bridge report missing', 'ratio': None}
    summary, cells = load(baseline)
    if set(cells) != set(frozen) or set(candidate_cells) != set(frozen):
        return {'status': 'unavailable', 'reason': 'cost cohorts incomplete or unmatched', 'ratio': None}
    if summary.get('protocol') not in ('agentic-source-importance-v3', 'agentic-source-importance-v3-mask-v2'):
        raise ValueError('cost baseline must be the retained SI-v3 bridge')
    if summary.get('execution_sha256') == candidate_summary.get('execution_sha256'):
        raise ValueError('cost bridge reused candidate execution')
    if _execution_settings(summary) != _execution_settings(candidate_summary):
        raise ValueError('cost bridge judge configuration differs')
    for field in ('preprocessing', 'max_tokens'):
        if summary.get(field) != candidate_summary.get(field):
            raise ValueError('cost bridge preprocessing or budget differs')
    if summary.get('fixed_maps_sha256') or candidate_summary.get('fixed_maps_sha256'):
        return {'status': 'unavailable', 'reason': 'cost requires end-to-end runs', 'ratio': None}
    completed = [0, 0]
    for cid, original in frozen.items():
        for index, group in enumerate((cells, candidate_cells)):
            comparable(original, group[cid])
            sources = _sources(group[cid])
            eligible = [s for s in sources.values() if s.get('status') not in NA_STATUSES]
            completed[index] += bool(eligible) and all(_grade(s) is not None for s in eligible)
    hours = [s.get('timing', {}).get('si', {}).get('gpu_hours') for s in (summary, candidate_summary)]
    if any(type(h) not in (int, float) or not math.isfinite(h) or not h > 0 for h in hours) or not all(completed):
        return {'status': 'unavailable', 'reason': 'positive warm timing/completions missing', 'ratio': None}
    per_cell = [hours[i] / completed[i] for i in (0, 1)]
    return {'status': 'counted', 'completed_eligible_cells': {'bridge': completed[0], 'candidate': completed[1]},
            'warm_gpu_hours': {'bridge': hours[0], 'candidate': hours[1]},
            'warm_gpu_hours_per_completed_eligible_cell': {'bridge': per_cell[0], 'candidate': per_cell[1]},
            'ratio': per_cell[1] / per_cell[0], 'bridge_execution_sha256': summary['execution_sha256']}


def summarize_controls(inputs, report):
    """Read a constructed freeze and judge report; never trust a supplied pass flag."""
    inputs, report = Path(inputs), Path(report)
    from analysis.scripts.prepare_si_v4_evaluation import load_freeze
    from analysis.interpretability.pipeline import source_importance_v4 as v4
    manifest, frozen_cells, tasks = load_freeze(inputs)
    frozen = {c['cell_id']: c for c in frozen_cells}
    if manifest.get('constructed') is not True:
        raise ValueError('constructed controls require constructed frozen inputs')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_hash(inputs / name) != expected:
            raise ValueError('constructed input hash mismatch')
    summary, cells = load(report)
    if summary.get('manifest_sha256') != file_hash(inputs / 'manifest.json'):
        raise ValueError('constructed report manifest mismatch')
    counts, details = Counter(), []
    expectations = list(rows(inputs / 'expectations.jsonl.gz'))
    if len({e['cell_id'] for e in expectations}) != len(expectations):
        raise ValueError('duplicate constructed expectation')
    if set(cells) - {e['cell_id'] for e in expectations}:
        raise ValueError('unexpected constructed result cell')
    for expected in expectations:
        counts['attempted_pairs'] += 1
        current = cells.get(expected['cell_id'], {})
        source = _sources(current).get(expected['url'], {})
        if source and source.get('source_sha256') != expected['source_sha256']:
            raise ValueError('constructed source hash mismatch')
        if current and current.get('masked_answer_sha256') != expected['masked_answer_sha256']:
            raise ValueError('constructed answer hash mismatch')
        grade = _grade(source)
        validation_error = None
        try:
            if current:
                _compatible_partial(frozen[expected['cell_id']], current)
            if source:
                mapper = tasks[frozen[expected['cell_id']]['map_task_id']]
                mapped = v4.reproduce_map_result(mapper, current['map_result'])
                if mapped['map_sha256'] != current.get('map_sha256'):
                    raise ValueError('constructed map hash mismatch')
                if source.get('status') == 'scored':
                    raw = source.get('raw_output')
                    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != source.get('raw_output_sha256'):
                        raise ValueError('constructed raw output missing/hash mismatch')
                    dependency = tasks[_sources(frozen[expected['cell_id']])[expected['url']]['dependency_id']]
                    parsed = v4.materialize_source(dependency, mapper, mapped)['validator'](raw)
                    if parsed != source.get('parsed_output') or parsed['importance'] != grade:
                        raise ValueError('constructed output differs from validated raw output')
                elif source.get('status') in NA_STATUSES and source['status'] != mapped['eligibility']:
                    raise ValueError('constructed absence status differs from map')
        except (ValueError, KeyError, TypeError) as error:
            validation_error = str(error)
        counts['structural_errors'] += validation_error is not None
        interval = expected['acceptable_grade_range']
        assessed = validation_error is None and (grade is not None or (interval is None and source.get('status') in NA_STATUSES))
        good = assessed and ((interval is None and grade is None) or
                            (interval is not None and grade is not None and interval[0] <= grade <= interval[1]))
        counts['evaluated_pairs'] += assessed
        counts['expectation_failures'] += assessed and not good
        counts['missing_pairs'] += not assessed
        counts['unsupported_grade4_or_grade5'] += expected['expected_relation'] in ('unsupported', 'contradicted') and grade in (4, 5)
        details.append({'cell_id': expected['cell_id'], 'grade': grade, 'status': source.get('status', 'missing'),
                        'expected_range': interval, 'expected_relation': expected['expected_relation'], 'in_range': good,
                        'structural_validation_error': validation_error})
    return {'format_version': 'si-v4-controls-assessment-v1', 'inputs': str(inputs.resolve()),
        'report': str(report.resolve()), 'input_manifest_sha256': file_hash(inputs / 'manifest.json'),
        'report_sha256': file_hash(report / 'cells.jsonl.gz'), 'split': manifest.get('split'),
        **dict(counts), 'details': details}


def evaluate(candidate, *, inputs, design=ABSOLUTE_DESIGN, references=None,
             reference_packets=None, repeats=None, repeat_cohort=(), cost_baseline=None,
             phase='development', resource_usage=None, controls=None, selection=None, supplements=None):
    """Assess one frozen v4 candidate against absolute gates, without filtering failures.

    References are independent per-reviewer grade/map/support JSONL records.
    Repeats contains complete execution lists for each mode, including the first
    run when reused. A missing receipt/reference yields an unavailable gate.
    """
    design_path = Path(design)
    policy = json.loads(design_path.read_text())
    if policy.get('format_version') != 'si-v4-absolute-evaluation-design-v1':
        raise ValueError('versioned absolute evaluation design required')
    if phase not in ('development', 'fresh'):
        raise ValueError('unknown evaluation phase')
    from analysis.scripts.prepare_si_v4_evaluation import load_freeze
    manifest, frozen_cells, tasks = load_freeze(inputs)
    frozen = {cell['cell_id']: cell for cell in frozen_cells}
    summary, candidate_cells = load(candidate)
    if set(candidate_cells) - set(frozen):
        raise ValueError('candidate contains cells outside frozen cohort')
    if summary.get('manifest_sha256') != file_hash(Path(inputs) / 'manifest.json'):
        raise ValueError('candidate report is not bound to frozen input manifest')
    packets = _load_packets(reference_packets)
    reference_groups, issues = defaultdict(list), []
    for reference in rows(references) if references else ():
        kind, cid = reference.get('record_type'), reference.get('cell_id')
        if kind not in ('grade', 'map', 'support') or cid not in frozen:
            raise ValueError('reference record type/cell is outside this evaluation')
        if reference.get('reviewer_model') not in policy['reference_models']:
            raise ValueError('unexpected reference reviewer model')
        cell = frozen[cid]
        if reference.get('masked_answer_sha256') != cell.get('masked_answer_sha256'):
            raise ValueError('reference answer hash mismatch')
        packet = packets.get(reference.get('packet_sha256'))
        if packet is None or packet.get('record_type') != kind or packet.get('cell_id') != cid:
            issues.append({'cell_id': cid, 'error': 'reference_packet_provenance_unavailable'})
            continue
        if packet.get('masked_answer_sha256') != cell.get('masked_answer_sha256'):
            raise ValueError('reference packet answer hash mismatch')
        j1 = tasks.get(cell.get('j1_task_id'), {})
        original_body = {'request': j1.get('request'), 'answer': j1.get('answer')}
        if any(packet.get('body', {}).get(k) != v for k, v in original_body.items()):
            raise ValueError('reference packet original input differs')
        if kind == 'grade' and (reference.get('frozen_before_grade_review') is not True or
                                reference.get('whole_source_checked') is not True):
            raise ValueError('grade reference requires independent blind freeze and complete source review')
        if kind == 'map':
            if reference.get('map_sha256') != candidate_cells.get(cid, {}).get('map_sha256'):
                raise ValueError('map reference hash mismatch')
            if packet.get('map_sha256') != reference.get('map_sha256'):
                raise ValueError('map packet hash mismatch')
            if set(packet.get('body', {})) != {'request', 'answer', 'masked_answer', 'answer_map'}:
                raise ValueError('map packet includes source-dependent or missing fields')
            key = (kind, cid)
        else:
            url = reference.get('url')
            source = _sources(cell).get(url)
            if source is None or reference.get('source_sha256') != source.get('source_sha256'):
                raise ValueError('reference source hash mismatch')
            if packet.get('url') != url or packet.get('source_sha256') != source['source_sha256']:
                raise ValueError('reference packet source join differs')
            dep = tasks.get(source.get('dependency_id', source.get('judge_task_id')), {})
            if packet.get('body', {}).get('source') != {'title': dep.get('source_title'), 'text': dep.get('source_text')}:
                raise ValueError('reference packet source content differs')
            if kind == 'grade' and set(packet.get('body', {})) != {'request', 'answer', 'source'}:
                raise ValueError('grade packet is not blinded to candidate map and outputs')
            key = (kind, cid, url)
            if kind == 'support':
                actual = _sources(candidate_cells.get(cid, {})).get(url, {})
                if reference.get('candidate_raw_output_sha256') != actual.get('raw_output_sha256'):
                    raise ValueError('support assessment output hash mismatch')
                if packet.get('candidate_raw_output_sha256') != reference.get('candidate_raw_output_sha256'):
                    raise ValueError('support packet output hash differs')
                mapped = candidate_cells.get(cid, {}).get('map_result', {}).get('parsed_output')
                parsed = actual.get('parsed_output')
                expected_body = {**original_body, 'source': {'title': dep.get('source_title'), 'text': dep.get('source_text')},
                                 'answer_map': mapped, 'candidate_output': parsed, 'source_assessment_id': 'source',
                                 'findings': [{'finding_id': _digest({'index': i, 'finding': f}), **f}
                                              for i, f in enumerate((parsed or {}).get('findings', []))]}
                if packet.get('map_sha256') != (mapped or {}).get('map_sha256') or packet.get('body') != expected_body:
                    raise ValueError('support packet content differs from candidate/source/map')
                key += (reference.get('finding_id'),)
        reference_groups[key].append(reference)
    reviewers = policy['reference_models']
    counts, map_counts, support_counts = Counter(), Counter(), Counter()
    subgroups, cell_rows, source_rows, support_rows = defaultdict(Counter), [], [], []
    source_audits, decisive_errors = [], []
    seen_findings = set()
    for cid, original in frozen.items():
        current = candidate_cells.get(cid)
        counts['frozen_cells'] += 1
        counts['reported_cells'] += current is not None
        if current is not None:
            _compatible_partial(original, current)
        sources = _sources(current or {})
        map_result = (current or {}).get('map_result') or {}
        map_record = tasks.get(original.get('map_task_id'))
        validated_map, map_validation_error, faithful = None, None, False
        if map_record and map_result:
            try:
                from analysis.interpretability.pipeline import source_importance_v4 as v4
                validated_map = v4.reproduce_map_result(map_record, map_result)
                if validated_map['map_sha256'] != (current or {}).get('map_sha256'):
                    raise ValueError('reported map hash differs from validated map')
            except (ValueError, KeyError, TypeError) as error:
                map_validation_error = str(error)
                validated_map = None
        expected_map = original.get('map_task_id') is not None
        if expected_map:
            map_counts['expected'] += 1
            map_counts['produced'] += bool((current or {}).get('map_sha256'))
            j1 = tasks.get(original.get('j1_task_id'), {})
            assessed = assess_map_references(original, map_record, validated_map,
                reference_groups[('map', cid)], packets, reviewers,
                original_body={'request': j1.get('request'), 'answer': j1.get('answer')}) if validated_map else None
            map_counts['reference_complete'] += bool(assessed and assessed['complete'])
            map_counts['essential_errors'] += bool(assessed and assessed['essential_errors'])
            map_counts['disagreements'] += bool(assessed and assessed['disagreements'])
            faithful = bool(assessed and assessed['status'] == 'pass')
            map_counts['faithful'] += faithful
            map_counts['unresolved_or_error'] += not faithful
        outcomes = Counter()
        for url, source in _sources(original).items():
            observed = sources.get(url, {})
            status = observed.get('status', 'missing')
            grade = _grade(observed)
            if status == 'scored' and grade is None:
                status = 'invalid_retained_score'
            counts['frozen_sources'] += 1
            counts['sources_' + status] += 1
            outcomes[status] += 1
            counts['eligible_sources'] += status not in NA_STATUSES
            counts['scored_sources'] += grade is not None
            interval, disagreement = _ranges(reference_groups[('grade', cid, url)], reviewers)
            counts['reference_complete'] += _independent(reference_groups[('grade', cid, url)], reviewers) is not None
            counts['reference_disagreements'] += disagreement
            counts['references_resolved'] += interval is not None
            counts['references_unresolved_or_missing'] += interval is None
            in_range = interval is not None and grade is not None and interval[0] <= grade <= interval[1]
            counts['grade_in_range'] += in_range
            counts['resolved_and_scored'] += interval is not None and grade is not None
            counts['missed_positive_reference'] += grade == 0 and interval is not None and interval[0] > 0
            parsed = observed.get('parsed_output') or {}
            retained = observed.get('status') == 'scored'
            counts['retained_outputs'] += retained
            valid, structural_error = False, None
            if retained:
                try:
                    raw = observed.get('raw_output')
                    if not isinstance(raw, str):
                        raise ValueError('retained source raw output missing')
                    if hashlib.sha256(raw.encode()).hexdigest() != observed.get('raw_output_sha256'):
                        raise ValueError('retained source raw output hash mismatch')
                    if validated_map is None:
                        raise ValueError('retained source map unavailable/invalid: ' + str(map_validation_error))
                    dep = tasks[source.get('dependency_id', source.get('judge_task_id'))]
                    reproduced = v4.materialize_source(dep, map_record, validated_map)['validator'](raw)
                    if reproduced != parsed or grade is None or reproduced.get('importance') != grade or reproduced.get('status') != 'scored':
                        raise ValueError('retained source differs from validated raw output')
                    valid = True
                except (ValueError, KeyError, TypeError) as error:
                    structural_error = str(error)
            counts['structurally_valid_retained_outputs'] += valid
            source_rows.append({'cell_id': cid, 'url': url, 'status': status, 'grade': grade,
                'reference_range': interval, 'reference_disagreement': disagreement,
                'grade_in_range': in_range if interval is not None else None,
                'structural_validation_error': structural_error,
                'inference_error': observed.get('error'),
                'validation_attempt_count': observed.get('validation_attempt_count'),
                'deterministic_repairs': observed.get('deterministic_repairs', [])})
            if parsed:
                key = ('support', cid, url, 'source')
                seen_findings.add(key)
                support_counts['source_reviews_expected'] += 1
                audited = _independent(reference_groups[key], reviewers)
                complete_audit = bool(audited) and all(a.get('whole_source_checked') is True and
                    a.get('missed_support') in ('confirmed', 'none', 'unresolved') and
                    a.get('role_consistency') in ('consistent', 'inconsistent', 'map_issue', 'unresolved') and
                    isinstance(a.get('decisive_error_categories'), list) and
                    set(a['decisive_error_categories']) <= {'entity', 'qualification', 'negation', 'essential_content'} and
                    isinstance(a.get('explanation'), str) and a['explanation'].strip() for a in audited)
                support_counts['source_reviews_complete'] += complete_audit
                support_counts['source_reviews_unavailable'] += not complete_audit
                missed = [a['missed_support'] for a in audited] if complete_audit else []
                roles = [a['role_consistency'] for a in audited] if complete_audit else []
                support_counts['confirmed_missed_support'] += bool(missed and all(v == 'confirmed' for v in missed))
                support_counts['confirmed_role_inconsistency'] += bool(roles and all(v == 'inconsistent' for v in roles))
                support_counts['confirmed_map_issue'] += bool(roles and all(v == 'map_issue' for v in roles))
                support_counts['source_review_disagreements'] += len(set(missed)) > 1 or len(set(roles)) > 1
                categories = sorted(set.intersection(*(set(a['decisive_error_categories']) for a in audited))) if complete_audit else []
                audit = {'cell_id': cid, 'url': url, 'complete': complete_audit,
                    'missed_support': missed, 'role_consistency': roles,
                    'confirmed_decisive_error_categories': categories,
                    'map_independently_faithful': faithful, 'reviewer_records': reference_groups[key]}
                source_audits.append(audit)
                if categories:
                    decisive_errors.append(audit)
            for index, finding in enumerate(parsed.get('findings', [])):
                finding_id = _digest({'index': index, 'finding': finding})
                key = ('support', cid, url, finding_id)
                seen_findings.add(key)
                audited = _independent(reference_groups[key], reviewers)
                judgments = [a.get('judgment') for a in audited] if audited else []
                support_counts['findings_expected'] += 1
                if len(judgments) == len(reviewers) and len(set(judgments)) == 1 and judgments[0] in ('supported', 'unsupported'):
                    support_counts[judgments[0]] += 1
                else:
                    support_counts['ambiguous_or_missing'] += 1
                witness = [a.get('witness_assessment') for a in audited] if audited else []
                label = witness[0] if len(witness) == len(reviewers) and len(set(witness)) == 1 else 'unresolved'
                support_counts['witness_' + (label if label in ('sufficient', 'insufficient') else 'unresolved')] += 1
                support_rows.append({'cell_id': cid, 'url': url, 'finding_id': finding_id,
                    'candidate_relation': finding.get('relation'), 'finding': finding,
                    'reference_judgments': judgments, 'witness_assessments': witness,
                    'reviewer_records': reference_groups[key]})
            fields = {key: original.get(key, 'unknown') for key in ('model', 'method', 'engine', 'condition')}
            from analysis.scripts.prepare_si_v4_evaluation import answer_style, readiness_range
            fields.update(style=original.get('style', answer_style(original)),
                          readiness_stratum=original.get('readiness_stratum', readiness_range(original)))
            for field, value in fields.items():
                group = subgroups[f'{field}:{value if value is not None else "unknown"}']
                group['frozen_sources'] += 1
                group['scored'] += grade is not None
                group['status_' + status] += 1
                group['reference_resolved'] += interval is not None
                group['reference_unresolved_or_missing'] += interval is None
                group['grade_in_range'] += in_range
        expected_sources = _sources(original)
        complete = bool(expected_sources) and all(_grade(sources.get(url, {})) is not None for url in expected_sources)
        counts['complete_cells'] += complete
        cell_rows.append({'cell_id': cid, 'reported': current is not None, 'complete': complete,
                          'input_status': original.get('status'), 'source_statuses': dict(outcomes),
                          'map_status': map_result.get('parsed_output', {}).get('status', 'missing'),
                          'map_validation_error': map_validation_error,
                          'map_result': map_result})
    if any(key[0] == 'support' and key not in seen_findings for key in reference_groups):
        raise ValueError('support reference identifies a finding absent from candidate')
    repeat_selection = repeat_cohort if isinstance(repeat_cohort, dict) else None
    cohort = list(repeat_selection.get('selected_cell_ids', [])) if repeat_selection else list(repeat_cohort)
    if len(cohort) != len(set(cohort)) or not set(cohort) <= set(frozen):
        raise ValueError('repeat cohort must contain distinct frozen cells')
    repeat_reports = {mode: _repeat_metrics(summary, frozen, (repeats or {}).get(mode, []), cohort,
                        mode, policy['uncached_repetitions']) for mode in policy['repeat_modes']}
    repeat_executions = [e['execution_sha256'] for result in repeat_reports.values()
                         for e in result['executions'] if e['execution_sha256']]
    if len(set(repeat_executions)) != len(repeat_executions):
        raise ValueError('repeat execution reused across evaluation modes')
    cost = _cost(summary, candidate_cells, frozen, cost_baseline)
    metrics = {'eligible_source_completion': rate(counts['scored_sources'], counts['eligible_sources']),
        'retained_structural_validity': rate(counts['structurally_valid_retained_outputs'], counts['retained_outputs']),
        'reference_grade_range_agreement': rate(counts['grade_in_range'], counts['references_resolved']),
        'grade_range_agreement_when_scored': rate(counts['grade_in_range'], counts['resolved_and_scored']),
        'map_fidelity': rate(map_counts['faithful'], map_counts['expected']),
        'resolved_reference_pair_support': rate(support_counts['witness_sufficient'], support_counts['witness_sufficient'] + support_counts['witness_insufficient']),
        'unresolved_pair_fraction': rate(support_counts['witness_unresolved'], support_counts['findings_expected']),
        'whole_source_relation_agreement': rate(support_counts['supported'], support_counts['supported'] + support_counts['unsupported'])}
    thresholds, gates = policy['thresholds'], {}
    for name, metric in metrics.items():
        key = name + ('_max' if name == 'unresolved_pair_fraction' else '_min')
        if key in thresholds:
            value = metric['fraction']
            passed = value is not None and (value <= thresholds[key] if key.endswith('_max') else value >= thresholds[key])
            gates[name] = {'status': 'pass' if passed else 'unavailable' if value is None else 'fail',
                           'observed': value, 'threshold': thresholds[key]}
    gates['input_integrity'] = {'status': 'pass' if not issues else 'fail', 'issues': issues}
    gates['reference_coverage'] = {'status': 'pass' if counts['reference_complete'] == counts['frozen_sources'] and
        map_counts['reference_complete'] == map_counts['expected'] else 'unavailable',
        'unresolved_source_references': counts['references_unresolved_or_missing'],
        'missing_independent_reviews': counts['frozen_sources'] - counts['reference_complete']}
    gates['essential_map_errors'] = {'status': 'pass' if map_counts['essential_errors'] == 0 and map_counts['reference_complete'] == map_counts['expected']
        else 'fail' if map_counts['essential_errors'] else 'unavailable', 'count': map_counts['essential_errors']}
    gates['whole_source_review'] = {'status': 'pass' if support_counts['source_reviews_expected'] > 0 and
        support_counts['source_reviews_complete'] == support_counts['source_reviews_expected'] else 'unavailable',
        'complete': support_counts['source_reviews_complete'], 'expected': support_counts['source_reviews_expected']}
    gates['decisive_support_errors'] = {'status': 'fail' if decisive_errors else
        'pass' if gates['whole_source_review']['status'] == 'pass' else 'unavailable',
        'cases': decisive_errors,
        'criterion': 'independently confirmed decisive entity/qualification/negation/essential-content errors in revised outputs'}
    role_violations = [audit for audit in source_audits if audit['complete'] and
                      (all(v == 'inconsistent' for v in audit['role_consistency']) or
                       all(v == 'map_issue' for v in audit['role_consistency']))]
    gates['fixed_map_contract'] = {'status': 'fail' if role_violations else
        'pass' if gates['whole_source_review']['status'] == 'pass' else 'unavailable',
        'cases': role_violations,
        'criterion': 'full-support roles must not be redefined; substantive map issues require repair'}
    for mode, result in repeat_reports.items():
        available = result['counts']['executions_available'] == policy['uncached_repetitions'] and len(cohort) == policy['repeat_cells']
        checks = [('exact', 'repeat_exact_grade_agreement_min'), ('within_one', 'repeat_within_one_grade_min'),
                  ('nonempty_top_group', 'nonempty_top_group_agreement_min')]
        for metric, threshold in checks:
            value = result[metric]['fraction']
            gates[mode + '_' + metric] = {'status': 'unavailable' if not available or value is None else
                'pass' if value >= thresholds[threshold] and not result['issues'] else 'fail', 'observed': value,
                'threshold': thresholds[threshold]}
    gates['warm_compute'] = {'status': 'unavailable' if cost['ratio'] is None else
        'pass' if cost['ratio'] <= thresholds['warm_gpu_hours_per_completed_eligible_cell_ratio_max'] else 'fail',
        'observed': cost['ratio'], 'threshold': thresholds['warm_gpu_hours_per_completed_eligible_cell_ratio_max']}
    usage = json.loads(Path(resource_usage).read_text()) if isinstance(resource_usage, (str, Path)) else resource_usage
    budget = policy['budget']
    valid_usage = isinstance(usage, dict) and usage.get('complete') is True and type(usage.get('gpu_hours_total')) in (int, float)
    valid_usage = valid_usage and isinstance(usage.get('gpu_hours_by_phase'), dict) and set(usage['gpu_hours_by_phase']) == {'development', 'fresh'}
    valid_usage = valid_usage and all(type(usage['gpu_hours_by_phase'].get(p)) in (int, float) for p in ('development', 'fresh'))
    if valid_usage:
        values = [usage['gpu_hours_total'], *usage['gpu_hours_by_phase'].values()]
        valid_usage = all(math.isfinite(v) and v >= 0 for v in values) and abs(usage['gpu_hours_total'] - sum(usage['gpu_hours_by_phase'].values())) < 1e-6
    within_budget = valid_usage and usage['gpu_hours_total'] <= budget['gpu_hours_total_max'] and all(
        usage['gpu_hours_by_phase'][p] <= budget['gpu_hours_' + p + '_max'] for p in ('development', 'fresh'))
    gates['finite_budget'] = {'status': 'unavailable' if not valid_usage else 'pass' if within_budget else 'fail',
                             'limits': budget, 'observed': usage}
    controls_value = json.loads(Path(controls).read_text()) if isinstance(controls, (str, Path)) else controls
    if isinstance(controls_value, dict) and controls_value.get('inputs') and controls_value.get('report'):
        verified_controls = summarize_controls(controls_value['inputs'], controls_value['report'])
        if controls_value != verified_controls:
            raise ValueError('constructed receipt differs from saved evidence')
        controls_value = verified_controls
        controls_summary, _ = load(controls_value['report'])
        if (_execution_settings(controls_summary) != _execution_settings(summary) or
            controls_summary.get('configuration', {}).get('code_revision') != summary.get('configuration', {}).get('code_revision') or
            any(controls_summary.get(field) != summary.get(field) for field in ('preprocessing', 'max_tokens', 'protocol'))):
            raise ValueError('constructed judge settings differ from candidate')
    else:
        controls_value = None
    expected_controls = policy['constructed_heldout_pairs' if phase == 'fresh' else 'constructed_development_pairs']
    valid_controls = (isinstance(controls_value, dict) and controls_value.get('attempted_pairs') == expected_controls and
        controls_value.get('evaluated_pairs') == expected_controls and
        controls_value.get('split') == ('heldout' if phase == 'fresh' else 'development') and type(controls_value.get('unsupported_grade4_or_grade5')) is int)
    gates['constructed_controls'] = {'status': 'unavailable' if not valid_controls else
        'pass' if controls_value['unsupported_grade4_or_grade5'] == 0 and controls_value['expectation_failures'] == 0 else 'fail', 'observed': controls_value}
    supplement_value = json.loads(Path(supplements).read_text()) if isinstance(supplements, (str, Path)) else supplements
    if isinstance(supplement_value, dict) and supplement_value.get('evidence'):
        from analysis.scripts.prepare_si_v4_evaluation import summarize_supplements
        verified_supplements = summarize_supplements(**supplement_value['evidence'])
        if supplement_value != verified_supplements:
            raise ValueError('supplement receipt differs from saved evidence')
        supplement_value = verified_supplements
        for name in ('order_report', 'role_report'):
            if not supplement_value['evidence'].get(name):
                continue
            supplemental_summary, _ = load(supplement_value['evidence'][name])
            if (_execution_settings(supplemental_summary) != _execution_settings(summary) or
                supplemental_summary.get('configuration', {}).get('code_revision') != summary.get('configuration', {}).get('code_revision') or
                any(supplemental_summary.get(field) != summary.get(field) for field in ('preprocessing', 'max_tokens', 'protocol'))):
                raise ValueError('supplementary judge settings differ from candidate')
    else:
        supplement_value = None
    gates['supplementary_controls'] = {'status': 'unavailable' if supplement_value is None or
        supplement_value['status'] == 'incomplete' else 'pass' if supplement_value['status'] == 'pass' else 'fail',
        'observed': supplement_value}
    expected_cells = policy['historical_development_cells'] if phase == 'development' else policy['fresh_core_cells'] + policy['fresh_stress_cells']
    gates['cohort_size'] = {'status': 'pass' if len(frozen) == expected_cells else 'fail',
                          'observed': len(frozen), 'expected': expected_cells}
    gates['fresh_evaluation'] = {'status': 'pass' if phase == 'fresh' else 'unavailable',
                               'reason': 'development evidence cannot establish fresh acceptance'}
    selection_value = json.loads(Path(selection).read_text()) if isinstance(selection, (str, Path)) else selection
    if phase == 'fresh':
        selection_valid = isinstance(selection_value, dict) and selection_value.get('format_version') == 'si-v4-fresh-selection-v1'
        selection_valid = selection_valid and set(selection_value.get('selected_cell_ids', [])) == set(frozen)
        selection_valid = selection_valid and selection_value.get('selection_uses_judge_outcomes') is False
        selection_valid = selection_valid and bool(selection_value.get('exclusion_manifest_sha256'))
        if selection_valid:
            core, stress = selection_value.get('core_prompt_ids', []), selection_value.get('stress_prompt_ids', [])
            selection_valid = (len(set(core)) == policy['fresh_core_paired_prompts'] and
                len(set(stress)) == policy['fresh_stress_paired_prompts'] and not set(core) & set(stress))
            grouped = defaultdict(list)
            for cell in frozen.values():
                grouped[cell.get('prompt_id')].append(cell)
            selection_valid = selection_valid and set(grouped) == set(core + stress) and all(
                sorted(c.get('model', '') for c in group) == ['llama4', 'qwen38'] and
                len({(c.get('method'), c.get('engine'), c.get('condition')) for c in group}) == 1
                for group in grouped.values())
            selection_valid = selection_valid and not set(grouped) & set(selection_value.get('excluded_prompt_ids', []))
            for cell in frozen.values():
                keywords = cell.get('keyword_memberships', {}).get('keyword_ids', [])
                selection_valid = selection_valid and bool(keywords) and not set(keywords) & set(selection_value.get('excluded_keyword_ids', []))
        gates['fresh_selection_provenance'] = {'status': 'pass' if selection_valid else 'unavailable'}
        repeats_selected = isinstance(repeat_selection, dict) and repeat_selection.get('selection_uses_judge_outcomes') is False
        repeats_selected = repeats_selected and repeat_selection.get('input_manifest_sha256') == file_hash(Path(inputs) / 'manifest.json')
        gates['repeat_selection_provenance'] = {'status': 'pass' if repeats_selected else 'unavailable'}
    ready = all(g['status'] == 'pass' for g in gates.values())
    development_passed = all(g['status'] == 'pass' for name, g in gates.items() if name != 'fresh_evaluation')
    reconsider = any(audit['map_independently_faithful'] for audit in decisive_errors)
    decision = ('Ready to scale v4' if ready else 'Judge configuration needs reconsideration' if reconsider else
                'Needs a specific further repair')
    return {'format_version': 'si-v4-absolute-evaluation-v1', 'phase': phase,
        'decision': decision, 'development_gate_passed': development_passed,
        'decision_scope': 'screening recommendation; no production launch or model reconfiguration authorized',
        'decision_reasons': [name for name, gate in gates.items() if gate['status'] != 'pass'],
        'scientific_result': False, 'reference_is_human_gold': False,
        'design_sha256': file_hash(design_path), 'input_manifest_sha256': file_hash(Path(inputs) / 'manifest.json'),
        'candidate_execution_sha256': summary.get('execution_sha256'),
        'candidate_report': str(Path(candidate).resolve()), 'input_directory': str(Path(inputs).resolve()),
        'candidate_run_summary': summary,
        'counts': dict(counts), 'maps': dict(map_counts), 'map_fidelity_scope': 'initial candidate maps only',
        'support': dict(support_counts),
        'metrics': metrics, 'subgroups': dict(subgroups), 'repeats': repeat_reports, 'cost': cost,
        'gates': gates, 'cells': cell_rows, 'sources': source_rows, 'support_audits': support_rows,
        'source_audits': source_audits,
        'reference_records': [record for group in reference_groups.values() for record in group],
        'limitations': ['Reference model agreement is not ground truth.',
            'All attempted frozen cells remain in coverage denominators; missing scores are not zero.',
            'Historical comparisons and relative quality improvements do not determine absolute acceptance.',
            'Full-source relation assessments and collective witness sufficiency are separate.',
            'Persistent substantive support errors require judge-configuration review before another cycle.']}


def write_absolute_html(path, report):
    """A portable report with no network dependencies or unescaped source content."""
    import html
    escaped = lambda value: html.escape(str(value))
    gates = ''.join('<tr><td>' + escaped(name) + '</td><td>' + escaped(gate['status']) +
                    '</td><td><pre>' + escaped(json.dumps(gate, ensure_ascii=False, indent=2)) + '</pre></td></tr>'
                    for name, gate in report['gates'].items())
    text = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    text += '<title>SI-v4 absolute evaluation</title><style>body{max-width:1100px;margin:3rem auto;padding:0 1rem;font:16px system-ui;color:#192b31;background:#f6f7f4}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bcc8c7;padding:.6rem;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer}</style>'
    text += '<h1>SI-v4 absolute evaluation</h1><p><strong>' + escaped(report['decision']) + '</strong></p>'
    text += '<p>Phase: ' + escaped(report['phase']) + '. Model-reference evidence; consensus is not ground truth.</p>'
    text += '<table><caption>Acceptance gates</caption><thead><tr><th>Criterion</th><th>Status</th><th>Evidence</th></tr></thead><tbody>' + gates + '</tbody></table>'
    text += '<details><summary>Complete attempted-case report</summary><pre>' + escaped(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre></details></html>'
    with Path(path).open('x') as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--design", type=Path, default=ABSOLUTE_DESIGN)
    parser.add_argument("--reference-packets", type=Path)
    parser.add_argument("--repeat-fixed-map", action="append", type=Path, default=[])
    parser.add_argument("--repeat-end-to-end", action="append", type=Path, default=[])
    parser.add_argument("--repeat-cohort", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--phase", choices=("development", "fresh"), default="development")
    parser.add_argument("--resource-usage", type=Path)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--supplements", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--repeat", action="append", type=Path, default=[])
    parser.add_argument("--repeat-kind", choices=("fixed_map", "end_to_end"), default="end_to_end")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.absolute:
        if args.inputs is None:
            parser.error('--absolute requires --inputs')
        cohort = json.loads(args.repeat_cohort.read_text()) if args.repeat_cohort else []
        report = evaluate(args.candidate, inputs=args.inputs, design=args.design,
            references=args.references, reference_packets=args.reference_packets,
            repeats={'fixed_map': args.repeat_fixed_map, 'end_to_end': args.repeat_end_to_end},
            repeat_cohort=cohort, cost_baseline=args.baseline, phase=args.phase,
            resource_usage=args.resource_usage, controls=args.controls, selection=args.selection,
            supplements=args.supplements)
    else:
        if args.baseline is None:
            parser.error('historical comparison requires --baseline; use --absolute for candidate-only evaluation')
        report = compare(args.baseline, args.candidate, references=args.references,
                         repetitions=args.repeat, repeat_kind=args.repeat_kind)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if args.absolute:
        write_absolute_html(args.html or args.output.with_suffix('.html'), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

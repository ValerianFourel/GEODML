#!/usr/bin/env python3
"""Reconcile and publish existing JUPITER work. Never submit an allocation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_hour_sync import (
    Exchange,
    HubStore,
    atomic,
    checkpoint_files,
)
from analysis.interpretability.pipeline.agentic_hour_updates import (
    progress,
    publish_progress,
    replan,
)
from analysis.interpretability.pipeline.agentic_hours import canonical, inventory
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
from analysis.scripts.prepare_agentic_qwen_inputs import scheduler_gate
from analysis.scripts.prepare_horeka_qwen import immutable_json
from analysis.scripts.prepare_shared_hour_inputs import stage
from analysis.scripts.publish_agentic_dataset import build_manifest
from analysis.scripts.reconcile_agentic_dataset import reconcile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-inputs', type=Path)
    parser.add_argument('--keyword-priority', type=Path)
    parser.add_argument('--quota-evidence', type=Path)
    parser.add_argument('--since', required=True)
    parser.add_argument('--include-job-id', action='append', default=[])
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--stripes', type=int, default=256)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    args = parser.parse_args(argv)
    def scheduler():
        value = capture(plan={'plan_id': 'shared-hours-bootstrap'}, since=args.since,
                        include_job_ids=args.include_job_id)
        return {**value, 'cluster': 'jupiter'}
    snapshot = scheduler()
    scheduler_gate(snapshot)
    review = reconcile(args.source, scheduler_snapshot=snapshot, stripe_count=args.stripes, apply=False)
    tasks, done, blocked = inventory(args.source, stripes=args.stripes)
    configurations = {}
    for row in tasks:
        key = row['model'] + ':' + row['configuration_sha256']
        configurations[key] = configurations.get(key, 0) + 1
    print(json.dumps({'registered_tasks': len(tasks), 'verified_completed': len(done),
                      'blocked': len(blocked), 'configurations': configurations,
                      'reconciliation_actions': len(review['actions']),
                      'reconciliation_blocked': review['blocked'][:20],
                      'reconciliation_blocked_count': len(review['blocked'])}, indent=2))
    if not args.publish:
        return 0
    if not args.model_inputs or not args.keyword_priority or not args.quota_evidence:
        raise ValueError('publication requires frozen model inputs, keyword priority and fresh quota evidence')
    from analysis.scripts.manage_agentic_hours import health
    if not health({'dataset_root': str(args.source), 'cluster': 'jupiter'},
                  args.quota_evidence)['safe_to_admit']:
        raise ValueError('fresh safe JUPITER storage evidence required')
    repository = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain', '--untracked-files=all'], text=True):
        raise ValueError('publish requires a clean committed checkout')
    exchange = Exchange(HubStore(args.repo_id), args.output / 'journal')
    if exchange.snapshot()[1]['current_plan']:
        raise ValueError('shared plan already exists; use the saved site and update workflow')
    # Keep immutable bootstrap evidence; never overwrite the historical source plans.
    args.output.mkdir(parents=True, exist_ok=True)
    atomic(args.output / 'scheduler-before.json', canonical(snapshot))
    result = reconcile(args.source, scheduler_snapshot=snapshot, stripe_count=args.stripes, apply=True)
    atomic(args.output / 'reconciliation.json', canonical(result))
    if result['blocked']:
        raise ValueError('reconciliation blocked; no input publication attempted')
    root = args.output / 'dataset'
    stage(args.source, root, json.loads(args.model_inputs.read_bytes()), args.keyword_priority,
          scheduler(), stripes=args.stripes)
    scheduler_gate(scheduler())
    tasks, _, _ = inventory(root, stripes=args.stripes)
    _, outcomes = checkpoint_files(root, {t['fingerprint']: t for t in tasks}, stripes=args.stripes)
    bundle = exchange.upload(root, list(build_manifest(root)['files']), outcomes=outcomes,
                             metadata={'kind': 'frozen-inputs'})
    immutable_json(args.output / 'publication.json', {'input_bundle': bundle, 'source_commit': pin})
    calibration = args.output / 'calibration.json'
    immutable_json(calibration, json.loads(args.calibration.read_bytes()) if args.calibration else {})
    site = {'cluster': 'jupiter', 'dataset_root': str(root.resolve()), 'input_bundle': bundle,
            'calibration': str(calibration.resolve()), 'source_commit': pin,
            'plan_dir': str((args.output / 'plans').resolve()), 'journal': str((args.output / 'journal').resolve()),
            'quota_evidence': str(args.quota_evidence.resolve()), 'stripes': args.stripes, 'attempts': []}
    immutable_json(args.output / 'site.json', site)
    scheduler_gate(scheduler())
    _, state = exchange.snapshot()
    result = replan(exchange, site, state, {})
    revision, state = exchange.snapshot()
    plan = json.loads(exchange.store.read(f"coordination/plans/{state['current_plan']}.json", revision))
    report = progress(root, state, revision=revision, deferred=plan['deferred'],
                      prior_completed=plan['completed_before_plan'], stripes=args.stripes)
    publish_progress(exchange, report)
    print(json.dumps({'publication': result, 'input_bundle': bundle, 'site': str(args.output / 'site.json'),
                      'progress': report['cells'], 'allocation_submitted': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

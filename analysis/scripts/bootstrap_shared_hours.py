#!/usr/bin/env python3
"""Reconcile and publish existing JUPITER work. Never submit an allocation."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.interpretability.pipeline.agentic_audit_progress import (
    audit_progress,
    audit_stage,
)
from analysis.interpretability.pipeline.agentic_hour_sync import (
    REGISTRY_PATH,
    Exchange,
    HubStore,
    atomic,
    checkpoint_files,
)
from analysis.interpretability.pipeline.agentic_hour_updates import (
    ALLOWED_MODELS,
    PREFERRED_CLUSTER,
    markdown,
    progress,
    publish_progress,
    replan,
)
from analysis.interpretability.pipeline.agentic_hours import (
    canonical,
    digest,
    empty_registry,
    inventory,
)
from analysis.scripts.capture_agentic_scheduler_snapshot import capture
from analysis.scripts.prepare_agentic_qwen_inputs import scheduler_gate
from analysis.scripts.prepare_horeka_qwen import immutable_json
from analysis.scripts.prepare_shared_hour_inputs import stage
from analysis.scripts.publish_agentic_dataset import build_manifest
from analysis.scripts.reconcile_agentic_dataset import reconcile


@audit_stage("dispatch")
def publish_dispatch(source, exchange, snapshot, *, stripes=256):
    """Publish the complete audited inventory without copying input payloads."""
    audit_progress(phase="scheduler_and_hub")
    scheduler_gate(snapshot)
    revision, state = exchange.snapshot()
    if state != empty_registry():
        raise ValueError('shared registry already in use; use the saved site update workflow')
    audit_progress(phase="reconciliation")
    review = reconcile(source, scheduler_snapshot=snapshot, stripe_count=stripes, apply=False)
    if review['actions'] or review['blocked']:
        raise ValueError('dispatch requires reconciled legacy work')
    audit_progress(phase="inventory")
    tasks, done, blocked = inventory(source, stripes=stripes)
    from analysis.interpretability.pipeline.agentic_dataset import iter_sealed_rows
    bins = {p['prompt_id']: p.get('axis_bin') for p in iter_sealed_rows(source, 'prompts')}
    audit_progress(phase="dispatch_rows", tasks_total=len(tasks), tasks_prepared=0, verified_completed=len(done))
    rows = []
    for task in sorted(tasks, key=lambda t: (t['model'], t['priority_rank'], t['keyword_id'], t['prompt_id'], t['fingerprint'])):
        fp = task['fingerprint']
        status = 'verified_completed' if fp in done else 'blocked' if fp in blocked else 'awaiting_calibration_and_inputs'
        rows.append({**task, 'axis_bin': bins.get(task['prompt_id']), 'dispatch_status': status,
                     'preferred_cluster': PREFERRED_CLUSTER[task['model']],
                     'allowed_clusters': sorted(c for c, models in ALLOWED_MODELS.items() if task['model'] in models)})
        audit_progress(tasks_prepared=len(rows))
    audit_progress(phase="serialize_and_hash")
    document = {'format_version': 'geodml-dispatch-inventory-v1', 'runnable': False,
                'allocation_approved': False, 'hour_packages': None,
                'reference_hour': {'cluster': 'jupiter', 'nodes': 1, 'gpus': 4, 'gpu_type': 'GH200', 'seconds': 3600},
                'ordering': {'batch': 'keyword_priority_forward', 'interactive': 'keyword_priority_reverse',
                             'within_keyword': 'finish_remaining_prompt_cells'},
                'registered_tasks': len(tasks), 'verified_completed': len(done), 'blocked': len(blocked),
                'missing_models': sorted(set(PREFERRED_CLUSTER) - {t['model'] for t in tasks}),
                'tasks': rows}
    plan_id = digest(document)
    path = f'coordination/dispatch/{plan_id}.json.gz'
    summary = {k: v for k, v in document.items() if k != 'tasks'}
    summary.update(dispatch_id=plan_id, inventory_path=path, scheduler_snapshot=snapshot,
                   note='Full task dispatch inventory. Hour sizing and frozen input publication remain required. '
                        'Verified completions are observations, not transferred result payloads.')
    audit_progress(phase="check_existing_publication")
    if exchange.store.read(path, revision) is None:
        audit_progress(phase="compress_and_upload")
        result = exchange.store.commit(revision, {path: gzip.compress(canonical(document), mtime=0),
                                       'coordination/dispatch.json': canonical(summary)},
                                       'Publish audited keyword/bin dispatch inventory')
    else:
        result = revision
    return {**summary, 'published_revision': result, 'allocation_submitted': False}


def publish_tracker(source, exchange, snapshot, *, stripes=256):
    """Publish observations only; pending legacy jobs never become shared claims."""
    if (snapshot.get('cluster') != 'jupiter' or snapshot.get('complete') is not True
            or not 0 <= time.time() - snapshot.get('captured_at_epoch', 0) <= 120):
        raise ValueError('tracker requires a fresh complete JUPITER scheduler snapshot')
    revision, state = exchange.snapshot()
    if state != empty_registry():
        raise ValueError('shared registry already in use; use the saved site update workflow')
    tasks, _, _ = inventory(source, stripes=stripes)
    report = progress(source, state, revision=revision, stripes=stripes,
                      deferred={t['fingerprint']: 'awaiting_reconciliation' for t in tasks})
    report.update(tracker_only=True, runnable=False, observed_at_epoch=time.time(),
                  scheduler_snapshot=snapshot,
                  note='Observational counts only. No hour assignments. Reconcile legacy jobs, '
                       'publish frozen inputs and calibrate before planning runnable hours.')
    notice = ('# Tracker setup: no runnable hours\n\n' + report['note'] + '\n\n'
              + 'Legacy jobs at scheduler capture: ' + json.dumps(snapshot.get('jobs', [])) + '\n\n')
    result = exchange.store.commit(revision, {
        REGISTRY_PATH: canonical(state),
        'coordination/progress.json': canonical(report),
        'coordination/progress.md': notice.encode() + markdown(report),
    }, 'Initialize observational shared-hour tracker without runnable assignments')
    return {'tracker_revision': result, 'cells': report['cells'], 'task_count': report['task_count'],
            'legacy_jobs': snapshot.get('jobs', []), 'runnable_hours': 0, 'allocation_submitted': False}


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
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument('--publish', action='store_true')
    publication.add_argument('--publish-dispatch', action='store_true', help='publish the full audited task inventory without input payloads')
    publication.add_argument('--publish-tracker', action='store_true',
                             help='publish observations and an empty registry, even while legacy jobs are pending')
    parser.add_argument('--repo-id', default='ValerianFourel/geodml-experiment-v2-paper-private')
    args = parser.parse_args(argv)
    os.environ.setdefault("GEODML_AUDIT_PROGRESS", "1")
    @audit_stage("scheduler")
    def scheduler():
        value = capture(plan={'plan_id': 'shared-hours-bootstrap'}, since=args.since,
                        include_job_ids=args.include_job_id)
        return {**value, 'cluster': 'jupiter'}
    snapshot = scheduler()
    if args.publish_dispatch:
        result = publish_dispatch(args.source, Exchange(HubStore(args.repo_id), args.output / 'journal'),
                                  snapshot, stripes=args.stripes)
        print(json.dumps(result, indent=2))
        return 0
    if args.publish_tracker:
        result = publish_tracker(args.source, Exchange(HubStore(args.repo_id), args.output / 'journal'),
                                 snapshot, stripes=args.stripes)
        print(json.dumps(result, indent=2))
        return 0
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

"""Maintained launchers fail before preparation and never request namespaces."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.scripts import inference_network_namespace as network
from analysis.scripts import verify_inference_allocation as boundary
from analysis.scripts.inference_endpoint_security import EndpointSecurityError

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('cluster', ['jupiter', 'horeka'])
@pytest.mark.parametrize('exclusive', [True, False])
def test_existing_verifier_without_parent_export(monkeypatch, cluster, exclusive):
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.delenv('GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY', raising=False)
    monkeypatch.delenv(network.MARKER, raising=False)
    for key in ['SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID']:
        monkeypatch.delenv(key, raising=False)
    for key, value in {'SLURM_JOB_ID': '42', 'SLURM_JOB_NUM_NODES': '1',
                       'SLURM_STEP_NUM_NODES': '1', 'SLURM_JOB_NODELIST': 'compute-1'}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(network.sys, 'platform', 'linux')
    monkeypatch.setattr(network.shutil, 'which', lambda name: '/usr/bin/scontrol')
    monkeypatch.setattr(network.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=f'JobId=42 JobState=RUNNING NodeList=compute-1 Exclusive={"NODE" if exclusive else "USER"}', stderr=''))
    monkeypatch.setattr(boundary.subprocess, 'check_output', lambda *a, **kw: 'compute-1\n')
    monkeypatch.setattr(boundary.socket, 'gethostname', lambda: 'compute-1.example')
    monkeypatch.setattr(network, '_enter_private_namespace', lambda *a: pytest.fail('unshare invoked'))
    if not exclusive:
        with pytest.raises(EndpointSecurityError, match='whole_node_exclusivity'):
            boundary.verify(cluster)
        return
    receipt = boundary.verify(cluster)
    assert receipt['native_authentication_required'] is True
    assert receipt['loopback_transport_required'] is True
    assert os.environ['GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY'] == '1'
    assert os.environ['NCCL_SOCKET_IFNAME'] == 'lo'
    assert network.ensure_private_network_namespace(['worker']) == receipt
    monkeypatch.setattr(boundary.socket, 'gethostname', lambda: 'login-1')
    with pytest.raises(EndpointSecurityError, match='compute node'):
        boundary.verify(cluster)


@pytest.mark.parametrize('launcher', ['run_inference_wave_worker.sbatch', 'run_agentic_generator_existing_allocation.sh', 'run_agentic_bootstrap.sh'])
def test_shell_launchers_select_mode_after_setup_and_stop_early(tmp_path, launcher):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    calls = tmp_path / 'calls'
    # Site activation deliberately supplies the wrong setting; launcher must override it.
    venv = tmp_path / 'venv/bin'
    venv.mkdir(parents=True)
    (venv / 'activate').write_text('export GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY=0\n')
    setup = tmp_path / 'site.sh'
    setup.write_text(f'export ACL_ARR_VENV={tmp_path / "venv"}\n')
    bash_env = tmp_path / 'bashenv'
    bash_env.write_text('module() { :; }\nexport -f module\n')
    python = bin_dir / 'python3'
    python.write_text('#!/bin/bash\nprintf "%s %s\\n" "$GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY" "$*" >> "$CALLS"\nexit 42\n')
    python.chmod(0o755)
    for command in ['git', 'unshare']:
        path = bin_dir / command
        path.write_text('#!/bin/bash\necho UNEXPECTED >> "$CALLS"\nexit 99\n')
        path.chmod(0o755)
    controller = bin_dir / 'scontrol'
    controller.write_text('#!/bin/bash\necho NumNodes=1 NumCPUs=288 NodeList=compute-1 StartTime=unused EndTime=unused\n')
    controller.chmod(0o755)
    env = {**os.environ, 'SLURM_JOB_START_TIME': '100', 'SLURM_JOB_END_TIME': '3700', 'PATH': f'{bin_dir}:/usr/bin:/bin', 'BASH_ENV': str(bash_env),
           'CALLS': str(calls), 'ACL_ARR_ENVIRONMENT_FILE': str(setup), 'ACL_ARR_VENV': str(tmp_path / 'venv'),
           'SLURM_JOB_ID': '42', 'GEODML_EXECUTION_REPOSITORY': str(REPO), 'GEODML_EXECUTION_COMMIT': 'a' * 40,
           'GEODML_WAVE_ROOT': '/missing', 'GEODML_WAVE_OUTPUT_ROOT': '/missing', 'GEODML_WORKER_LAUNCHER': '/missing',
           'GEODML_APPROVED_WALLTIME': '01:00:00', 'GEODML_ALLOCATION_ESTIMATE': 'fixture'}
    env.pop('GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY', None)
    argv = ['bash', str(REPO / 'analysis/scripts/slurm/jupiter' / launcher)]
    if launcher == 'run_agentic_generator_existing_allocation.sh':
        argv += ['launch', '42', 'qwen38', '/missing', '/missing', '01:00:00', '4']
    if launcher == 'run_agentic_bootstrap.sh':
        argv += ['/must-not-run-preparation']
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 42, result.stderr
    suffix = ' --exec /must-not-run-preparation' if launcher == 'run_agentic_bootstrap.sh' else ''
    assert calls.read_text().splitlines() == [f'1 {REPO}/analysis/scripts/verify_inference_allocation.py --cluster jupiter{suffix}']


@pytest.mark.parametrize('mode', ['batch', 'interactive'])
def test_shared_hours_verify_before_reading_backlog_or_loading_models(tmp_path, monkeypatch, mode):
    from analysis.interpretability.pipeline import agentic_hour_runtime as runtime
    attempt = tmp_path / 'attempt.json'
    attempt.write_text(json.dumps({'cluster': 'jupiter', 'cluster_profile': {'cluster': 'jupiter', 'execution_boundary': 'exclusive-slurm-node'}, 'request': {'mode': mode}}))
    def denied(cluster, boundary_mode, **kwargs):
        assert (cluster, boundary_mode) == ('jupiter', 'exclusive-slurm-node')
        raise EndpointSecurityError('exclusive node unavailable')
    monkeypatch.setattr(runtime, 'verify_boundary', denied)
    monkeypatch.setattr(runtime, 'validate_request', lambda *args: pytest.fail('expensive path reached'))
    with pytest.raises(EndpointSecurityError, match='exclusive node unavailable'):
        runtime.execute(attempt, expected_job_id='42')


def test_prepared_resume_checks_boundary_before_any_artifact(monkeypatch):
    from analysis.scripts import run_jupiter_prepared_backlog as resume
    monkeypatch.setattr(resume, 'verify', lambda *args: (_ for _ in ()).throw(EndpointSecurityError('denied')))
    with pytest.raises(EndpointSecurityError, match='denied'):
        resume.main(['--runtime-environment', '/missing', '--scheduler-snapshot', '/missing'])


def test_horeka_does_not_accept_jupiter_scheduler_representation(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in ["SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in {"SLURM_JOB_ID": "42", "SLURM_JOB_NUM_NODES": "1",
                       "SLURM_JOB_NODELIST": "compute-1", "SLURM_STEP_NUM_NODES": "1",
                       "SLURM_JOB_CPUS_PER_NODE": "288"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(network.sys, "platform", "linux")
    monkeypatch.setattr(network.shutil, "which", lambda name: "/usr/bin/scontrol")
    monkeypatch.setattr(network.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="JobId=42 JobState=RUNNING NodeList=compute-1 NumNodes=1 "
        "NumCPUs=288 OverSubscribe=NO AllocTRES=cpu=288,node=1", stderr=""))
    monkeypatch.setattr(boundary.subprocess, "check_output", lambda *a, **kw: "compute-1")
    monkeypatch.setattr(boundary.socket, "gethostname", lambda: "compute-1")
    assert boundary.verify("jupiter")["status"] == "verified"
    with pytest.raises(EndpointSecurityError, match="whole_node_exclusivity"):
        boundary.verify("horeka")


def test_namespace_backend_requires_explicit_selection_and_command(monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []
    monkeypatch.setattr(network, "ensure_private_network_namespace", lambda command: calls.append(command) or {"mode": "fixture"})
    monkeypatch.setattr(boundary.os, "execvp", lambda *args: calls.append(args))
    assert boundary.main(["--cluster", "jupiter", "--boundary", "private-network-namespace"]) == 2
    assert calls == []
    assert boundary.main(["--cluster", "jupiter", "--boundary", "private-network-namespace", "--exec", "worker"]) == 0
    assert calls == [["worker"], ("worker", ["worker"])]
    assert os.environ["GEODML_ALLOW_EXCLUSIVE_SLURM_BOUNDARY"] == "0"

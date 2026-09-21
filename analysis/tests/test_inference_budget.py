"""The allocation clock includes model startup and never expands approved time."""

import pytest

from analysis.interpretability.pipeline.inference_budget import AllocationBudget


def test_unallocated_cpu_run_keeps_legacy_unbounded_behavior():
    budget = AllocationBudget.from_environment({})
    assert budget.can_start()
    assert budget.work_seconds_left() is None
    assert budget.record()["end_epoch"] is None


def test_admission_and_drain_use_actual_allocation_end(monkeypatch):
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.time", lambda: 1000.0)
    clock = [30.0]
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.monotonic", lambda: clock[0])
    budget = AllocationBudget.from_environment({"SLURM_JOB_END_TIME": "1600"})
    assert budget.can_start()
    assert budget.work_seconds_left() == 555
    clock[0] += 480
    assert not budget.can_start()
    assert budget.work_seconds_left() == 75
    clock[0] += 75
    assert budget.work_seconds_left() == 0
    assert budget.record()["stop_admission_epoch"] == 1480
    assert budget.record()["stop_work_epoch"] == 1555


def test_approved_cap_can_shorten_but_not_extend_slurm_deadline():
    env = {"SLURM_JOB_START_TIME": "1000", "SLURM_JOB_END_TIME": "20000",
           "GEODML_APPROVED_WALLTIME": "01:00:00"}
    assert AllocationBudget.from_environment(env).record()["end_epoch"] == 4600
    env["SLURM_JOB_END_TIME"] = "3000"
    assert AllocationBudget.from_environment(env).record()["end_epoch"] == 3000


def test_role_cap_can_shorten_but_not_extend_allocation():
    env = {"SLURM_JOB_END_TIME": "5000", "GEODML_ROLE_END_TIME": "2000"}
    assert AllocationBudget.from_environment(env).record()["end_epoch"] == 2000
    env["GEODML_ROLE_END_TIME"] = "9000"
    assert AllocationBudget.from_environment(env).record()["end_epoch"] == 5000


def test_budget_requires_real_slurm_deadline_for_batch_preflight():
    with pytest.raises(ValueError, match="SLURM_JOB_END_TIME"):
        AllocationBudget.from_environment({"GEODML_APPROVED_WALLTIME": "01:00:00"}, require=True)


@pytest.mark.parametrize("env", [
    {"SLURM_JOB_END_TIME": "nan"},
    {"SLURM_JOB_END_TIME": "infinite"},
    {"SLURM_JOB_END_TIME": "-1"},
    {"SLURM_JOB_END_TIME": "2000", "GEODML_START_MARGIN_SECONDS": "-1"},
    {"SLURM_JOB_END_TIME": "2000", "GEODML_CLEANUP_MARGIN_SECONDS": "nan"},
    {"SLURM_JOB_END_TIME": "2000", "GEODML_CLEANUP_MARGIN_SECONDS": "121"},
    {"SLURM_JOB_END_TIME": "2000", "GEODML_APPROVED_WALLTIME": "unlimited"},
])
def test_invalid_deadlines_and_margins_fail_closed(env):
    with pytest.raises(ValueError):
        AllocationBudget.from_environment(env)


def test_clock_jumps_do_not_extend_budget(monkeypatch):
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.time", lambda: 1000)
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.monotonic", lambda: 0)
    budget = AllocationBudget.from_environment({"SLURM_JOB_END_TIME": "1600"})
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.time", lambda: 10)
    monkeypatch.setattr("analysis.interpretability.pipeline.inference_budget.time.monotonic", lambda: 550)
    assert not budget.can_start()
    assert budget.work_seconds_left() == 5


def test_priority_yield_stops_admission_without_shortening_drain(tmp_path):
    marker = tmp_path / "stop"
    budget = AllocationBudget.from_environment({"GEODML_ADMISSION_STOP_FILE": str(marker)})
    assert budget.can_start()
    marker.touch()
    assert not budget.can_start()
    assert budget.admission_stop_reason() == "priority_yield"
    assert budget.work_seconds_left() is None

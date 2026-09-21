"""Sparse psslurm GPU environments must not bypass allocation checks."""

import pytest

from analysis.scripts.agentic_adaptive_gpu_resources import verify_gpu_resources

RECORD = (
    "JobId=1927510 JobState=RUNNING NumNodes=5 OverSubscribe=NO "
    "TresPerNode=gres/gpu:4"
)
GPUS = "GPU-a\nGPU-b\nGPU-c\nGPU-d\n"


def test_missing_gpu_variable_uses_exclusive_allocation_and_devices():
    assert verify_gpu_resources(RECORD, GPUS, {}, "1927510") == 4


@pytest.mark.parametrize("record,devices,environment", [
    (RECORD.replace("gres/gpu:4", "gres/gpu:1"), GPUS, {}),
    (RECORD.replace("OverSubscribe=NO", "OverSubscribe=YES"), GPUS, {}),
    (RECORD.replace("1927510", "123"), GPUS, {}),
    (RECORD.replace("RUNNING", "COMPLETED"), GPUS, {}),
    (RECORD, "GPU-a\n", {}),
    (RECORD, GPUS, {"SLURM_GPUS_ON_NODE": "1"}),
    (RECORD, GPUS, {"CUDA_VISIBLE_DEVICES": "0"}),
    (RECORD, GPUS, {"CUDA_VISIBLE_DEVICES": ""}),
])
def test_conflicting_or_insufficient_resources_fail(record, devices, environment):
    with pytest.raises(ValueError):
        verify_gpu_resources(record, devices, environment, "1927510")


def test_typed_gpu_request_and_visible_device_ids():
    assert verify_gpu_resources(
        RECORD.replace("gres/gpu:4", "gres/gpu:gh200:4"), GPUS,
        {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_GPUS_ON_NODE": "4"},
        "1927510",
    ) == 4


def test_total_allocated_gpus_when_per_node_field_is_absent():
    assert verify_gpu_resources(
        RECORD.replace("TresPerNode=gres/gpu:4", "AllocTRES=cpu=1440,gres/gpu=20"),
        GPUS, {}, "1927510",
    ) == 4

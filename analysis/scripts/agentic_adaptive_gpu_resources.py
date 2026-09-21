"""Verify four GPUs when JUPITER psslurm omits SLURM_GPUS_ON_NODE."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping


def verify_gpu_resources(
    job_record: str, devices: str, environment: Mapping[str, str], job_id: str,
) -> int:
    fields = dict(word.split("=", 1) for word in job_record.split() if "=" in word)
    if fields.get("JobId") != job_id or fields.get("JobState") != "RUNNING":
        raise ValueError("GPU gate: expected the running allocation " + job_id)
    if fields.get("NumNodes") != "5":
        raise ValueError("GPU gate: expected the approved five-node allocation")
    if fields.get("OverSubscribe") != "NO" and fields.get("Shared") != "0":
        raise ValueError("GPU gate: allocation must own its nodes exclusively")
    per_node = re.search(r"(?:^|,)gres/gpu(?::[^,:]+)?:([0-9]+)(?:,|$)",
                         fields.get("TresPerNode", ""))
    total = re.search(r"(?:^|,)gres/gpu=([0-9]+)(?:,|$)", fields.get("AllocTRES", ""))
    if per_node:
        allocated = int(per_node[1])
    elif total and int(total[1]) == 20:
        allocated = 4
    else:
        raise ValueError("GPU gate: scheduler does not report four GPUs per node: "
                         f"TresPerNode={fields.get('TresPerNode')} "
                         f"AllocTRES={fields.get('AllocTRES')}")
    uuids = {line.strip() for line in devices.splitlines() if line.strip()}
    if allocated != 4 or len(uuids) != 4:
        raise ValueError(f"GPU gate: allocated={allocated}, observed={len(uuids)}; need four")
    reported = environment.get("SLURM_GPUS_ON_NODE")
    if reported and reported != "4":
        raise ValueError("GPU gate: conflicting SLURM_GPUS_ON_NODE=" + reported)
    if "CUDA_VISIBLE_DEVICES" in environment:
        visible = {item.strip() for item in environment["CUDA_VISIBLE_DEVICES"].split(",")}
        if visible != {"0", "1", "2", "3"} and visible != uuids:
            raise ValueError("GPU gate: CUDA_VISIBLE_DEVICES does not expose all four GPUs")
    return 4


def main() -> None:
    job_id = os.environ["GEODML_EXPECTED_JOB_ID"]
    record = subprocess.check_output(
        ["scontrol", "show", "job", "--oneliner", job_id], text=True,
    )
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    )
    print(verify_gpu_resources(record, devices, os.environ, job_id))


if __name__ == "__main__":
    main()

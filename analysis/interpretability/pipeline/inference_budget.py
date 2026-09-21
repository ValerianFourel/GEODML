"""One allocation deadline shared by rolling inference and batch preflight.

Runtime metadata only: never include this budget in scientific task/resume
identities. Loading a model consumes the same allocation as executing requests.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def _seconds(value: str, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _walltime(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise ValueError("GEODML_APPROVED_WALLTIME must be [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError("approved wall-time must be positive")
    return total


@dataclass(frozen=True)
class AllocationBudget:
    end_epoch: float | None = None
    admission_margin_seconds: float = 120
    cleanup_margin_seconds: float = 45
    _end_monotonic: float | None = field(default=None, repr=False)
    admission_stop_file: Path | None = None

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None, *, require: bool = False,
    ) -> AllocationBudget:
        env = os.environ if environment is None else environment
        admission = _seconds(env.get("GEODML_START_MARGIN_SECONDS", "120"), "admission margin")
        cleanup = _seconds(env.get("GEODML_CLEANUP_MARGIN_SECONDS", "45"), "cleanup margin")
        if admission < cleanup:
            raise ValueError("admission margin must be at least the cleanup margin")
        raw_end = env.get("SLURM_JOB_END_TIME", "")
        raw_role_end = env.get("GEODML_ROLE_END_TIME", "")
        raw_approved = env.get("GEODML_APPROVED_WALLTIME", "")
        approved = _walltime(raw_approved) if raw_approved else None
        stop_file = Path(env["GEODML_ADMISSION_STOP_FILE"]) if env.get("GEODML_ADMISSION_STOP_FILE") else None
        if not raw_end:
            if require:
                raise ValueError("SLURM_JOB_END_TIME is required; do not reset a timer after model startup")
            return cls(admission_margin_seconds=admission, cleanup_margin_seconds=cleanup,
                       admission_stop_file=stop_file)
        end = _seconds(raw_end, "SLURM_JOB_END_TIME", positive=True)
        if raw_role_end:
            end = min(end, _seconds(raw_role_end, "GEODML_ROLE_END_TIME", positive=True))
        raw_start = env.get("SLURM_JOB_START_TIME", "")
        if raw_start:
            start = _seconds(raw_start, "SLURM_JOB_START_TIME", positive=True)
            if start > end:
                raise ValueError("Slurm job start is later than its end")
            if approved is not None:
                end = min(end, start + approved)
        return cls(
            end_epoch=end, admission_margin_seconds=admission,
            cleanup_margin_seconds=cleanup,
            _end_monotonic=time.monotonic() + end - time.time(),
            admission_stop_file=stop_file,
        )

    def can_start(self) -> bool:
        return self.admission_stop_reason() is None

    def admission_stop_reason(self) -> str | None:
        if self._end_monotonic is not None and (
            time.monotonic() >= self._end_monotonic - self.admission_margin_seconds
        ):
            return "allocation_deadline"
        if self.admission_stop_file is not None and self.admission_stop_file.exists():
            return "priority_yield"
        return None

    def work_seconds_left(self) -> float | None:
        if self._end_monotonic is None:
            return None
        return max(0.0, self._end_monotonic - self.cleanup_margin_seconds - time.monotonic())

    def record(self) -> dict[str, object]:
        return {
            "policy": "fill-approved-queue-v1",
            "end_epoch": self.end_epoch,
            "stop_admission_epoch": None if self.end_epoch is None else self.end_epoch - self.admission_margin_seconds,
            "stop_work_epoch": None if self.end_epoch is None else self.end_epoch - self.cleanup_margin_seconds,
            "admission_margin_seconds": self.admission_margin_seconds,
            "cleanup_margin_seconds": self.cleanup_margin_seconds,
            "queue_exhaustion": "stop_without_repeating_completed_work",
        }

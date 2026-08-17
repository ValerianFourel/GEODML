"""CPU-only tests for explicit multi-GPU loading contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from analysis.interpretability.utils import (
    multi_gpu_load_kwargs,
    validate_cuda_device_map,
)


def _fake_torch(gpu_count: int):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: gpu_count,
        get_device_properties=lambda index: SimpleNamespace(
            total_memory=40 * 1024**3
        ),
    )
    return SimpleNamespace(cuda=cuda, bfloat16="bfloat16", float16="float16")


class MultiGpuLoadingTests(unittest.TestCase):
    def test_balanced_four_gpu_contract_has_equal_budgets(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(4)}):
            kwargs = multi_gpu_load_kwargs(
                quantize=False,
                device_map_strategy="balanced",
                required_gpu_count=4,
            )
        self.assertEqual(kwargs["device_map"], "balanced")
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertNotIn("torch_dtype", kwargs)
        self.assertEqual(
            {key: kwargs["max_memory"][key] for key in range(4)},
            {0: "32GiB", 1: "32GiB", 2: "32GiB", 3: "32GiB"},
        )

    def test_four_gpu_contract_rejects_other_visible_count(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(2)}):
            with self.assertRaisesRegex(RuntimeError, "required exactly 4"):
                multi_gpu_load_kwargs(required_gpu_count=4)

    def test_invalid_device_map_strategy_is_rejected(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(4)}):
            with self.assertRaisesRegex(ValueError, "unsupported device-map"):
                multi_gpu_load_kwargs(device_map_strategy="not-a-strategy")

    def test_partial_automatic_placement_is_valid(self) -> None:
        used = validate_cuda_device_map(
            {"model.layers.0": 1, "model.layers.1": "cuda:2", "lm_head": 3},
            visible_gpu_count=4,
        )
        self.assertEqual(used, {1, 2, 3})

    def test_cpu_only_automatic_placement_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not use any CUDA GPU"):
            validate_cuda_device_map(
                {"model": "cpu"},
                visible_gpu_count=4,
            )

    def test_unavailable_device_in_automatic_placement_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unavailable CUDA GPUs"):
            validate_cuda_device_map(
                {"model": "cuda:4"},
                visible_gpu_count=4,
            )


if __name__ == "__main__":
    unittest.main()

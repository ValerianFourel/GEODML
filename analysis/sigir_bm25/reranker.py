"""One cross-encoder per GPU next to the vLLM server, with a CPU fallback.

The original run scored on CPU; on HoreKa that left the GPUs nearly idle (vLLM: 2-5 running requests, 1% KV cache)
while cells queued for the CPU reranker. The same model in fp32 on GPU reproduced the logged CPU shortlists for 1,000
of 1,000 validation answers (rebuttal gate G2, max score difference 7.7e-6). Each call takes a free GPU replica; a CUDA
out-of-memory error (vLLM holds most of each GPU) moves that call to the CPU replica. ``counts`` records the device
of every call, and the per-part summary carries it.
"""
from __future__ import annotations

import queue
from collections import Counter
from collections.abc import Callable, Sequence
from threading import Lock

from analysis.interpretability.pipeline.agentic_search import Snippet


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"


class DevicePoolScorer:
    """SnippetScorer over several replicas (one per device); a call waits for a free replica."""

    deterministic = True

    def __init__(self, replicas: Sequence, fallback_factory: Callable[[], object] | None = None,
                 on_oom: Callable[[], None] | None = None) -> None:
        if not replicas:
            raise ValueError("at least one replica is required")
        if any(getattr(r, "deterministic", False) is not True for r in replicas):
            raise ValueError("every replica must declare deterministic=True")
        self.model_id = replicas[0].model_id
        self.model_revision = replicas[0].model_revision
        self._free: queue.Queue = queue.Queue()
        for i, r in enumerate(replicas):
            self._free.put((i, r))
        self._fallback_factory, self._fallback, self._on_oom = fallback_factory, None, on_oom
        self._lock = Lock()
        self.counts: Counter[str] = Counter()

    def _cpu(self):
        with self._lock:
            if self._fallback is None:
                if self._fallback_factory is None:
                    raise RuntimeError("no CPU fallback configured")
                self._fallback = self._fallback_factory()
            return self._fallback

    def score(self, query: str, snippets: Sequence[Snippet]) -> list[float]:
        i, replica = self._free.get()
        try:
            values = list(replica.score(query, list(snippets)))
            device = f"gpu{i}"
        except Exception as error:  # noqa: BLE001 - only out-of-memory falls back; anything else propagates
            if not _is_oom(error):
                raise
            if self._on_oom is not None:
                self._on_oom()
            values, device = None, "cpu_after_oom"
        finally:
            self._free.put((i, replica))
        if values is None:
            values = list(self._cpu().score(query, list(snippets)))
        with self._lock:
            self.counts[device] += 1
        return values


def cuda_pool(path: str, batch_size: int = 32) -> DevicePoolScorer:
    """bge-reranker replicas on every visible GPU (fp32, as the original), CPU fallback on out-of-memory."""
    import torch

    from analysis.interpretability.pipeline.agentic_search import SentenceTransformersCrossEncoderScorer

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("--device cuda-pool needs at least one visible GPU")
    make = lambda device: SentenceTransformersCrossEncoderScorer(  # noqa: E731
        path, model_revision=None, device=device, batch_size=batch_size, local_files_only=True)
    return DevicePoolScorer([make(f"cuda:{i}") for i in range(n)], fallback_factory=lambda: make("cpu"),
                            on_oom=torch.cuda.empty_cache)

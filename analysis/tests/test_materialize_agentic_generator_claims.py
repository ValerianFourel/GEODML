"""Historical generator materialization opens only after every shared claim succeeds."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from analysis.interpretability.pipeline.agentic_search import (
    ContextCompactor,
    LexicalOverlapScorer,
)
from analysis.scripts.materialize_agentic_generator_claims import require_complete
from analysis.scripts.run_agentic_search_integration_smoke import run_smoke
from analysis.tests.test_run_agentic_search_integration_smoke import (
    _FakeClientContext,
    _smoke_inputs,
)


def test_complete_shared_registry_passes_without_writing_materialized_output(tmp_path):
    inputs = replace(_smoke_inputs(tmp_path), shared_claim_root=tmp_path / "claims")
    asyncio.run(
        run_smoke(
            inputs,
            client_context=_FakeClientContext(),
            compactor=ContextCompactor(LexicalOverlapScorer()),
        )
    )
    target = replace(inputs, output=tmp_path / "not-created")
    assert require_complete(target) == 12
    assert not target.output.exists()


def test_missing_shared_claim_closes_materialization_barrier(tmp_path):
    inputs = replace(_smoke_inputs(tmp_path), shared_claim_root=tmp_path / "claims")
    with pytest.raises(RuntimeError, match='"missing": 12'):
        require_complete(inputs)

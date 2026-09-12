from __future__ import annotations

from collections import Counter
import hashlib

import pytest

from analysis.scripts.select_readiness_axis_pilot import select_candidates


def _population() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prompts = []
    axes = []
    rank = 0
    for keyword_index in range(5):
        for axis_bin in range(4):
            for copy in range(3):
                candidate_id = f"p-{keyword_index}-{axis_bin}-{copy}"
                question = f"Question {candidate_id}?"
                digest = hashlib.sha256(question.encode()).hexdigest()
                percentile = (axis_bin + (copy + 1) / 4) / 4
                prompts.append({
                    "candidate_id": candidate_id,
                    "keyword_id": f"keyword-{keyword_index}",
                    "keyword": f"topic {keyword_index}",
                    "question": question,
                    "question_sha256": digest,
                    "target_normalized_axis_1": percentile,
                })
                axes.append({
                    "candidate_id": candidate_id,
                    "text_sha256": digest,
                    "axis_1_rank": rank,
                    "axis_1_percentile_0_1": percentile,
                    "consensus_axis_1_z": percentile,
                })
                rank += 1
    return prompts, axes


def test_selection_is_deterministic_and_balanced() -> None:
    prompts, axes = _population()

    first, diagnostics = select_candidates(
        prompts,
        axes,
        sample_size=20,
        axis_bins=4,
        master_seed=17,
    )
    repeated, _ = select_candidates(
        prompts,
        axes,
        sample_size=20,
        axis_bins=4,
        master_seed=17,
    )

    assert [row.candidate_id for row in first] == [
        row.candidate_id for row in repeated
    ]
    assert Counter(row.axis_bin for row in first) == {0: 5, 1: 5, 2: 5, 3: 5}
    assert set(Counter(row.keyword_id for row in first).values()) == {4}
    assert diagnostics["sample_size"] == 20
    assert diagnostics["keyword_count"] == 5


def test_selection_rejects_identity_mismatch() -> None:
    prompts, axes = _population()
    axes.pop()

    with pytest.raises(ValueError, match="candidate ID sets differ"):
        select_candidates(
            prompts,
            axes,
            sample_size=20,
            axis_bins=4,
            master_seed=17,
        )

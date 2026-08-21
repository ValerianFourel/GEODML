"""CPU-only contracts for LLM2Vec-Gen latent readiness feedback."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from analysis.interpretability.pipeline.readiness_latent_feedback import (
    anchor_exact_keyword_question,
    fit_latent_coordinate_bridge,
    run_latent_feedback,
    steer_reconstruction_state,
)
from analysis.interpretability.pipeline.readiness_hf_dataset import atomic_jsonl
from analysis.interpretability.pipeline.readiness_prompt_population import (
    ReadinessSubspaceBounds,
)
from analysis.scripts.generate_llm2vec_gen_feedback_proposals import (
    _join_calibration,
)
from analysis.scripts import generate_llm2vec_gen_feedback_proposals


class _FakeReconstructionBackend:
    model_name = "fake-llm2vec-gen"

    def encode(self, texts, *, batch_size, max_length):
        del batch_size, max_length
        states = []
        for text in texts:
            lower = text.lower()
            axis_1 = 1.0 if "implement" in lower else 0.5 if "compare" in lower else 0.0
            axis_2 = 1.0 if "execute" in lower else 0.5 if "choose" in lower else 0.0
            states.append(np.array([[axis_1, axis_2, 0.25]], dtype=np.float32))
        return np.zeros((len(texts), 2), dtype=np.float32), np.stack(states)

    def decode(self, state, *, max_new_tokens):
        del max_new_tokens
        axis_1, axis_2 = np.asarray(state)[0, :2]
        stage = "implement" if axis_1 >= 0.75 else "compare" if axis_1 >= 0.25 else "understand"
        mode = "execute" if axis_2 >= 0.75 else "choose" if axis_2 >= 0.25 else "research"
        # Deliberately omit the keyword.  The deterministic anchor must restore it.
        return f"How can a team {stage} and {mode} the best approach?"


class _FakeFrozenScorer:
    model_name = "fake-frozen-llm2vec"

    def score(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            axis_1 = 1.0 if "implement" in lower else 0.5 if "compare" in lower else 0.0
            axis_2 = 1.0 if "execute" in lower else 0.5 if "choose" in lower else 0.0
            rows.append((axis_1, axis_2))
        return np.asarray(rows, dtype=np.float64)


class _FakeValidator:
    model_name = "independent-fake-judge"

    def review(self, question, keyword):
        accepted = (
            keyword in question
            and question.endswith("?")
            and "how" in question.lower()
        )
        return accepted, "accepted" if accepted else "not a search question"


class _UnusedEmbedder:
    model_name = "unused-fake-embedder"

    def __init__(self, *args, **kwargs):
        del args, kwargs


class ReadinessLatentFeedbackTests(unittest.TestCase):
    def _bridge(self):
        coordinates = np.asarray(
            [
                (0.0, 0.0),
                (0.0, 0.5),
                (0.0, 1.0),
                (0.5, 0.0),
                (0.5, 0.5),
                (0.5, 1.0),
                (1.0, 0.0),
                (1.0, 0.5),
                (1.0, 1.0),
                (0.25, 0.75),
            ],
            dtype=np.float64,
        )
        states = np.zeros((len(coordinates), 1, 3), dtype=np.float64)
        states[:, 0, :2] = coordinates
        states[:, 0, 2] = np.linspace(-0.2, 0.2, len(coordinates))
        return fit_latent_coordinate_bridge(states, coordinates, ridge_penalty=1e-6)

    def test_bridge_recovers_two_coordinate_directions(self):
        bridge = self._bridge()
        self.assertEqual(bridge.state_shape, (1, 3))
        self.assertEqual(bridge.calibration_item_count, 10)
        self.assertAlmostEqual(bridge.directions[0, 0, 0], 1.0, places=4)
        self.assertAlmostEqual(bridge.directions[1, 0, 1], 1.0, places=4)
        self.assertLess(abs(bridge.directions[0, 0, 1]), 1e-4)
        self.assertLess(abs(bridge.directions[1, 0, 0]), 1e-4)

    def test_steering_is_bounded_and_preserves_residual_state(self):
        bridge = self._bridge()
        state = np.array([[0.0, 0.0, 7.0]])
        proposed, step = steer_reconstruction_state(
            state,
            bridge,
            observed_coordinates=(0.0, 0.0),
            target_coordinates=(1.0, 1.0),
            coordinate_step_limit=0.25,
        )
        np.testing.assert_allclose(step, (0.25, 0.25))
        self.assertAlmostEqual(proposed[0, 0], 0.25, places=4)
        self.assertAlmostEqual(proposed[0, 1], 0.25, places=4)
        self.assertGreater(proposed[0, 2], 6.9)
        outside_proposed, outside_step = steer_reconstruction_state(
            state,
            bridge,
            observed_coordinates=(-0.2, 1.2),
            target_coordinates=(0.0, 1.0),
            coordinate_step_limit=0.25,
        )
        self.assertTrue(np.isfinite(outside_proposed).all())
        np.testing.assert_allclose(outside_step, (0.2, -0.2))

    def test_keyword_anchor_is_deterministic_and_hard_validated(self):
        question = anchor_exact_keyword_question(
            "How can a team compare practical approaches?",
            "abandoned cart recovery",
        )
        self.assertEqual(
            question,
            "Regarding abandoned cart recovery, how can a team compare practical approaches?",
        )
        json_question = anchor_exact_keyword_question(
            '{"question":"How can a team compare \\"available\\" approaches?"}',
            "abandoned cart recovery",
        )
        self.assertIn('compare "available" approaches', json_question)

    def test_closed_loop_reembeds_and_reaches_target(self):
        keyword = "abandoned cart recovery"
        result = run_latent_feedback(
            initial_question=(
                "How can a team understand abandoned cart recovery through online research?"
            ),
            keyword=keyword,
            target_coordinates=(1.0, 1.0),
            bridge=self._bridge(),
            reconstruction_backend=_FakeReconstructionBackend(),
            scorer=_FakeFrozenScorer(),
            validator=_FakeValidator(),
            maximum_rounds=4,
            step_scales=(1.0,),
            coordinate_step_limit=0.5,
            distance_tolerance=0.01,
        )
        self.assertTrue(result.accepted_within_tolerance)
        self.assertEqual(result.stop_reason, "decoded-question-within-tolerance")
        self.assertIn(keyword, result.best_question)
        self.assertEqual(result.best_normalized_axis_1, 1.0)
        self.assertEqual(result.best_normalized_axis_2, 1.0)
        self.assertEqual(result.completed_round_count, 2)
        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(all(row.hard_valid for row in result.attempts))
        self.assertTrue(all(row.semantic_valid for row in result.attempts))

    def test_bridge_rejects_rank_deficient_calibration(self):
        coordinates = np.tile((0.5, 0.5), (10, 1))
        with self.assertRaisesRegex(ValueError, "do not identify two directions"):
            fit_latent_coordinate_bridge(
                np.zeros((10, 1, 3)), coordinates, ridge_penalty=1e-3
            )

    def test_existing_corpus_and_frozen_coordinates_form_development_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus.jsonl"
            coordinates = root / "coordinates.jsonl"
            atomic_jsonl(
                corpus,
                (
                    {
                        "item_id": f"item:{index}",
                        "split": "development" if index < 12 else "confirmation",
                        "text": f"Search question number {index} for calibration",
                    }
                    for index in range(13)
                ),
            )
            atomic_jsonl(
                coordinates,
                (
                    {
                        "item_id": f"item:{index}",
                        "split": "development" if index < 12 else "confirmation",
                        "axis_1": -1.0 + 2.0 * (index % 4) / 3.0,
                        "axis_2": -1.0 + 2.0 * (index // 4) / 2.0,
                        "usable_for_axis": index != 0,
                    }
                    for index in range(13)
                ),
            )
            bounds = ReadinessSubspaceBounds(
                axis_1_low=-1.0,
                axis_1_high=1.0,
                axis_2_low=-1.0,
                axis_2_high=1.0,
                lower_quantile=0.05,
                upper_quantile=0.95,
                reference_split="development",
                reference_item_count=12,
            )
            rows = _join_calibration(
                corpus,
                coordinates,
                bounds,
                minimum_items=10,
                maximum_items=10,
            )
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["split"] == "development" for row in rows))
            self.assertNotIn("item:0", {row["item_id"] for row in rows})
            self.assertTrue(
                all(
                    0.0 <= row[name] <= 1.0
                    for row in rows
                    for name in ("normalized_axis_1", "normalized_axis_2")
                )
            )

    def test_command_line_mock_smoke_writes_proposal_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                name: root / f"{name}.jsonl"
                for name in ("tasks", "candidates", "corpus", "coordinates")
            }
            target = {
                "target_id": "readiness-support-target:000",
                "target_index": 0,
                "axis_1_index": 0,
                "axis_2_index": 0,
                "normalized_axis_1": 1.0,
                "normalized_axis_2": 1.0,
                "raw_axis_1": 1.0,
                "raw_axis_2": 1.0,
            }
            atomic_jsonl(
                paths["tasks"],
                [
                    {
                        "task_id": "task:one",
                        "keyword_id": "keyword:one",
                        "keyword": "abandoned cart recovery",
                        "target": target,
                        "round_index": 0,
                        "generator_id": "seed-generator",
                        "generation_seed": 17,
                        "requested_candidate_count": 1,
                        "feedback": "No earlier candidate",
                    }
                ],
            )
            initial_question = (
                "How can a team understand abandoned cart recovery through online research?"
            )
            atomic_jsonl(
                paths["candidates"],
                [
                    {
                        "candidate_id": "candidate:one",
                        "task_id": "task:one",
                        "keyword_id": "keyword:one",
                        "keyword": "abandoned cart recovery",
                        "target_id": target["target_id"],
                        "target_index": 0,
                        "target_normalized_axis_1": 1.0,
                        "target_normalized_axis_2": 1.0,
                        "target_raw_axis_1": 1.0,
                        "target_raw_axis_2": 1.0,
                        "round_index": 0,
                        "generator_id": "seed-generator",
                        "generator_model": "seed-model",
                        "candidate_slot": 0,
                        "generation_seed": 17,
                        "question": initial_question,
                        "question_sha256": "0" * 64,
                        "proposal_kind": "causal-lm",
                    }
                ],
            )
            calibration = []
            for axis_1 in (0.0, 0.5, 1.0):
                for axis_2 in (0.0, 0.5, 1.0):
                    stage = (
                        "implement"
                        if axis_1 == 1
                        else "compare" if axis_1 == 0.5 else "understand"
                    )
                    mode = (
                        "execute"
                        if axis_2 == 1
                        else "choose" if axis_2 == 0.5 else "research"
                    )
                    calibration.append(
                        (f"How can a team {stage} and {mode} this topic?", axis_1, axis_2)
                    )
            calibration.append(
                ("How can a team compare and execute this topic?", 0.5, 1.0)
            )
            atomic_jsonl(
                paths["corpus"],
                (
                    {
                        "item_id": f"item:{index}",
                        "split": "development",
                        "text": text,
                    }
                    for index, (text, _, _) in enumerate(calibration)
                ),
            )
            atomic_jsonl(
                paths["coordinates"],
                (
                    {
                        "item_id": f"item:{index}",
                        "split": "development",
                        "axis_1": axis_1,
                        "axis_2": axis_2,
                        "usable_for_axis": True,
                    }
                    for index, (_, axis_1, axis_2) in enumerate(calibration)
                ),
            )
            bounds = root / "bounds.json"
            bounds.write_text(
                json.dumps(
                    {
                        "axis_1_low": 0.0,
                        "axis_1_high": 1.0,
                        "axis_2_low": 0.0,
                        "axis_2_high": 1.0,
                        "lower_quantile": 0.05,
                        "upper_quantile": 0.95,
                        "reference_split": "development",
                        "reference_item_count": 10,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fitted_map = root / "map.json"
            fitted_map.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            argv = [
                "generate_llm2vec_gen_feedback_proposals.py",
                "--tasks",
                str(paths["tasks"]),
                "--initial-candidates",
                str(paths["candidates"]),
                "--calibration-corpus",
                str(paths["corpus"]),
                "--calibration-coordinates",
                str(paths["coordinates"]),
                "--map",
                str(fitted_map),
                "--bounds",
                str(bounds),
                "--embedding-model",
                "fake-embedding",
                "--llm2vec-gen-model",
                "fake-gen",
                "--judge-model",
                "fake-judge",
                "--judge-cache-dir",
                str(root / "judge-cache"),
                "--output-dir",
                str(output),
                "--step-scales",
                "1.0",
                "--coordinate-step-limit",
                "0.5",
                "--distance-tolerance",
                "0.01",
            ]
            fake_map = SimpleNamespace(
                embedding_model="fake-embedding@revision", map_id="fake-map"
            )
            script = generate_llm2vec_gen_feedback_proposals
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    script, "load_readiness_embedding_map", return_value=fake_map
                ),
                mock.patch.object(
                    script,
                    "LLM2VecGenBackend",
                    return_value=_FakeReconstructionBackend(),
                ),
                mock.patch.object(script, "LLM2VecPromptEmbedder", _UnusedEmbedder),
                mock.patch.object(
                    script, "FrozenReadinessScorer", return_value=_FakeFrozenScorer()
                ),
                mock.patch.object(
                    script, "IndependentValidatorAdapter", return_value=_FakeValidator()
                ),
                mock.patch.object(
                    script.LocalSearchQuestionValidator,
                    "from_model",
                    return_value=object(),
                ),
            ):
                self.assertEqual(script.main(), 0)
            for name in (
                "feedback_proposals.jsonl",
                "feedback_trace.jsonl",
                "bridge_state.restricted-local.npz",
                "run_manifest.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            proposals = (output / "feedback_proposals.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn("abandoned cart recovery", proposals)


if __name__ == "__main__":
    unittest.main()

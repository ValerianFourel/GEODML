"""CPU contracts for local generation input preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace


try:
    import dotenv  # noqa: F401
except ImportError:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda: None)

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from analysis.interpretability.utils import (
    _chat_template_tokenization_kwargs,
    _generation_sampling_kwargs,
    _local_tokenizer_compatibility_kwargs,
)


class LocalRankerGenerationTests(unittest.TestCase):
    def test_chat_template_always_requests_attention_mask(self) -> None:
        self.assertEqual(
            _chat_template_tokenization_kwargs(None),
            {"return_dict": True, "return_attention_mask": True},
        )

    def test_structured_template_options_are_preserved(self) -> None:
        self.assertEqual(
            _chat_template_tokenization_kwargs({"enable_thinking": False}),
            {
                "enable_thinking": False,
                "return_dict": True,
                "return_attention_mask": True,
            },
        )

    def test_deterministic_generation_neutralizes_sampling_defaults(self) -> None:
        self.assertEqual(
            _generation_sampling_kwargs(0.0),
            {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
        )

    def test_positive_temperature_enables_sampling(self) -> None:
        self.assertEqual(
            _generation_sampling_kwargs(0.25),
            {"do_sample": True, "temperature": 0.25, "top_p": 1.0},
        )

    def test_negative_temperature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            _generation_sampling_kwargs(-0.1)

    def test_legacy_extra_special_token_list_is_translated_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory)
            config = {
                "extra_special_tokens": ["<image_soft_token>", "<unused_token>"],
                "tokenizer_class": "GemmaTokenizerFast",
            }
            path = model / "tokenizer_config.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            self.assertEqual(
                _local_tokenizer_compatibility_kwargs(model),
                {
                    "additional_special_tokens": config["extra_special_tokens"],
                    "extra_special_tokens": {},
                },
            )
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), config)

    def test_mapping_extra_special_tokens_need_no_compatibility_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Path(temporary_directory)
            (model / "tokenizer_config.json").write_text(
                json.dumps({"extra_special_tokens": {"boi_token": "<start_of_image>"}}),
                encoding="utf-8",
            )
            self.assertEqual(_local_tokenizer_compatibility_kwargs(model), {})


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for local generation input preparation."""

from __future__ import annotations

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

from analysis.interpretability.utils import _chat_template_tokenization_kwargs


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


if __name__ == "__main__":
    unittest.main()

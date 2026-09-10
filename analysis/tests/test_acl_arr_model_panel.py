import json
from pathlib import Path
import unittest


class AclArrModelPanelTests(unittest.TestCase):
    def test_every_model_uses_the_common_2048_answer_token_budget(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        payload = json.loads(
            (repository / "analysis/config/acl_arr_model_panel.template.json").read_text()
        )

        self.assertEqual(len(payload["models"]), 4)
        self.assertEqual(
            {model["answer_max_tokens"] for model in payload["models"]},
            {2048},
        )


if __name__ == "__main__":
    unittest.main()

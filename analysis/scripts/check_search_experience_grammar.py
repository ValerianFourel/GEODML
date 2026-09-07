#!/usr/bin/env python3
"""Check serving schemas with the installed vLLM backend, without model loading."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
    from analysis.interpretability.pipeline.search_experience import answer_schema, judge_schema
    from analysis.scripts.run_acl_arr_vllm import _rerank_schema

    for name, schema in (("rerank", _rerank_schema(10)),
                         ("answer", answer_schema()), ("judge", judge_schema())):
        validate_xgrammar_grammar(SamplingParams(
            structured_outputs=StructuredOutputsParams(json=schema)))
        print(f"SEARCH_GRAMMAR=PASS stage={name}", flush=True)
    print("No model weights loaded; inference not tested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

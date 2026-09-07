"""Read-only native Mistral loading preflight; never constructs a model."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "mistralai/Mistral-Small-4-119B-2603"
REVISION = "a11f36bebf709121056b1dbcc943d1c6afbe494d"


def check(snapshot, *, config_loader, supported_architectures, config_only=False):
    snapshot = Path(snapshot)
    params = snapshot / "params.json"
    if not params.is_file():
        raise ValueError("native params.json missing from pinned snapshot; no HF fallback")
    weights = sorted(snapshot.glob("consolidated*.safetensors"))
    present = bool(weights) and all(p.is_file() and p.stat().st_size > 0 for p in weights)
    if not config_only and not present:
        raise ValueError("native consolidated safetensors missing; no download or weight substitution")
    original = json.loads((snapshot / "config.json").read_text())
    config = config_loader(str(snapshot), trust_remote_code=False,
                           revision=REVISION, config_format="mistral")
    text = getattr(config, "text_config", config)
    for obj in (config, text):
        architectures = getattr(obj, "architectures", None)
        if not architectures or not any(a in supported_architectures for a in architectures):
            raise ValueError(f"native architecture unresolved or unregistered: {architectures}")
    # Verify core dimensions against the same snapshot's HF representation.
    for field in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                  "n_routed_experts", "num_experts_per_tok", "kv_lora_rank", "q_lora_rank"):
        expected = original["text_config"][field]
        if getattr(text, field, None) != expected:
            raise ValueError(f"native/HF configuration disagreement: {field}")
    return {"model_id": MODEL, "revision": REVISION, "snapshot": str(snapshot),
            "config_format": "mistral", "load_format": "mistral", "tokenizer_mode": "mistral",
            "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
            "hf_config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
            "native_config": config.to_dict(),
            "config_only": config_only, "native_weights_present": present,
            "weight_files": [{"name": p.name, "size_bytes": p.stat().st_size} for p in weights if p.is_file()],
            "weights_loaded": False, "gpu_compatibility_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshots", type=Path, required=True)
    parser.add_argument("--config-only", action="store_true", help="Inspect configuration even if native weights are absent; not serving readiness")
    parser.add_argument("--summary", action="store_true", help="Print concise status without piping JSON")
    args = parser.parse_args()
    rows = [r for r in json.loads(args.model_snapshots.read_text())["models"]
            if r["model_id"] == MODEL and r["revision"] == REVISION]
    if len(rows) != 1:
        raise ValueError("expected exactly one pinned Mistral snapshot")
    from vllm.transformers_utils.config import get_config
    from vllm.model_executor.models import ModelRegistry
    result = check(rows[0]["snapshot"], config_loader=get_config,
                   supported_architectures=ModelRegistry.get_supported_archs(), config_only=args.config_only)
    if args.summary:
        config = result["native_config"]
        text = config.get("text_config", config)
        print("CONFIG_CHECK=PASS")
        print("TOP_ARCH=" + str(config.get("architectures")))
        print("TEXT_ARCH=" + str(text.get("architectures")))
        print("NATIVE_WEIGHTS_PRESENT=" + str(result["native_weights_present"]))
        print("CONFIG_ONLY=" + str(args.config_only))
        print("GPU_COMPATIBILITY_VERIFIED=False")
    else:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

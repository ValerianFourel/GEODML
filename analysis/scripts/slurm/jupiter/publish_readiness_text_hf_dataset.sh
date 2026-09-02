#!/bin/bash -l
# Verify and privately publish one finalized AxisGEO text dataset.
# This wrapper never requests Slurm resources and never prints the HF token.

set -euo pipefail
umask 077

: "${GEODML_EXPECTED_COMMIT:?Set the exact pushed Git commit}"
: "${GEODML_REPOSITORY:?Set the exact clean repository checkout}"
: "${GEODML_LLM2VEC_EXPORT_VENV:?Set the existing export environment}"
: "${READINESS_TEXT_DATASET_ROOT:?Set the finalized unified dataset}"
: "${READINESS_HF_REPO_ID:?Set the private Hugging Face dataset repository}"
: "${READINESS_HF_CONFIRM_REPO_ID:?Repeat the exact repository identifier}"
: "${READINESS_HF_PUBLISH_RECEIPT:?Set a fresh publication receipt path}"

[[ "$READINESS_HF_REPO_ID" == "$READINESS_HF_CONFIRM_REPO_ID" ]] || {
    echo "Hugging Face repository confirmation differs" >&2
    exit 2
}
[[ "$(git -C "$GEODML_REPOSITORY" rev-parse HEAD)" == "$GEODML_EXPECTED_COMMIT" ]]
[[ -z "$(git -C "$GEODML_REPOSITORY" status --porcelain)" ]]
[[ -x "$GEODML_LLM2VEC_EXPORT_VENV/bin/python" ]]
[[ -s "$READINESS_TEXT_DATASET_ROOT/dataset_manifest.json" ]]
[[ -s "$READINESS_TEXT_DATASET_ROOT/checksums.json" ]]
[[ ! -e "$READINESS_HF_PUBLISH_RECEIPT" ]]

jupiter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$jupiter_dir/readiness_jupiter_runtime.sh"
readiness_bootstrap_jupiter_control_runtime \
    "HF_PRIVATE_PUBLISH_CONTROL_RUNTIME=PASS"
source "$GEODML_LLM2VEC_EXPORT_VENV/bin/activate"

if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(python -c '
from huggingface_hub import get_token
token = get_token()
if not token:
    raise SystemExit("No cached Hugging Face token; run hf auth login first")
print(token)
')"
    export HF_TOKEN
fi

python -c '
from huggingface_hub import HfApi
identity = HfApi(token=__import__("os").environ["HF_TOKEN"]).whoami()
print("HF_ACCOUNT=" + str(identity["name"]))
'

cd "$GEODML_REPOSITORY"
python "$GEODML_REPOSITORY/analysis/scripts/build_readiness_text_hf_dataset.py" verify \
    --dataset-dir "$READINESS_TEXT_DATASET_ROOT"

python "$GEODML_REPOSITORY/analysis/scripts/build_readiness_text_hf_dataset.py" publish \
    --dataset-dir "$READINESS_TEXT_DATASET_ROOT" \
    --repo-id "$READINESS_HF_REPO_ID" \
    --confirm-repo-id "$READINESS_HF_CONFIRM_REPO_ID"

python - "$READINESS_TEXT_DATASET_ROOT" "$READINESS_HF_REPO_ID" \
    "$READINESS_HF_PUBLISH_RECEIPT" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
import pathlib
import sys

from huggingface_hub import HfApi, hf_hub_download

dataset_root = pathlib.Path(sys.argv[1]).resolve()
repo_id = sys.argv[2]
receipt = pathlib.Path(sys.argv[3]).resolve()
token = os.environ["HF_TOKEN"]
info = HfApi(token=token).dataset_info(
    repo_id,
    revision="main",
    files_metadata=True,
)
verified_files = {}
for name in ("dataset_manifest.json", "checksums.json"):
    local = dataset_root / name
    remote = pathlib.Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=name,
            repo_type="dataset",
            revision=info.sha,
            token=token,
        )
    )
    local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
    remote_sha = hashlib.sha256(remote.read_bytes()).hexdigest()
    if local_sha != remote_sha:
        raise SystemExit(f"remote {name} differs from the uploaded local artifact")
    verified_files[name] = local_sha
manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
payload = {
    "format_version": "axisgeo-hf-private-publication-receipt-v1",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "repo_id": repo_id,
    "visibility": "private",
    "revision": info.sha,
    "dataset_root": str(dataset_root),
    "table_counts": manifest["table_counts"],
    "verified_remote_files": verified_files,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
temporary = receipt.with_suffix(receipt.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(receipt)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

unset HF_TOKEN
echo "HF_PRIVATE_PUBLICATION=PASS"
echo "receipt=$READINESS_HF_PUBLISH_RECEIPT"

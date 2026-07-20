#!/usr/bin/env bash
# Upload this bundle to a HuggingFace Dataset repo.
#
# Prereqs (one-time):
#   pip install -U huggingface_hub
#   hf auth login                   # or: export HF_TOKEN=<your token>
#
# Usage:
#   bash scripts/upload_to_hf.sh <namespace/repo> [--private]
# Example:
#   bash scripts/upload_to_hf.sh ValerianFourel/geodml-papersize --private

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <namespace/repo> [--private]" >&2
  exit 2
fi
REPO="$1"
VISIBILITY="${2:-}"

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"

# 1) Create the repo (no-op if it already exists).
CREATE_FLAGS=(--repo-type dataset)
if [ "$VISIBILITY" = "--private" ]; then
  CREATE_FLAGS+=(--private)
fi
hf repos create "$REPO" "${CREATE_FLAGS[@]}" -y || true

# 2) Upload. The 'hf upload' command auto-uses LFS for large files and
#    auto-chunks multipart. For a 3.5 GB bundle with ~1700 files,
#    upload-large-folder is more robust (parallel, resumable).
hf upload-large-folder "$REPO" . \
  --repo-type dataset \
  --exclude "scripts/__pycache__/*" \
  --exclude "**/.DS_Store"

echo
echo "Done. Dataset is at: https://huggingface.co/datasets/$REPO"

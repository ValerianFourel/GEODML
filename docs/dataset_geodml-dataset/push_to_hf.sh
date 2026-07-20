#!/usr/bin/env bash
# Push the consolidated dataset to HF.
# REQUIRES: real files (not symlinks). Run build with COPY_HTML=1 COPY_DATA=1
# first if anything in the tree is a symlink.
#
# Usage:
#   huggingface-cli login                # paste write-scoped token
#   REPO=ValerianFourel/geodml-papersize-full ./push_to_hf.sh

set -euo pipefail
DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=ValerianFourel/geodml-papersize-full}"

if find "$DATASET_ROOT" -type l -not -path '*/archives/*' | head -1 | grep -q . ; then
  echo "ERROR: tree contains symlinks — HF won't follow them. Rebuild with:"
  echo "  COPY_HTML=1 COPY_DATA=1 FORCE=1 bash <repo>/scripts/build_dataset_mirror.sh"
  exit 2
fi

hf upload-large-folder "$REPO" "$DATASET_ROOT" --repo-type dataset

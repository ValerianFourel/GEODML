#!/usr/bin/env bash
# Unpack every data/runs/*/phase2/html_cache.tar.gz in-place.
# After this runs you'll have data/runs/<run>/phase2/html_cache/*.html
# (~28 GB total across the 8 runs).

set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$BUNDLE_ROOT/data/runs"

for tarball in "$RUNS_DIR"/*/phase2/html_cache.tar.gz; do
  [ -f "$tarball" ] || continue
  run_phase2="$(dirname "$tarball")"
  if [ -d "$run_phase2/html_cache" ]; then
    echo "skip (already unpacked): $run_phase2/html_cache"
    continue
  fi
  echo "unpack: $tarball"
  tar -C "$run_phase2" -xzf "$tarball"
done

echo
echo "Done. Total:"
du -sh "$RUNS_DIR"/*/phase2/html_cache 2>/dev/null || true

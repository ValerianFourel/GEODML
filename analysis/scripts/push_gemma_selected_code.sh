#!/bin/bash
# Run with bash on the development machine; never force-push or add files.
set -euo pipefail
repo="${1:?Usage: bash push_gemma_selected_code.sh /path/to/gemma-selected-cells-release}"
branch=codex/gemma-selected-cells
test "$(git -C "$repo" branch --show-current)" = "$branch"
test -z "$(git -C "$repo" status --porcelain --untracked-files=all)"
pin="$(git -C "$repo" rev-parse HEAD)"
git -C "$repo" push origin "$pin:refs/heads/$branch"
remote_pin="$(git -C "$repo" ls-remote origin "refs/heads/$branch" | awk '{print $1}')"
test "$remote_pin" = "$pin"
printf 'PUBLISHED_PIN=%s\n' "$pin"

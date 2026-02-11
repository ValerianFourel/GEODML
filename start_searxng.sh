#!/bin/bash
# Start SearXNG on port 8888 using Apptainer/Singularity/Docker
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export APPTAINER_CACHEDIR="$SCRIPT_DIR/.cache/apptainer"
export APPTAINER_TMPDIR="$SCRIPT_DIR/.cache/apptainer-tmp"

# Auto-detect container runtime
if command -v apptainer &>/dev/null; then
    CONTAINER_RUNTIME="apptainer"
elif command -v singularity &>/dev/null; then
    CONTAINER_RUNTIME="singularity"
elif command -v docker &>/dev/null; then
    CONTAINER_RUNTIME="docker"
else
    echo "ERROR: No container runtime found (apptainer/singularity/docker)"
    exit 1
fi

echo "Starting SearXNG on http://localhost:8888 ..."
echo "  Runtime: $CONTAINER_RUNTIME"

if [ "$CONTAINER_RUNTIME" = "apptainer" ] || [ "$CONTAINER_RUNTIME" = "singularity" ]; then
    $CONTAINER_RUNTIME exec \
        --pwd /usr/local/searxng \
        --bind "$SCRIPT_DIR/searxng-config:/etc/searxng" \
        --env SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
        "$SCRIPT_DIR/searxng.sif" \
        /usr/local/searxng/entrypoint.sh
elif [ "$CONTAINER_RUNTIME" = "docker" ]; then
    docker run --rm -p 8888:8888 \
        -v "$SCRIPT_DIR/searxng-config:/etc/searxng" \
        -e SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \
        searxng/searxng:latest
fi

#!/bin/bash
# GEODML Experiment Setup — works on Jülich JUWELS (Apptainer) and local machines (Docker/native)
#
# Usage:
#   bash setup.sh
#
# This script:
#   1. Pulls SearXNG container image via Apptainer (or Docker fallback)
#   2. Creates Python venv with experiment dependencies
#   3. Writes SearXNG config
#   4. Creates start/stop helper scripts

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SEARXNG_CONFIG="$SCRIPT_DIR/searxng-config"

echo "=== GEODML Experiment Setup ==="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Redirect caches to project space (avoids HPC home quota issues)
# ---------------------------------------------------------------------------
export PIP_CACHE_DIR="$SCRIPT_DIR/.cache/pip"
export APPTAINER_CACHEDIR="$SCRIPT_DIR/.cache/apptainer"
export APPTAINER_TMPDIR="$SCRIPT_DIR/.cache/apptainer-tmp"
mkdir -p "$PIP_CACHE_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# Clear any broken home pip cache
rm -rf "$HOME/.cache/pip" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 2: Pull SearXNG container
# ---------------------------------------------------------------------------
echo "[1/3] Setting up SearXNG..."

if command -v apptainer &>/dev/null; then
    CONTAINER_RUNTIME="apptainer"
    SIF_PATH="$SCRIPT_DIR/searxng.sif"
    if [ ! -f "$SIF_PATH" ]; then
        echo "  Pulling SearXNG image via Apptainer..."
        apptainer pull "$SIF_PATH" docker://searxng/searxng:latest
    else
        echo "  SearXNG image already exists: $SIF_PATH"
    fi
elif command -v singularity &>/dev/null; then
    CONTAINER_RUNTIME="singularity"
    SIF_PATH="$SCRIPT_DIR/searxng.sif"
    if [ ! -f "$SIF_PATH" ]; then
        echo "  Pulling SearXNG image via Singularity..."
        singularity pull "$SIF_PATH" docker://searxng/searxng:latest
    else
        echo "  SearXNG image already exists: $SIF_PATH"
    fi
elif command -v docker &>/dev/null; then
    CONTAINER_RUNTIME="docker"
    echo "  Using Docker for SearXNG"
    docker pull searxng/searxng:latest 2>/dev/null || true
else
    echo "  WARNING: No container runtime found (apptainer/singularity/docker)"
    echo "  SearXNG will not be available. DuckDuckGo fallback will be used."
    CONTAINER_RUNTIME="none"
fi
echo "  Container runtime: $CONTAINER_RUNTIME"

# ---------------------------------------------------------------------------
# Step 3: Write SearXNG config
# ---------------------------------------------------------------------------
echo "  Writing SearXNG config..."
mkdir -p "$SEARXNG_CONFIG"

cat > "$SEARXNG_CONFIG/settings.yml" << 'EOF'
use_default_settings: true

general:
  instance_name: "geodml-local"
  debug: false

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

server:
  port: 8888
  bind_address: "0.0.0.0"
  secret_key: "geodml-experiment-key-2026"
  limiter: false
  public_instance: false

ui:
  static_use_hash: true

engines:
  - name: google
    disabled: false
  - name: bing
    disabled: false
  - name: duckduckgo
    disabled: false
  - name: brave
    disabled: false
  - name: startpage
    disabled: false
EOF

cat > "$SEARXNG_CONFIG/limiter.toml" << 'EOF'
[botdetection.ip_limit]
link_token = false
EOF

# ---------------------------------------------------------------------------
# Step 4: Python venv
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Setting up Python environment..."

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
fi
source "$SCRIPT_DIR/venv/bin/activate"
pip install --no-cache-dir --upgrade pip setuptools > /dev/null 2>&1
pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"
deactivate
echo "  Python venv ready"

# ---------------------------------------------------------------------------
# Step 5: Create helper scripts
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Creating helper scripts..."

mkdir -p "$DATA_DIR"

# --- start_searxng.sh ---
cat > "$SCRIPT_DIR/start_searxng.sh" << STARTER
#!/bin/bash
# Start SearXNG on port 8888
SCRIPT_DIR="$SCRIPT_DIR"
export APPTAINER_CACHEDIR="$SCRIPT_DIR/.cache/apptainer"
export APPTAINER_TMPDIR="$SCRIPT_DIR/.cache/apptainer-tmp"

CONTAINER_RUNTIME="$CONTAINER_RUNTIME"

echo "Starting SearXNG on http://localhost:8888 ..."

if [ "\$CONTAINER_RUNTIME" = "apptainer" ] || [ "\$CONTAINER_RUNTIME" = "singularity" ]; then
    \$CONTAINER_RUNTIME exec \\
        --bind "\$SCRIPT_DIR/searxng-config:/etc/searxng" \\
        --env SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \\
        "\$SCRIPT_DIR/searxng.sif" \\
        python3 -m searx.webapp
elif [ "\$CONTAINER_RUNTIME" = "docker" ]; then
    docker run --rm -p 8888:8888 \\
        -v "\$SCRIPT_DIR/searxng-config:/etc/searxng" \\
        -e SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml \\
        searxng/searxng:latest
else
    echo "No container runtime. Run setup.sh first."
    exit 1
fi
STARTER
chmod +x "$SCRIPT_DIR/start_searxng.sh"

# --- stop_searxng.sh ---
cat > "$SCRIPT_DIR/stop_searxng.sh" << 'STOPPER'
#!/bin/bash
# Stop SearXNG
pkill -f "searx.webapp" 2>/dev/null && echo "SearXNG stopped" || echo "SearXNG not running"
STOPPER
chmod +x "$SCRIPT_DIR/stop_searxng.sh"

# --- run_experiment.sh ---
cat > "$SCRIPT_DIR/run_experiment.sh" << RUNNER
#!/bin/bash
# Run the full experiment: start SearXNG, run search, summarize
SCRIPT_DIR="$SCRIPT_DIR"
source "\$SCRIPT_DIR/venv/bin/activate"

echo "=== GEODML Experiment ==="
echo ""

# Check if SearXNG is running
if curl -s "http://127.0.0.1:8888/" > /dev/null 2>&1; then
    echo "SearXNG: running on port 8888"
else
    echo "SearXNG: not running. Starting in background..."
    bash "\$SCRIPT_DIR/start_searxng.sh" &
    sleep 8
    if curl -s "http://127.0.0.1:8888/" > /dev/null 2>&1; then
        echo "SearXNG: started"
    else
        echo "SearXNG: failed to start (DuckDuckGo fallback will be used)"
    fi
fi
echo ""

# Run the experiment
python "\$SCRIPT_DIR/run_ai_search.py" --mode standalone

# Summarize
LATEST=\$(ls -t "\$SCRIPT_DIR/results"/ai_search_rankings.json 2>/dev/null | head -1)
if [ -n "\$LATEST" ]; then
    echo ""
    python "\$SCRIPT_DIR/summarize_results.py" "\$LATEST"
fi
RUNNER
chmod +x "$SCRIPT_DIR/run_experiment.sh"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup Complete ==="
echo ""
echo "  Container runtime: $CONTAINER_RUNTIME"
echo "  Python venv:       $SCRIPT_DIR/venv"
echo "  SearXNG config:    $SEARXNG_CONFIG"
echo ""
echo "Quick start:"
echo "  1. bash start_searxng.sh &     # start SearXNG in background"
echo "  2. source venv/bin/activate"
echo "  3. python run_ai_search.py --mode standalone"
echo ""
echo "Or run everything at once:"
echo "  bash run_experiment.sh"
echo ""
echo "After the experiment:"
echo "  python summarize_results.py results/ai_search_rankings.json"

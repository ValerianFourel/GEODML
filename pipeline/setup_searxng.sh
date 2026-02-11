#!/bin/bash
# Setup a local SearXNG instance (no Docker needed).
# SearXNG is a Python app — runs in its own venv.
#
# Usage:
#   bash setup_searxng.sh          # install
#   bash start_searxng.sh          # start (created by this script)
#   bash start_searxng.sh &        # start in background
#
# After starting, set in .env.local:
#   SEARXNG_URL=http://127.0.0.1:8888

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SEARXNG_DIR="$SCRIPT_DIR/searxng-local"

echo "=== Setting up local SearXNG ==="

# Clone
if [ ! -d "$SEARXNG_DIR/.git" ]; then
    echo "Cloning SearXNG..."
    git clone --depth 1 https://github.com/searxng/searxng.git "$SEARXNG_DIR"
else
    echo "SearXNG already cloned at $SEARXNG_DIR"
fi

# Create a separate venv for SearXNG
echo "Creating SearXNG venv..."
python3 -m venv "$SEARXNG_DIR/venv"
source "$SEARXNG_DIR/venv/bin/activate"

# Install deps first, then package
echo "Installing dependencies..."
pip install --upgrade pip setuptools > /dev/null
pip install -r "$SEARXNG_DIR/requirements.txt"
echo "Installing SearXNG..."
pip install --no-build-isolation "$SEARXNG_DIR"

deactivate

# Write config
echo "Writing config..."
mkdir -p "$SEARXNG_DIR/etc/searxng"
cat > "$SEARXNG_DIR/etc/searxng/settings.yml" << 'SETTINGS'
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
  bind_address: "127.0.0.1"
  secret_key: "geodml-experiment-local-key"
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
SETTINGS

# Create start script
cat > "$SCRIPT_DIR/start_searxng.sh" << STARTER
#!/bin/bash
export SEARXNG_SETTINGS_PATH="$SEARXNG_DIR/etc/searxng/settings.yml"
source "$SEARXNG_DIR/venv/bin/activate"
echo "SearXNG starting on http://127.0.0.1:8888 ..."
echo "Test: curl 'http://127.0.0.1:8888/search?q=test&format=json' | python -m json.tool | head"
echo "Press Ctrl+C to stop."
python -m searx.webapp
STARTER
chmod +x "$SCRIPT_DIR/start_searxng.sh"

echo ""
echo "=== Done ==="
echo ""
echo "Step 1: Start SearXNG:"
echo "  bash start_searxng.sh &"
echo ""
echo "Step 2: Make sure .env.local has:"
echo "  SEARXNG_URL=http://127.0.0.1:8888"
echo ""
echo "Step 3: Run the experiment:"
echo "  python run_ai_search.py --mode standalone"

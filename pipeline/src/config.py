import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env.local from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env.local")

HF_TOKEN = os.getenv("HF_TOKEN", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "https://searx.be")

# Fallback SearXNG instances tried in order when the primary returns 403/error
SEARXNG_FALLBACK_URLS = [
    SEARXNG_URL,
    "https://search.bus-hit.me",
    "https://searx.tiekoetter.com",
    "https://search.ononoki.org",
    "https://searx.oxf.fi",
]

# Perplexica instance
PERPLEXICA_URL = os.getenv("PERPLEXICA_URL", "http://localhost:3000")

TOP_N = 10
RESULTS_DIR = PROJECT_ROOT / "results"
KEYWORDS_FILE = PROJECT_ROOT / "keywords.txt"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

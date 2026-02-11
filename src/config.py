import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env.local from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env.local")

HF_TOKEN = os.getenv("HF_TOKEN", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")

TOP_N = 10
RESULTS_DIR = PROJECT_ROOT / "results"
KEYWORDS_FILE = PROJECT_ROOT / "keywords.txt"

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

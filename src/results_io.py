import csv
import json
from pathlib import Path
from src.config import RESULTS_DIR


def save_results(data: dict, filename: str) -> Path:
    """Save results dict to a JSON file in the results directory."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON: {filepath}")
    return filepath


def load_results(filename: str) -> dict:
    """Load results from a JSON file in the results directory."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "r") as f:
        return json.load(f)


def results_to_csv(data: dict, filename: str) -> Path:
    """Flatten results to CSV with columns: keyword, rank, domain, source.

    Expects data format:
    {
        "source": "ai_search" | "duckduckgo" | "google",
        "rankings": {
            "keyword1": ["domain1", "domain2", ...],
            "keyword2": [...],
        }
    }
    """
    filepath = RESULTS_DIR / filename
    source = data.get("source", "unknown")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "rank", "domain", "source"])
        for keyword, domains in data.get("rankings", {}).items():
            for rank, domain in enumerate(domains, 1):
                writer.writerow([keyword, rank, domain, source])

    print(f"  Saved CSV: {filepath}")
    return filepath

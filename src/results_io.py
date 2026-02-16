import csv
import json
from pathlib import Path
from src.config import RESULTS_DIR


def save_results(data: dict, filename: str) -> Path:
    """Save results dict to a JSON file in the results directory."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {filepath}")
    return filepath


def load_results(filename: str) -> dict:
    """Load results from a JSON file in the results directory."""
    filepath = RESULTS_DIR / filename
    with open(filepath, "r") as f:
        return json.load(f)


def results_to_csv(query_results: list[dict], source: str, filename: str) -> Path:
    """Flatten per-query results to CSV with full provenance.

    Columns: keyword, rank, domain, source, query_timestamp_utc,
             first_serp_position, total_raw_results

    query_results: list of per-query result dicts (from search functions).
    source: label like "ai_search", "duckduckgo", "google".
    """
    filepath = RESULTS_DIR / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "keyword", "rank", "domain", "url", "source",
            "query_timestamp_utc", "first_serp_position", "total_raw_results",
        ])

        for qr in query_results:
            keyword = qr.get("query") or qr.get("keyword", "")
            timestamp = qr.get("query_timestamp_utc", "")
            raw_results = qr.get("raw_results", [])
            total_raw = len(raw_results)

            # Build domain → first SERP position and domain → URL lookups
            import tldextract
            domain_first_pos = {}
            domain_url = {}
            for rr in raw_results:
                url = rr.get("url", "")
                d = rr.get("domain") or ""
                if not d:
                    ext = tldextract.extract(url)
                    d = f"{ext.domain}.{ext.suffix}" if ext.domain and ext.suffix else ""
                if d and d not in domain_first_pos:
                    domain_first_pos[d] = rr.get("position", "")
                    domain_url[d] = url

            # Use ranked_results if available (has URLs), fall back to ranked_domains
            ranked_results = qr.get("ranked_results", [])
            if ranked_results:
                for rank, entry in enumerate(ranked_results, 1):
                    domain = entry["domain"]
                    url = entry.get("url") or domain_url.get(domain, "")
                    first_pos = domain_first_pos.get(domain, "")
                    writer.writerow([
                        keyword, rank, domain, url, source,
                        timestamp, first_pos, total_raw,
                    ])
            else:
                domains = qr.get("ranked_domains", [])
                for rank, domain in enumerate(domains, 1):
                    url = domain_url.get(domain, "")
                    first_pos = domain_first_pos.get(domain, "")
                    writer.writerow([
                        keyword, rank, domain, url, source,
                        timestamp, first_pos, total_raw,
                    ])

    print(f"  Saved CSV: {filepath}")
    return filepath

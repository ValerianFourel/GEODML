#!/usr/bin/env python3
"""Extract unique (domain, url) pairs from all_results.csv."""

import csv
from pathlib import Path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract unique domains from all_results CSV")
    parser.add_argument("--filter", type=str, default=None,
                        help="Use the filtered CSV (e.g. 'searxng' reads all_results_searxng.csv)")
    args = parser.parse_args()

    results_dir = Path("results")
    suffix = f"_{args.filter}" if args.filter else ""
    in_path = results_dir / f"all_results{suffix}.csv"

    if not in_path.exists():
        print(f"Run extract_all_results.py --filter {args.filter or ''} first to generate {in_path.name}")
        return

    # domain -> best URL (first non-empty one encountered)
    domain_url = {}
    with open(in_path) as f:
        for row in csv.DictReader(f):
            domain = row["domain"]
            url = row["url"]
            if domain and domain not in domain_url:
                domain_url[domain] = url
            elif domain and not domain_url[domain] and url:
                domain_url[domain] = url

    out_path = results_dir / f"unique_domains{suffix}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "url"])
        writer.writeheader()
        for domain in sorted(domain_url):
            writer.writerow({"domain": domain, "url": domain_url[domain]})

    print(f"Unique domains: {len(domain_url)}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

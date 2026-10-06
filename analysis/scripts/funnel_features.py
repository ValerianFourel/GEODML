#!/usr/bin/env python3
"""Feature table of the funnel study (``analysis/docs/funnel_study.md``), built on the Mac.

  probe   print inputs, hashes and join coverage (writes nothing)
  build   write <output>/{rows,docs,urls,domains,keywords,google}.parquet, coverage.json, manifest.json
  verify  recheck an existing feature folder against the snapshots (row digest, counts)

Outcome-free: nothing here reads a generator answer. Page-body features are re-extracted from raw
HTML for every V2 URL; experiment-1 values are used only to validate the extraction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from analysis.interpretability.pipeline import funnel_features as ff  # noqa: E402
from analysis.interpretability.pipeline import funnel_rows as fr  # noqa: E402
from analysis.scripts import page_readiness_ordering as readiness  # noqa: E402

FORMAT_VERSION = "funnel-features-v1"
CELLS = ("duckduckgo_Llama-3.3-70B-Instruct_serp20_top10", "duckduckgo_Llama-3.3-70B-Instruct_serp50_top10",
         "duckduckgo_Qwen2.5-72B-Instruct_serp20_top10", "duckduckgo_Qwen2.5-72B-Instruct_serp50_top10",
         "searxng_Llama-3.3-70B-Instruct_serp20_top10", "searxng_Llama-3.3-70B-Instruct_serp50_top10",
         "searxng_Qwen2.5-72B-Instruct_serp20_top10", "searxng_Qwen2.5-72B-Instruct_serp50_top10")
LINK_FEATURES = ("body_ext_citations", "body_auth_citations", "body_internal_links", "body_outbound_links")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def html_name(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- inputs

def load_inputs(args) -> dict:
    import pandas as pd
    paths = {"duckduckgo": args.snapshot_ddg, "searxng": args.snapshot_searxng}
    rows = fr.snapshot_rows(paths)
    raw = []
    for engine, path in paths.items():
        frame = pd.read_parquet(path)
        frame["engine"] = engine
        raw.append(frame)
    raw = pd.concat(raw, ignore_index=True)
    corpus = pd.read_parquet(args.corpus)
    exp1 = pd.read_parquet(args.experiment1).drop_duplicates()
    exp1 = exp1[exp1["url"].notna() & (exp1["url"] != "")]
    return {"rows": rows, "raw": raw, "corpus": corpus, "exp1": exp1, "paths": paths}


def row_table(rows: fr.SnapshotRows, raw, corpus) -> "pd.DataFrame":
    import pandas as pd
    table = pd.DataFrame({"row_id": np.arange(len(rows)), "engine": rows.engine, "keyword": rows.keyword,
                          "position": rows.position, "url": rows.url, "title": rows.title, "snippet": rows.snippet,
                          "doc_id": rows.doc_id})
    extra = raw[["engine", "keyword", "position", "url", "score", "engines"]].copy()
    extra["position"] = pd.to_numeric(extra["position"], errors="coerce")
    extra = extra.dropna(subset=["position"]).drop_duplicates(["engine", "keyword", "position", "url"])
    extra["position"] = extra["position"].astype(np.int64)
    # SearXNG stores the upstream engines as "startpage|duckduckgo"
    extra["searxng_engine_count"] = extra["engines"].apply(
        lambda e: len([x for x in str(e).split("|") if x]) if isinstance(e, str) and e else np.nan)
    table = table.merge(extra[["engine", "keyword", "position", "url", "score", "searxng_engine_count"]].rename(
        columns={"score": "searxng_score"}), on=["engine", "keyword", "position", "url"], how="left", validate="one_to_one")
    table.loc[table["engine"] != "searxng", ["searxng_score", "searxng_engine_count"]] = np.nan
    corpus_row = {s: i for i, s in enumerate(corpus["snippet_id"])}
    table["corpus_row"] = table["doc_id"].map(corpus_row)
    if table["corpus_row"].isna().any():
        raise ValueError(f"{int(table['corpus_row'].isna().sum())} snapshot rows are missing from the corpus")
    table["corpus_row"] = table["corpus_row"].astype(np.int64)
    return table


# ---------------------------------------------------------------- HTML copies

_PARSE: dict = {}


def _parse_copy(job):
    url, source, fetched, size, payload = job
    meta = _PARSE["meta"][url]
    html = payload.decode("utf-8", errors="replace")
    ok, reason = ff.usable_html(html, size)
    out = {"url": url, "html_source": source, "html_fetched": fetched, "html_bytes": size, "html_usable": int(ok),
           "html_reason": reason}
    if not ok:
        return out
    try:
        features = ff.body_features(html, meta["domain"], meta["snippet"])
        if not features["body_words_ok"]:
            return {**out, "html_usable": 0, "html_reason": "too_few_words"}
        out.update(features)
        from analysis.interpretability.pipeline import features as page
        soup = page._get_soup(html)
        host = meta["host"]
        out.update({"body_ext_citations_host": page.extract_t4a_ext_citations_any(soup, host),
                    "body_auth_citations_host": page.extract_t4b_auth_citations(soup, host),
                    "body_internal_links_host": page.extract_internal_links(soup, host),
                    "body_outbound_links_host": page.extract_outbound_links(soup, host)})
    except Exception as error:  # an unparsable page is a missing copy, recorded with its reason
        return {**out, "html_usable": 0, "html_reason": f"parse_error:{type(error).__name__}"}
    return out


def html_copies(args, needed: dict, workers: int, limit=None):
    """Parse every usable copy of every needed URL: the 8 experiment-1 archives and the gap-fill pages."""
    import multiprocessing
    jobs_done, results = 0, []

    def flush(batch):
        if not batch:
            return
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                results.extend(pool.map(_parse_copy, batch, chunksize=8))
        else:
            results.extend(map(_parse_copy, batch))

    for cell in CELLS:
        archive = Path(args.runs) / cell / "phase2" / "html_cache.tar.gz"
        if not archive.exists():
            continue
        batch = []
        with tarfile.open(archive) as tar:
            for member in tar:
                name = os.path.basename(member.name)
                if not member.isfile() or name.startswith("._") or not name.endswith(".html"):
                    continue
                for url in needed.get(name[:-5], ()):
                    payload = tar.extractfile(member).read()
                    fetched = datetime.fromtimestamp(member.mtime, timezone.utc).date().isoformat()
                    batch.append((url, f"exp1:{cell}", fetched, member.size, payload))
                if len(batch) >= 400:
                    flush(batch)
                    jobs_done += len(batch)
                    batch = []
                    print(json.dumps({"parsed": jobs_done, "archive": cell}), flush=True)
                if limit and jobs_done >= limit:
                    break
        flush(batch)
        jobs_done += len(batch)
    if args.gapfill and Path(args.gapfill).exists():
        batch = []
        for path in sorted(Path(args.gapfill).rglob("*.html")):
            if path.name.startswith("._"):
                continue
            for url in needed.get(path.stem, ()):
                batch.append((url, f"gapfill:{path.parent.parent.parent.name}", args.gapfill_date, path.stat().st_size, path.read_bytes()))
            if len(batch) >= 400:
                flush(batch)
                jobs_done += len(batch)
                batch = []
                print(json.dumps({"parsed": jobs_done, "archive": "gapfill"}), flush=True)
        flush(batch)
    return results


def choose_copies(copies, link_convention: str) -> dict:
    """One copy per URL: usable copies only, fetch date closest to the reference date, then source label."""
    reference = ff.REFERENCE_DATE.date()
    by_url = defaultdict(list)
    for c in copies:
        by_url[c["url"]].append(c)
    chosen = {}
    for url, items in by_url.items():
        usable = [c for c in items if c["html_usable"]]
        if not usable:
            reasons = sorted({c["html_reason"] for c in items})
            chosen[url] = {"url": url, "html_usable": 0, "html_reason": ";".join(reasons), "html_copies": len(items)}
            continue
        best = min(usable, key=lambda c: (abs((datetime.fromisoformat(c["html_fetched"]).date() - reference).days), c["html_source"]))
        row = dict(best)
        if link_convention == "host":
            for f in LINK_FEATURES:
                row[f] = row.get(f"{f}_host")
        for f in LINK_FEATURES:
            row.pop(f"{f}_host", None)
        row.update({"html_copies": len(items), "html_usable_copies": len(usable)})
        chosen[url] = row
    return chosen


def retest_agreement(copies) -> dict:
    """Agreement between usable copies of the same URL (first two copies by source label)."""
    by_url = defaultdict(list)
    for c in copies:
        if c["html_usable"]:
            by_url[c["url"]].append(c)
    pairs = [sorted(v, key=lambda c: c["html_source"])[:2] for v in by_url.values() if len(v) >= 2]
    out = {"urls_with_two_copies": len(pairs)}
    for f in ("body_word_count", "body_stats_density", "body_question_headings", "body_structured_data", "body_freshness",
              "body_internal_links", "body_outbound_links"):
        a = np.asarray([p[0].get(f) for p in pairs], float)
        b = np.asarray([p[1].get(f) for p in pairs], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 2:
            out[f] = {"n": int(ok.sum()), "exact_agreement": float(np.mean(a[ok] == b[ok])),
                      "spearman": float(_spearman(a[ok], b[ok]))}
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic if len(a) > 2 else float("nan")


# ---------------------------------------------------------------- tables

def build_tables(args, workers: int, html_limit=None) -> dict:
    import pandas as pd
    from analysis.interpretability.pipeline import features as page

    data = load_inputs(args)
    rows, exp1, corpus = data["rows"], data["exp1"], data["corpus"]
    table = row_table(rows, data["raw"], corpus)
    known = dict(exp1.groupby("url")["domain"].agg(lambda d: d.mode().iat[0] if len(d.mode()) else None))

    urls = pd.DataFrame({"url": sorted(set(table["url"]))})
    resolved = [ff.registrable_domain(u, known) for u in urls["url"]]
    urls["domain"] = [d for d, _ in resolved]
    urls["domain_source"] = [s for _, s in resolved]
    urls["url_normalized"] = urls["url"].map(ff.normalize_url)
    urls["ad_target"] = urls["url"].map(ff.decode_bing_ad)
    urls = pd.concat([urls, pd.DataFrame([ff.url_features(u, d) for u, d in zip(urls["url"], urls["domain"])])], axis=1)

    first_snippet = table.drop_duplicates("url").set_index("url")["snippet"]
    meta = {u: {"domain": d, "host": (urlsplit(u).hostname or "").lower().removeprefix("www."), "snippet": first_snippet[u]}
            for u, d in zip(urls["url"], urls["domain"])}
    needed = defaultdict(list)
    for u in urls["url"]:
        needed[html_name(u)].append(u)
    _PARSE["meta"] = meta
    copies = html_copies(args, needed, workers, limit=html_limit)

    # link-domain convention: the one that agrees best with experiment 1 on shared URLs (outcome-free)
    exp1_links = exp1.groupby("url")["conf_internal_links"].median()
    agreement = {}
    for convention, column in (("registrable", "body_internal_links"), ("host", "body_internal_links_host")):
        pairs = [(c[column], exp1_links[c["url"]]) for c in copies if c["html_usable"] and c["url"] in exp1_links.index
                 and c.get(column) is not None and exp1_links[c["url"]] == exp1_links[c["url"]]]
        agreement[convention] = float(_spearman(*map(np.asarray, zip(*pairs)))) if len(pairs) > 2 else float("nan")
    convention = "host" if agreement.get("host", -1) > agreement.get("registrable", -1) else "registrable"
    chosen = choose_copies(copies, convention)
    body = pd.DataFrame([chosen.get(u, {"url": u, "html_usable": 0, "html_reason": "no_copy", "html_copies": 0}) for u in urls["url"]])
    urls = urls.merge(body, on="url", how="left", validate="one_to_one")

    # experiment-1 values on shared URLs: validation only
    validation_columns = {"conf_word_count": "body_word_count", "treat_stats_density": "body_stats_density",
                          "treat_question_headings": "body_question_headings", "treat_structured_data": "body_structured_data",
                          "treat_freshness": "body_freshness", "conf_internal_links": "body_internal_links"}
    medians = exp1.groupby("url")[list(validation_columns)].median()
    urls = urls.merge(medians.add_prefix("exp1_"), left_on="url", right_index=True, how="left")
    validation = {}
    for source, mine in validation_columns.items():
        a, b = urls[mine].astype(float), urls[f"exp1_{source}"].astype(float)
        ok = a.notna() & b.notna()
        if ok.sum() > 2:
            validation[mine] = {"n": int(ok.sum()), "spearman": float(_spearman(a[ok].values, b[ok].values)),
                                "exact_agreement": float(np.mean(a[ok].values == b[ok].values))}

    # domains (block C1)
    domains = pd.DataFrame({"domain": sorted(set(urls["domain"]))})
    dfs = pd.read_parquet(args.domain_table)
    dfs["domain"] = dfs["domain"].astype(str).str.lower().str.strip()
    dfs = dfs.drop_duplicates("domain")
    domains = domains.merge(dfs, on="domain", how="left")
    opr = exp1.groupby("domain").agg(opr=("X1_domain_authority", lambda v: ff.constant_value(list(v))[0]),
                                     opr_values=("X1_domain_authority", lambda v: ff.constant_value(list(v))[1]),
                                     opr_global_rank=("X1_global_rank", lambda v: ff.constant_value(list(v))[0]),
                                     moz_da=("conf_domain_authority", "median"))
    domains = domains.merge(opr, left_on="domain", right_index=True, how="left")
    llms = pd.read_parquet(args.llms)
    llms["domain"] = llms["domain"].astype(str).str.lower()
    domains = domains.merge(llms.drop_duplicates("domain")[["domain", "has_llms_txt"]], on="domain", how="left")
    domains["brand_list"] = domains["domain"].isin(page.BRAND_DOMAINS).astype(int)
    domains["earned_list"] = domains["domain"].isin(page.EARNED_DOMAINS).astype(int)
    domains["dfs_age_years_at_reference"] = [
        (ff.REFERENCE_DATE - datetime.fromisoformat(str(t).replace(" +00:00", "+00:00").replace("Z", "+00:00"))).days / 365.25
        if isinstance(t, str) and t[:4].isdigit() else np.nan for t in domains["dfs_created_datetime"]]

    # keywords
    dfs_dir = Path(args.dataforseo)
    keywords = pd.DataFrame({"keyword": sorted(set(table["keyword"]))})
    kd = pd.read_parquet(dfs_dir / "bulk_keyword_difficulty.parquet")[["keyword", "keyword_difficulty"]]
    ov = pd.read_parquet(dfs_dir / "keyword_overview.parquet")
    ov = ov.rename(columns={"keyword_info.search_volume": "kw_search_volume", "keyword_info.cpc": "kw_cpc",
                            "keyword_info.competition": "kw_competition", "search_intent_info.main_intent": "kw_main_intent",
                            "keyword_properties.keyword_difficulty": "kw_difficulty_overview"})
    ga = pd.read_parquet(dfs_dir / "google_ads_search_volume.parquet")
    si = pd.read_parquet(dfs_dir / "search_intent.parquet")
    keywords = (keywords.merge(kd, on="keyword", how="left")
                .merge(ov[["keyword", "kw_search_volume", "kw_cpc", "kw_competition", "kw_main_intent", "kw_difficulty_overview"]],
                       on="keyword", how="left")
                .merge(ga[["keyword", "ga_search_volume", "ga_cpc", "ga_competition"]], on="keyword", how="left")
                .merge(si[["keyword", "si_main_intent"]], on="keyword", how="left"))
    keywords["kw_difficulty"] = keywords["keyword_difficulty"].fillna(keywords["kw_difficulty_overview"])
    keywords["kw_search_volume"] = keywords["kw_search_volume"].fillna(keywords["ga_search_volume"])
    keywords["kw_cpc"] = keywords["kw_cpc"].fillna(keywords["ga_cpc"])
    keywords["kw_main_intent"] = keywords["kw_main_intent"].fillna(keywords["si_main_intent"])
    terciles = tuple(np.nanquantile(keywords["kw_difficulty"].astype(float), [1 / 3, 2 / 3]))
    mods = [ff.keyword_moderators(i if isinstance(i, str) else None, None if d != d else float(d), terciles)
            for i, d in zip(keywords["kw_main_intent"], keywords["kw_difficulty"])]
    keywords = pd.concat([keywords, pd.DataFrame(mods)], axis=1)

    # Google organic results for the same keywords (block C1), by URL and by registrable domain
    google = pd.read_parquet(dfs_dir / "serp_google_organic.parquet")
    google = google[google["url"].notna()].copy()
    google["url_normalized"] = google["url"].map(ff.normalize_url)
    google["domain"] = [ff.registrable_domain(u)[0] for u in google["url"]]
    google = google[["keyword", "url_normalized", "domain", "rank_group"]]
    by_url = google.groupby(["keyword", "url_normalized"])["rank_group"].min().rename("google_rank_url")
    by_domain = google.groupby(["keyword", "domain"])["rank_group"].min().rename("google_rank_domain")
    table = table.merge(urls[["url", "url_normalized", "domain"]], on="url", how="left")
    table = table.merge(by_url, left_on=["keyword", "url_normalized"], right_index=True, how="left")
    table = table.merge(by_domain, left_on=["keyword", "domain"], right_index=True, how="left")
    table["google_top20_url"] = table["google_rank_url"].notna().astype(int)
    table["google_top20_domain"] = table["google_rank_domain"].notna().astype(int)

    docs = corpus.drop(columns=[c for c in ("urls", "engines", "models", "keywords", "title", "snippet", "text") if c in corpus.columns]).copy()
    docs["corpus_row"] = np.arange(len(docs))
    first_title = table.drop_duplicates("doc_id").set_index("doc_id")
    surface = [ff.snippet_features(first_title.at[d, "title"], first_title.at[d, "snippet"], first_title.at[d, "domain"], int(g))
               if d in first_title.index else {} for d, g in zip(corpus["snippet_id"], corpus["glued_titles"])]
    docs = pd.concat([docs.reset_index(drop=True), pd.DataFrame(surface)], axis=1)

    return {"rows": table, "docs": docs, "urls": urls, "domains": domains, "keywords": keywords, "google": google,
            "copies": copies, "retest": retest_agreement(copies), "validation": validation, "link_agreement": agreement,
            "link_convention": convention, "snapshot_rows": rows, "paths": data["paths"]}


def coverage(tables) -> dict:
    rows, urls, domains = tables["rows"], tables["urls"], tables["domains"]
    joined = rows.merge(urls[["url", "html_usable"]], on="url", how="left").merge(
        domains[["domain", "dfs_organic_count", "opr", "has_llms_txt"]], on="domain", how="left")
    out = {}
    for engine, part in joined.groupby("engine"):
        out[engine] = {"rows": int(len(part)), "html_usable": float(part["html_usable"].fillna(0).mean()),
                       "dfs_domain": float(part["dfs_organic_count"].notna().mean()), "open_pagerank": float(part["opr"].notna().mean()),
                       "llms_txt": float(part["has_llms_txt"].notna().mean()), "google_top20_url": float(part["google_top20_url"].mean()),
                       "google_top20_domain": float(part["google_top20_domain"].mean())}
    return out


def build(args) -> int:
    output = Path(args.output).resolve()
    for candidate in (output, output.with_name(output.name + ".partial")):
        if candidate.exists():
            raise ValueError(f"refusing to overwrite {candidate}")
    workers = max(1, args.workers or 1)
    tables = build_tables(args, workers, html_limit=args.html_limit)
    partial = readiness.new_directory(output)
    for name in ("rows", "docs", "urls", "domains", "keywords", "google"):
        tables[name].to_parquet(partial / f"{name}.parquet", index=False)
    readiness.write_json(partial / "coverage.json", {"by_engine": coverage(tables), "retest": tables["retest"],
                                                     "validation_against_experiment1": tables["validation"],
                                                     "link_domain_agreement": tables["link_agreement"],
                                                     "link_domain_convention": tables["link_convention"],
                                                     "html_copies_parsed": len(tables["copies"])})
    inputs = {name: {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))} for name, p in (
        ("snapshot_duckduckgo", args.snapshot_ddg), ("snapshot_searxng", args.snapshot_searxng), ("corpus", args.corpus),
        ("experiment1", args.experiment1), ("domain_table", args.domain_table), ("llms_txt", args.llms))}
    readiness.write_json(partial / "manifest.json", {
        "format_version": FORMAT_VERSION, "created_at": readiness.now(), "git_commit": readiness.git_commit(),
        "inputs": inputs, "snapshot_sha256": tables["snapshot_rows"].snapshot_sha256,
        "row_table_digest": fr.row_table_digest(tables["snapshot_rows"]),
        "reference_date": ff.REFERENCE_DATE.date().isoformat(), "gapfill_declared_fetch_date": args.gapfill_date,
        "counts": {name: int(len(tables[name])) for name in ("rows", "docs", "urls", "domains", "keywords", "google")},
        "rules": {"html_copy": "usable (>=2KB, >=50 words, no bot challenge); fetch date closest to the reference date; then source label",
                  "domain": "experiment-1 domain for known URLs, else offline public-suffix list; ad redirects resolve to the advertiser",
                  "google": "min rank_group by (keyword, normalized URL) and by (keyword, registrable domain)"},
        "versions": {"tldextract": __import__("tldextract").__version__, "textstat": __import__("textstat").__version__},
        "html_limit": args.html_limit})
    partial.rename(output)
    print(json.dumps({"output": str(output), "coverage": coverage(tables)}, indent=1), flush=True)
    return 0


def probe(args) -> int:
    data = load_inputs(args)
    rows = data["rows"]
    print(json.dumps({"rows": len(rows), "snapshot_sha256": rows.snapshot_sha256, "exclusions": rows.exclusions,
                      "row_table_digest": fr.row_table_digest(rows), "corpus_pages": int(len(data["corpus"])),
                      "experiment1_rows": int(len(data["exp1"]))}, indent=1))
    return 0


def verify(args) -> int:
    import pandas as pd
    folder = Path(args.features)
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = fr.snapshot_rows({"duckduckgo": args.snapshot_ddg, "searxng": args.snapshot_searxng})
    digest = fr.row_table_digest(rows)
    table = pd.read_parquet(folder / "rows.parquet", columns=["row_id"])
    ok = digest == manifest["row_table_digest"] and len(table) == len(rows) and rows.snapshot_sha256 == manifest["snapshot_sha256"]
    print(json.dumps({"row_table_digest_matches": digest == manifest["row_table_digest"], "rows": len(table),
                      "snapshots_match": rows.snapshot_sha256 == manifest["snapshot_sha256"], "verified": ok}))
    return 0 if ok else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    home = Path.home() / "Hamburg"
    data = home / "GEODML_Analysis/geodml_data/data"
    for name in ("probe", "build", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot-ddg", type=Path, default=data / "serp/phase0_top20_ddg.parquet")
        p.add_argument("--snapshot-searxng", type=Path, default=data / "serp/phase0_top20_searxng.parquet")
        if name == "verify":
            p.add_argument("--features", type=Path, required=True)
            continue
        p.add_argument("--corpus", type=Path, required=True, help="snippet-embeddings-corpus-v1/snippets.parquet")
        p.add_argument("--experiment1", type=Path, default=data / "main/full_experiment_data.parquet")
        p.add_argument("--runs", type=Path, default=data / "runs")
        p.add_argument("--gapfill", type=Path, default=home / "geodml-inputs/papersize-html")
        p.add_argument("--gapfill-date", default="2026-05-07")
        p.add_argument("--llms", type=Path, default=data / "domains_llms_txt.parquet")
        p.add_argument("--domain-table", type=Path, default=home / "geodml-inputs/emnlp-2026/data/dataforseo/domain_authority_dfs.parquet")
        p.add_argument("--dataforseo", type=Path, default=data / "dataforseo")
        p.add_argument("--workers", type=int, default=os.cpu_count())
        p.add_argument("--html-limit", type=int, help="development only: stop after this many HTML copies")
        p.add_argument("--output", type=Path, required=(name == "build"))
    args = parser.parse_args(argv)
    return {"probe": probe, "build": build, "verify": verify}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone LLM re-ranker using HF Inference API.

Used when running without Perplexica (standalone SearXNG + LLM pipeline).
The LLM re-ranks SearXNG results — this re-ranking is the experimental variable.
The keyword is passed as-is (bare search term, no sentence wrapping).
"""

import tldextract
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, TOP_N
from src.experiment_context import utcnow_iso

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _build_prompt(keyword: str, search_results: list[dict], top_n: int) -> str:
    """Build a minimal re-ranking prompt.

    The keyword is presented bare (exactly as typed into a search engine).
    The LLM sees the raw SERP and re-ranks by its own judgement — this
    re-ranking is what the experiment measures.
    """
    results_text = ""
    for r in search_results:
        domain = _extract_domain(r["url"])
        results_text += (
            f"{r['position']}. [{domain}] {r['title']} — {r['snippet'][:150]}\n"
        )

    return f"""Search keyword: {keyword}

Below are search engine results for the above keyword. Re-rank the results and return the top {top_n} software product domains, ordered by relevance to the keyword.

Exclude non-product sites: review aggregators, directories, Wikipedia, news, blogs, forums, YouTube.

Return only root domains, one per line. No explanations.

Search results:
{results_text}

Re-ranked product domains:"""


def _parse_domains(llm_output: str) -> list[str]:
    """Parse LLM output into a list of clean domain strings."""
    domains = []
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1." or "1)" or "- "
        for prefix in ["- ", "* "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
            if len(parts) > 1:
                line = parts[1].strip()

        ext = tldextract.extract(line)
        if ext.domain and ext.suffix:
            domain = f"{ext.domain}.{ext.suffix}"
            if domain not in domains:
                domains.append(domain)
    return domains


def _build_domain_url_map(search_results: list[dict]) -> dict:
    """Build domain → best URL mapping from raw search results.

    For each domain, keeps the highest-ranked (lowest position) URL.
    """
    domain_url = {}
    for r in search_results:
        url = r.get("url", "")
        domain = _extract_domain(url)
        if domain and domain not in domain_url:
            domain_url[domain] = url
    return domain_url


def _attach_urls(domains: list[str], domain_url_map: dict) -> list[dict]:
    """Attach the original SERP URL to each ranked domain."""
    return [
        {"domain": d, "url": domain_url_map.get(d, "")}
        for d in domains
    ]


def rank_domains_with_llm(
    keyword: str, search_results: list[dict], top_n: int = TOP_N
) -> dict:
    """Use an LLM to re-rank search results by relevance to the bare keyword.

    The LLM re-ranking is the experimental variable. The keyword is passed
    as a bare search term, not wrapped in a sentence.

    Returns dict with full provenance:
        keyword, llm_role, llm_model, prompt, raw_llm_response,
        llm_query_timestamp_utc, llm_response_timestamp_utc,
        ranked_domains, ranked_results, used_fallback, error
    """
    result = {
        "keyword": keyword,
        "llm_role": "re-ranker (LLM re-orders results by relevance)",
        "llm_model": MODEL_ID,
        "llm_parameters": {"max_tokens": 500, "temperature": 0.1},
        "prompt": None,
        "raw_llm_response": None,
        "llm_query_timestamp_utc": None,
        "llm_response_timestamp_utc": None,
        "ranked_domains": [],
        "ranked_results": [],
        "used_fallback": False,
        "error": None,
    }

    if not search_results:
        result["error"] = "no search results provided"
        return result

    domain_url_map = _build_domain_url_map(search_results)

    client = InferenceClient(token=HF_TOKEN)
    prompt = _build_prompt(keyword, search_results, top_n)
    result["prompt"] = prompt

    result["llm_query_timestamp_utc"] = utcnow_iso()

    try:
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            model=MODEL_ID,
            max_tokens=500,
            temperature=0.1,
        )
        llm_output = response.choices[0].message.content
    except Exception as e:
        result["llm_response_timestamp_utc"] = utcnow_iso()
        result["error"] = str(e)
        result["used_fallback"] = True
        domains = _fallback_extract(search_results, top_n)
        result["ranked_domains"] = domains
        result["ranked_results"] = _attach_urls(domains, domain_url_map)
        print(f"  [LLM] Error ranking for '{keyword}': {e}")
        return result

    result["llm_response_timestamp_utc"] = utcnow_iso()
    result["raw_llm_response"] = llm_output
    domains = _parse_domains(llm_output)[:top_n]
    result["ranked_domains"] = domains
    result["ranked_results"] = _attach_urls(domains, domain_url_map)

    return result


def _fallback_extract(search_results: list[dict], top_n: int) -> list[str]:
    """Fallback: extract unique domains from search results without LLM."""
    skip_domains = {
        "g2.com", "capterra.com", "wikipedia.org", "youtube.com",
        "reddit.com", "quora.com", "forbes.com", "techcrunch.com",
        "gartner.com", "trustradius.com", "softwareadvice.com",
        "getapp.com", "pcmag.com", "techradar.com", "cnet.com",
    }
    domains = []
    for r in search_results:
        d = _extract_domain(r["url"])
        if d and d not in domains and d not in skip_domains:
            domains.append(d)
        if len(domains) >= top_n:
            break
    return domains

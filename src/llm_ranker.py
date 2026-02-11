import tldextract
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, TOP_N


def _extract_domain(url: str) -> str:
    """Extract the root domain from a URL using tldextract."""
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def _build_prompt(keyword: str, search_results: list[dict], top_n: int) -> str:
    """Build the LLM prompt for domain ranking."""
    results_text = ""
    for i, r in enumerate(search_results, 1):
        domain = _extract_domain(r["url"])
        results_text += f"{i}. [{domain}] {r['title']} — {r['snippet'][:150]}\n"

    return f"""You are an expert at identifying B2B SaaS products. Given these search results for the query "{keyword}", extract and rank the top {top_n} most relevant B2B SaaS product domains.

Rules:
- Only include actual B2B SaaS product websites (not review sites like G2, Capterra, not directories, not Wikipedia, not news sites).
- Return only root domains (e.g., salesforce.com), one per line.
- Rank by relevance to the query "{keyword}".
- Return EXACTLY {top_n} domains or fewer if not enough qualify.

Search results:
{results_text}

Top {top_n} B2B SaaS domains (one per line, most relevant first):"""


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
            # Strip "1. " or "1) " patterns
            parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
            if len(parts) > 1:
                line = parts[1].strip()

        # Normalize with tldextract
        ext = tldextract.extract(line)
        if ext.domain and ext.suffix:
            domain = f"{ext.domain}.{ext.suffix}"
            if domain not in domains:
                domains.append(domain)
    return domains


def rank_domains_with_llm(
    keyword: str, search_results: list[dict], top_n: int = TOP_N
) -> list[str]:
    """Use an LLM via HF Inference API to rank domains from search results.

    Returns an ordered list of root domain strings.
    """
    if not search_results:
        return []

    client = InferenceClient(token=HF_TOKEN)
    prompt = _build_prompt(keyword, search_results, top_n)

    try:
        response = client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.3",
            max_new_tokens=300,
            temperature=0.1,
        )
    except Exception as e:
        print(f"  [LLM] Error ranking for '{keyword}': {e}")
        # Fallback: extract domains directly from search results
        return _fallback_extract(search_results, top_n)

    domains = _parse_domains(response)
    return domains[:top_n]


def _fallback_extract(search_results: list[dict], top_n: int) -> list[str]:
    """Fallback: extract unique domains from search results without LLM."""
    skip_domains = {
        "g2.com", "capterra.com", "wikipedia.org", "youtube.com",
        "reddit.com", "quora.com", "forbes.com", "techcrunch.com",
        "gartner.com", "trustradius.com", "softwareadvice.com",
    }
    domains = []
    for r in search_results:
        d = _extract_domain(r["url"])
        if d and d not in domains and d not in skip_domains:
            domains.append(d)
        if len(domains) >= top_n:
            break
    return domains

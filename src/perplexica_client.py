"""Perplexica API client for AI-powered search with LLM re-ranking.

Perplexica internally uses SearXNG for retrieval then an LLM (e.g. Qwen3-32B)
to re-rank and synthesize results. The re-ranking is the experimental variable
we are studying.
"""

import time
import random
import requests
import tldextract
from src.config import PERPLEXICA_URL
from src.experiment_context import utcnow_iso


def _extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ""


def discover_providers() -> dict:
    """Call GET /api/providers to discover available models.

    Returns the raw provider list from Perplexica.
    """
    try:
        resp = requests.get(f"{PERPLEXICA_URL}/api/providers", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"  [Perplexica] Failed to discover providers: {e}")
        return {}


def _find_provider_and_model(providers: dict, model_keyword: str) -> tuple:
    """Find a provider ID and model key matching a keyword (e.g. 'qwen3').

    Searches through all providers' chat models for a matching key.
    Returns (provider_id, model_key) or (None, None) if not found.
    """
    model_keyword_lower = model_keyword.lower()
    for provider in providers.get("providers", []):
        for model in provider.get("chatModels", []):
            if model_keyword_lower in model.get("key", "").lower():
                return provider["id"], model["key"]
    return None, None


def _find_embedding_model(providers: dict) -> tuple:
    """Find any available embedding model.

    Returns (provider_id, model_key) or (None, None).
    """
    for provider in providers.get("providers", []):
        embeddings = provider.get("embeddingModels", [])
        if embeddings:
            return provider["id"], embeddings[0]["key"]
    return None, None


def search_perplexica(
    query: str,
    providers: dict,
    chat_provider_id: str,
    chat_model_key: str,
    embed_provider_id: str,
    embed_model_key: str,
    optimization_mode: str = "balanced",
) -> dict:
    """Query Perplexica's /api/search endpoint with a bare keyword.

    The query is passed directly — no sentence wrapping — to keep the
    search as close to a raw engine query as possible. Perplexica's LLM
    then re-ranks the SearXNG results.

    Returns dict with full provenance:
        query, query_timestamp_utc, response_timestamp_utc,
        perplexica_instance, chat_model, embedding_model,
        optimization_mode, raw_response (message + sources),
        ranked_domains, error
    """
    query_time = utcnow_iso()

    result = {
        "query": query,
        "query_timestamp_utc": query_time,
        "perplexica_instance": PERPLEXICA_URL,
        "chat_model": {"provider_id": chat_provider_id, "key": chat_model_key},
        "embedding_model": {"provider_id": embed_provider_id, "key": embed_model_key},
        "optimization_mode": optimization_mode,
        "raw_response": None,
        "sources": [],
        "ranked_domains": [],
        "response_timestamp_utc": None,
        "error": None,
    }

    payload = {
        "chatModel": {
            "providerId": chat_provider_id,
            "key": chat_model_key,
        },
        "embeddingModel": {
            "providerId": embed_provider_id,
            "key": embed_model_key,
        },
        "sources": ["web"],
        "optimizationMode": optimization_mode,
        "query": query,
        "history": [],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{PERPLEXICA_URL}/api/search",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        result["error"] = str(e)
        result["response_timestamp_utc"] = utcnow_iso()
        print(f"  [Perplexica] Error querying '{query}': {e}")
        return result

    result["response_timestamp_utc"] = utcnow_iso()
    result["raw_response"] = data

    # Extract sources with their LLM-assigned order (this IS the re-ranking)
    sources = data.get("sources", [])
    for position, src in enumerate(sources, 1):
        meta = src.get("metadata", {})
        url = meta.get("url", "")
        domain = _extract_domain(url)
        result["sources"].append({
            "position": position,
            "url": url,
            "domain": domain,
            "title": meta.get("title", ""),
            "snippet": src.get("pageContent", "")[:300],
        })

    # Extract unique domains preserving the LLM re-ranked order
    seen = set()
    for s in result["sources"]:
        d = s["domain"]
        if d and d not in seen:
            seen.add(d)
            result["ranked_domains"].append(d)

    # Rate limit
    time.sleep(random.uniform(2, 4))

    return result

"""Thin DataForSEO REST client for the GEODML paper-size experiment.

Uses HTTP Basic auth with login + API password. Bulk endpoints accept up to 1000
targets per request. Responses are returned verbatim; callers are responsible for
extracting the fields they need.
"""

from __future__ import annotations

import os
import time
from typing import Any, Iterable, Sequence

import requests

BASE_URL = "https://api.dataforseo.com"
DEFAULT_TIMEOUT = 60
MAX_TARGETS_PER_REQUEST = 1000


class DataForSEOError(RuntimeError):
    pass


class DataForSEOClient:
    def __init__(
        self,
        login: str | None = None,
        password: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        login = login or os.getenv("DATAFORSEO_LOGIN")
        password = password or os.getenv("DATAFORSEO_PASSWORD")
        if not login or not password:
            raise DataForSEOError(
                "DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD must be set"
            )
        self._auth = (login, password)
        self._timeout = timeout
        self._session = requests.Session()

    def _post(self, path: str, payload: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        resp = self._session.post(
            url, json=payload, auth=self._auth, timeout=self._timeout
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status_code") != 20000:
            raise DataForSEOError(
                f"{path} returned {body.get('status_code')}: {body.get('status_message')}"
            )
        return body

    def bulk_ranks(self, domains: Sequence[str]) -> dict[str, Any]:
        return self._post(
            "/v3/backlinks/bulk_ranks/live",
            [{"targets": list(domains)}],
        )

    def bulk_backlinks(self, domains: Sequence[str]) -> dict[str, Any]:
        return self._post(
            "/v3/backlinks/bulk_backlinks/live",
            [{"targets": list(domains)}],
        )

    def bulk_referring_domains(self, domains: Sequence[str]) -> dict[str, Any]:
        return self._post(
            "/v3/backlinks/bulk_referring_domains/live",
            [{"targets": list(domains)}],
        )

    def bulk_spam_score(self, domains: Sequence[str]) -> dict[str, Any]:
        return self._post(
            "/v3/backlinks/bulk_spam_score/live",
            [{"targets": list(domains)}],
        )

    def bulk_keyword_difficulty(
        self,
        keywords: Sequence[str],
        location_code: int = 2276,
        language_code: str = "en",
    ) -> dict[str, Any]:
        return self._post(
            "/v3/dataforseo_labs/google/bulk_keyword_difficulty/live",
            [
                {
                    "keywords": list(keywords),
                    "location_code": location_code,
                    "language_code": language_code,
                }
            ],
        )

    def keyword_overview(
        self,
        keywords: Sequence[str],
        location_code: int = 2840,
        language_code: str = "en",
    ) -> dict[str, Any]:
        return self._post(
            "/v3/dataforseo_labs/google/keyword_overview/live",
            [
                {
                    "keywords": list(keywords),
                    "location_code": location_code,
                    "language_code": language_code,
                }
            ],
        )

    def google_ads_search_volume(
        self,
        keywords: Sequence[str],
        location_code: int = 2840,
        language_code: str = "en",
        search_partners: bool = False,
    ) -> dict[str, Any]:
        """Google Ads search_volume/live — ground-truth monthly US volumes.

        Covers keywords Labs doesn't have. Max 1000 keywords per task.
        Cost: $0.05 / task. Returns 0 or actual volume (not null) for most.
        """
        return self._post(
            "/v3/keywords_data/google_ads/search_volume/live",
            [
                {
                    "keywords": list(keywords),
                    "location_code": location_code,
                    "language_code": language_code,
                    "search_partners": search_partners,
                }
            ],
        )

    def search_intent(
        self,
        keywords: Sequence[str],
        language_code: str = "en",
    ) -> dict[str, Any]:
        """Labs search_intent/live — main_intent/foreign_intent per keyword.

        Works for keywords missing from keyword_overview. Max 1000 per task.
        Cost: $0.01/task + $0.0001/keyword.
        """
        return self._post(
            "/v3/dataforseo_labs/google/search_intent/live",
            [
                {
                    "keywords": list(keywords),
                    "language_code": language_code,
                }
            ],
        )

    def google_organic_live(
        self,
        keywords: Sequence[str],
        location_code: int = 2840,
        language_code: str = "en",
        depth: int = 20,
    ) -> dict[str, Any]:
        """Google Organic SERP — one task per keyword, max 100 tasks per POST."""
        payload = [
            {
                "keyword": kw,
                "location_code": location_code,
                "language_code": language_code,
                "depth": depth,
            }
            for kw in keywords
        ]
        return self._post("/v3/serp/google/organic/live/regular", payload)


def batched(seq: Sequence[str], n: int = MAX_TARGETS_PER_REQUEST) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def polite_sleep(seconds: float = 0.5) -> None:
    time.sleep(seconds)

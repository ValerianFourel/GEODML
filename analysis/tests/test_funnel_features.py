"""Funnel feature functions: URL and domain rules, visible-text features, page-body features at a fixed date."""

import base64
from datetime import datetime, timezone

import pytest

from analysis.interpretability.pipeline import funnel_features as ff
from analysis.scripts import funnel_features as build


def ad_url(target):
    payload = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    return f"https://www.bing.com/aclick?ld=e8x&u=a1{payload}&rlid=1"


@pytest.mark.parametrize("url,expected", [
    ("HTTPS://WWW.Example.com/Path/", "https://example.com/Path"),
    ("https://example.com/", "https://example.com/"),
    ("http://example.com:80/a?srsltid=xyz&b=2#frag", "http://example.com/a?b=2"),
    ("https://example.com:8443/a", "https://example.com:8443/a"),
    ("https://example.com/a?x=1&y=2", "https://example.com/a?x=1&y=2"),
])
def test_normalize_url(url, expected):
    assert ff.normalize_url(url) == expected


def test_registrable_domain_rules():
    assert ff.registrable_domain("https://help.shopify.com/en/x") == ("shopify.com", "public_suffix_list")
    assert ff.registrable_domain("https://www.bbc.co.uk/news") == ("bbc.co.uk", "public_suffix_list")
    assert ff.registrable_domain("https://x.example.com/a", {"https://x.example.com/a": "Example.COM"}) == ("example.com", "experiment1")
    assert ff.registrable_domain("http://10.1.2.3/page") == ("10.1.2.3", "ip_host")
    domain, source = ff.registrable_domain(ad_url("https://shop.acme.io/landing?a=1"))
    assert domain == "acme.io" and source == "ad_target:public_suffix_list"


def test_bing_ad_decoding():
    assert ff.decode_bing_ad(ad_url("https://acme.io/x")) == "https://acme.io/x"
    assert ff.decode_bing_ad("https://www.bing.com/search?q=x") is None
    assert ff.decode_bing_ad("https://acme.io/aclick?u=a1aGVsbG8") is None


def test_url_features():
    f = ff.url_features("https://blog.hubspot.com/marketing/crm?x=1", "hubspot.com")
    assert f["url_https"] == 1 and f["url_subdomain"] == 1 and f["url_path_depth"] == 2 and f["url_has_query"] == 1
    assert f["url_tld_com"] == 1 and f["url_user_content"] == 0 and f["url_ad_redirect"] == 0
    g = ff.url_features("http://www.reddit.com/r/x/", "reddit.com")
    assert g["url_https"] == 0 and g["url_subdomain"] == 0 and g["url_user_content"] == 1
    assert ff.url_features("https://www.irs.gov/a", "irs.gov")["url_tld_edu_gov"] == 1


def test_snippet_features():
    f = ff.snippet_features("Top 10 CRM tools in 2026?", "HubSpot costs $50 and cuts churn by 20 %.", "hubspot.com", 0)
    assert f["snip_title_question"] == 1 and f["snip_title_listicle"] == 1 and f["snip_year"] == 1
    assert f["snip_currency"] == 1 and f["snip_percent"] == 1 and f["snip_names_domain"] == 1 and f["snip_glued"] == 0
    g = ff.snippet_features("Pricing", "plain words only", "acme.io", 3)
    assert g["snip_title_question"] == 0 and g["snip_digits"] == 0 and g["snip_glued"] == 1 and g["snip_names_domain"] == 0


def page(body_words=200, date='2026-03-01', extra=""):
    words = " ".join(["alpha"] * body_words)
    return (f"<html><head><meta property='article:published_time' content='{date}'></head><body>"
            f"<h2>What is alpha?</h2><p>{words} 35% in 2024 {extra}</p>"
            "<script type='application/ld+json'>{\"@type\": \"FAQPage\"}</script>"
            "<a href='https://acme.io/in'>in</a><a href='https://other.org/out'>out</a></body></html>" + " " * 2048)


def test_body_features_use_the_reference_date_not_the_clock(monkeypatch):
    out = ff.body_features(page(), "acme.io", "alpha alpha beta")
    assert out["body_latest_date"] == "2026-03-01" and round(out["body_age_days"]) == 45 and out["body_freshness"] == 4
    assert out["body_question_headings"] == 1 and out["body_structured_data"] == 1 and out["body_has_date"] == 1
    assert out["body_internal_links"] == 1 and out["body_outbound_links"] == 1 and out["body_words_ok"] == 1
    old = ff.body_features(page(date="2023-01-01"), "acme.io", "alpha")
    assert old["body_freshness"] == 1  # 1,200 days before the reference date
    assert ff.freshness_bucket(None) == 0 and ff.freshness_bucket(-3) == 4 and ff.freshness_bucket(400) == 2


def test_usable_html_rules():
    assert ff.usable_html(None, 0) == (False, "missing")
    assert ff.usable_html("<html>x</html>", 100) == (False, "too_small")
    assert ff.usable_html("<html>Just a moment... checking your browser</html>" + " " * 3000, 4000) == (False, "bot_challenge")
    assert ff.usable_html(page(), 5000) == (True, "ok")


def test_keyword_moderators_and_constants():
    assert ff.keyword_moderators("transactional", 70, (30, 60)) == {"kw_commercial_or_transactional": 1, "kw_difficulty_tercile": 2}
    assert ff.keyword_moderators("informational", None, (30, 60)) == {"kw_commercial_or_transactional": 0, "kw_difficulty_tercile": None}
    assert ff.constant_value([3.0, 3.0, None, 4.0]) == (3.0, 2) and ff.constant_value([None, float("nan")]) == (None, 0)


def test_copy_choice_prefers_usable_copies_nearest_the_reference_date():
    copies = [{"url": "u", "html_source": "exp1:a", "html_fetched": "2026-03-27", "html_usable": 1, "body_word_count": 10,
               "body_internal_links": 1, "body_internal_links_host": 9},
              {"url": "u", "html_source": "exp1:b", "html_fetched": "2026-04-13", "html_usable": 1, "body_word_count": 20,
               "body_internal_links": 2, "body_internal_links_host": 8},
              {"url": "u", "html_source": "gapfill:c", "html_fetched": "2026-05-07", "html_usable": 0, "html_reason": "too_small"},
              {"url": "v", "html_source": "exp1:a", "html_fetched": "2026-04-13", "html_usable": 0, "html_reason": "bot_challenge"}]
    chosen = build.choose_copies(copies, "registrable")
    assert chosen["u"]["body_word_count"] == 20 and chosen["u"]["html_usable_copies"] == 2 and chosen["u"]["html_copies"] == 3
    assert "body_internal_links_host" not in chosen["u"] and chosen["v"]["html_usable"] == 0
    assert build.choose_copies(copies, "host")["u"]["body_internal_links"] == 8

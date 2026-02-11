"""Capture experiment provenance: who, where, when, with what."""

import platform
import socket
import sys
from datetime import datetime, timezone

import requests


def _get_public_ip() -> str:
    """Get the machine's public IP address."""
    for url in ["https://api.ipify.org", "https://ifconfig.me/ip"]:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text.strip()
        except Exception:
            continue
    return "unknown"


def _get_geo_from_ip(ip: str) -> dict:
    """Geolocate an IP address using ip-api.com (free, no key needed)."""
    if ip == "unknown":
        return {"city": "unknown", "region": "unknown", "country": "unknown",
                "lat": None, "lon": None, "isp": "unknown", "query_ip": ip}
    try:
        resp = requests.get(
            f"http://ip-api.com/json/{ip}?fields=status,city,regionName,country,lat,lon,isp,query",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            return {
                "city": data.get("city", "unknown"),
                "region": data.get("regionName", "unknown"),
                "country": data.get("country", "unknown"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "isp": data.get("isp", "unknown"),
                "query_ip": data.get("query", ip),
            }
    except Exception:
        pass
    return {"city": "unknown", "region": "unknown", "country": "unknown",
            "lat": None, "lon": None, "isp": "unknown", "query_ip": ip}


def _get_library_versions() -> dict:
    """Capture versions of key dependencies."""
    versions = {}
    for pkg in ["requests", "huggingface_hub", "duckduckgo_search",
                 "googlesearch", "tldextract", "dotenv"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed, version unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def collect_experiment_context() -> dict:
    """Collect full experiment provenance metadata.

    Call once at the start of each experiment run. Returns a dict
    suitable for embedding in result JSON files.
    """
    ip = _get_public_ip()
    geo = _get_geo_from_ip(ip)

    return {
        "experiment_start_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
        },
        "network": {
            "public_ip": ip,
            "geolocation": geo,
        },
        "library_versions": _get_library_versions(),
    }


def utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string. Use for per-query timestamps."""
    return datetime.now(timezone.utc).isoformat()

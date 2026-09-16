"""Private, authenticated loopback probes for model-serving launchers."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class EndpointSecurityError(RuntimeError):
    """The endpoint fails its access-control contract."""


def new_api_key() -> str:
    return secrets.token_urlsafe(32)


def validate_endpoint(base_url: str) -> None:
    match = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{1,5})/v1", base_url)
    if match is None or not 1 <= int(match.group(1)) <= 65535:
        raise EndpointSecurityError("endpoint must be literal http://127.0.0.1:PORT/v1")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise EndpointSecurityError("endpoint redirect refused")


def probe_endpoint(
    base_url: str, model_id: str, api_key: str, timeout_seconds: float = 2,
) -> dict:
    """Check identity and strict auth rejection without making an inference call."""
    validate_endpoint(base_url)
    if not isinstance(api_key, str) or not api_key or "\r" in api_key or "\n" in api_key:
        raise EndpointSecurityError("a nonempty valid API key is required")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("probe timeout must be positive and finite")
    # Neither inherited proxy settings nor redirects may receive credentials.
    opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(path, key, *, method="GET"):
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        url = base_url + path
        value = Request(url, data=b"{}" if method == "POST" else None,
                        headers=headers, method=method)
        try:
            with opener.open(value, timeout=timeout_seconds) as response:
                status = response.status
                if response.geturl() != url or 300 <= status < 400:
                    raise EndpointSecurityError("endpoint redirect refused")
                body = response.read(1024 * 1024 + 1) if status == 200 and key == api_key else b""
                return status, body
        except HTTPError as error:
            try:
                if 300 <= error.code < 400:
                    raise EndpointSecurityError("endpoint redirect refused") from None
                return error.code, b""
            finally:
                error.close()

    status, body = request("/models", api_key)
    if status in (401, 403):
        raise EndpointSecurityError("endpoint rejected the launcher API key")
    if status != 200:
        raise OSError(f"authenticated model endpoint is not ready: HTTP {status}")
    if len(body) > 1024 * 1024:
        raise EndpointSecurityError("model identity response exceeds size limit")
    try:
        payload = json.loads(body)
        served = {item["id"] for item in payload["data"] if isinstance(item, dict)}
    except (ValueError, TypeError, KeyError):
        raise RuntimeError("invalid model identity response") from None
    if model_id not in served:
        raise RuntimeError("endpoint model identity does not match the configured model")

    wrong_key = new_api_key()
    if wrong_key == api_key:
        wrong_key += "-wrong"
    for label, key in (("anonymous", None), ("wrong-key", wrong_key)):
        for method, path in (("GET", "/models"), ("POST", "/chat/completions")):
            status, _ = request(path, key, method=method)
            if status not in (401, 403):
                raise EndpointSecurityError(
                    f"{label} {method} {path} was not rejected: HTTP {status}"
                )
    return {
        "mode": "per-run-bearer-v1", "status": "verified", "loopback_only": True,
        "anonymous_rejected": True, "wrong_key_rejected": True, "model_verified": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new-key")
    probe = commands.add_parser("probe")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--model-id", required=True)
    probe.add_argument("--timeout-seconds", type=float, default=2)
    args = parser.parse_args(argv)
    if args.command == "new-key":
        print(new_api_key())
        return 0
    try:
        receipt = probe_endpoint(args.base_url, args.model_id,
                                 os.getenv("VLLM_API_KEY", ""), args.timeout_seconds)
    except (EndpointSecurityError, ValueError) as error:
        print(f"ENDPOINT_SECURITY=FAIL {type(error).__name__}")
        return 2
    except (OSError, RuntimeError) as error:
        print(f"ENDPOINT_SECURITY=NOT_READY {type(error).__name__}")
        return 1
    print("ENDPOINT_SECURITY=" + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

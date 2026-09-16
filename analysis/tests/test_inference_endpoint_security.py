"""Exercise authentication on a real loopback HTTP endpoint without any model."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from analysis.scripts import inference_endpoint_security as security


@contextmanager
def endpoint(*, anonymous_status=401, wrong_key_status=None, redirect=False,
             model="fixture/model", echo_error=False):
    key = security.new_api_key()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            authorized = self.headers.get("Authorization") == f"Bearer {key}"
            requests.append((self.command, self.path, authorized, body))
            if redirect:
                self.send_response(307)
                self.send_header("Location", "http://example.invalid/stolen")
                payload = b"redirect"
            elif authorized:
                self.send_response(200)
                payload = json.dumps({"data": [{"id": model}]}).encode()
            else:
                status = wrong_key_status if self.headers.get("Authorization") and wrong_key_status is not None else anonymous_status
                self.send_response(status)
                payload = key.encode() if echo_error else b"denied"
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", key, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("url", [
    "http://0.0.0.0:8010/v1", "http://localhost:8010/v1", "http://127.0.0.2:8010/v1",
    "http://127.0.0.1/v1", "https://127.0.0.1:8010/v1", "http://127.0.0.1:0/v1",
    "http://127.0.0.1:65536/v1", "http://key@127.0.0.1:8010/v1",
    "http://127.0.0.1:8010/v1?key=value", "http://127.0.0.1:8010/v1#fragment",
    "http://127.0.0.1:8010/v1/", "http://127.0.0.1:8010/v1\n",
])
def test_endpoint_rejects_noncanonical_or_nonloopback_urls(url):
    with pytest.raises(security.EndpointSecurityError):
        security.validate_endpoint(url)


def test_key_generation_is_fresh_and_endpoint_check_passes_without_proxy(monkeypatch):
    assert security.new_api_key() != security.new_api_key()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    with endpoint() as (base_url, key, requests):
        receipt = security.probe_endpoint(base_url, "fixture/model", key)
    assert receipt["status"] == "verified"
    assert receipt["anonymous_rejected"] is True
    assert receipt["wrong_key_rejected"] is True
    assert receipt["model_verified"] is True
    assert key not in json.dumps(receipt)
    negatives = [(method, path) for method, path, authorized, _ in requests if not authorized]
    assert negatives.count(("GET", "/v1/models")) == 2
    assert negatives.count(("POST", "/v1/chat/completions")) == 2
    assert all(body == b"{}" for method, _, _, body in requests if method == "POST")


@pytest.mark.parametrize("status", [200, 400, 404, 500])
def test_live_endpoint_without_strict_auth_rejection_is_rejected(status):
    with endpoint(anonymous_status=status) as (base_url, key, _), \
            pytest.raises(security.EndpointSecurityError, match="unauthenticated|anonymous"):
        security.probe_endpoint(base_url, "fixture/model", key)


def test_redirect_does_not_forward_credentials():
    with endpoint(redirect=True) as (base_url, key, requests), \
            pytest.raises(security.EndpointSecurityError, match="redirect"):
        security.probe_endpoint(base_url, "fixture/model", key)
    assert len(requests) == 1


def test_wrong_model_never_counts_as_ready():
    with endpoint(model="other/model") as (base_url, key, _), \
            pytest.raises(RuntimeError, match="model"):
        security.probe_endpoint(base_url, "fixture/model", key)


def test_failure_messages_do_not_echo_response_secrets():
    with endpoint(anonymous_status=200, echo_error=True) as (base_url, key, _):
        with pytest.raises(security.EndpointSecurityError) as caught:
            security.probe_endpoint(base_url, "fixture/model", key)
        assert key not in str(caught.value)


def test_missing_key_fails_before_network():
    with pytest.raises(security.EndpointSecurityError, match="key"):
        security.probe_endpoint("http://127.0.0.1:8010/v1", "fixture/model", "")


def test_endpoint_accepting_wrong_bearer_but_rejecting_anonymous_is_rejected():
    with endpoint(wrong_key_status=200) as (base_url, key, _), \
            pytest.raises(security.EndpointSecurityError, match="wrong-key"):
        security.probe_endpoint(base_url, "fixture/model", key)


def test_valid_probe_key_must_match_server_before_any_negative_probes():
    with endpoint() as (base_url, key, requests), \
            pytest.raises(security.EndpointSecurityError, match="launcher API key"):
        security.probe_endpoint(base_url, "fixture/model", key + "-wrong")
    assert len(requests) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_both_native_rejection_statuses_are_accepted(status):
    with endpoint(anonymous_status=status) as (base_url, key, _):
        assert security.probe_endpoint(base_url, "fixture/model", key)["status"] == "verified"


@pytest.mark.parametrize("mode,expected", [("ready", 0), ("not_ready", 1), ("insecure", 2)])
def test_cli_exit_codes_and_stdout_never_include_key(mode, expected, monkeypatch, capsys):
    key = security.new_api_key()
    monkeypatch.setenv("VLLM_API_KEY", key)

    def probe(base_url, model_id, api_key, timeout_seconds):
        assert api_key == key
        if mode == "not_ready":
            raise OSError("fixture deliberately echoed " + key)
        if mode == "insecure":
            raise security.EndpointSecurityError("fixture deliberately echoed " + key)
        return {"status": "verified"}

    monkeypatch.setattr(security, "probe_endpoint", probe)
    assert security.main(["probe", "--base-url", "http://127.0.0.1:8010/v1",
                          "--model-id", "fixture/model"]) == expected
    assert key not in capsys.readouterr().out


def test_cli_new_key_is_one_line_without_decorations(capsys):
    assert security.main(["new-key"]) == 0
    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert len(output.strip()) >= 32

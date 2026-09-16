import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from analysis.scripts.inference_endpoint_security import EndpointSecurityError
from analysis.scripts.run_acl_arr_vllm import VllmChatClient


def client(**kwargs):
    return VllmChatClient(
        base_url=kwargs.pop("base_url", "http://127.0.0.1:8010/v1"),
        api_key=kwargs.pop("api_key", None),
        server_model_name="test-model",
        timeout_seconds=1,
        maximum_attempts=1,
        **kwargs,
    )


def test_legacy_none_key_inherits_ephemeral_server_credential():
    with patch.dict("os.environ", {"VLLM_API_KEY": "test-key", "GEODML_INFERENCE_AUTH_REQUIRED": "1"}):
        instance = client()
    assert instance.api_key == "test-key"
    captured = {}
    session = SimpleNamespace(close=AsyncMock())

    def create_session(**kwargs):
        captured.update(kwargs)
        return session

    aiohttp = SimpleNamespace(ClientTimeout=lambda **kwargs: None, ClientSession=create_session)
    with patch.dict("sys.modules", {"aiohttp": aiohttp}), patch.object(instance, "verify_server_identity", AsyncMock()):
        asyncio.run(instance.__aenter__())
        asyncio.run(instance.__aexit__(None, None, None))
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["trust_env"] is False


@pytest.mark.parametrize("key", [None, "", "different-key"])
def test_managed_client_cannot_drop_or_override_server_key(key):
    environment = {"VLLM_API_KEY": "" if key is None else "test-key", "GEODML_INFERENCE_AUTH_REQUIRED": "1"}
    with patch.dict("os.environ", environment), pytest.raises(ValueError, match="credential"):
        client(api_key=key)


@pytest.mark.parametrize("base_url", [
    "http://0.0.0.0:8010/v1", "http://example.com/v1", "http://127.0.0.1.evil.test/v1",
    "http://name@127.0.0.1:8010/v1", "http://127.0.0.1:8010/v1?token=x",
])
def test_managed_client_does_not_send_run_key_to_remote_or_ambiguous_endpoint(base_url):
    with (
        patch.dict("os.environ", {"VLLM_API_KEY": "test-key", "GEODML_INFERENCE_AUTH_REQUIRED": "1"}),
        pytest.raises(EndpointSecurityError),
    ):
        client(base_url=base_url)


@pytest.mark.parametrize("escaped", [False, True])
def test_reflected_credential_is_rejected_and_redacted_from_failure_audit(escaped):
    secret = "test-secret-do-not-persist"
    events = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            body = json.dumps({"choices": [{"message": {"content": secret}}]})
            return body.replace(secret, "".join(f"\\u{ord(c):04x}" for c in secret)) if escaped else body

    instance = client(api_key=secret, audit_callback=events.append)
    instance.session = SimpleNamespace(post=lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError) as failure:
        asyncio.run(instance.complete(prompt="query", schema_name="test", schema={}, temperature=0, max_tokens=1, seed=1))
    assert secret not in str(failure.value)
    assert secret not in json.dumps(events)
    assert events[-1]["raw_output"] is None


def test_managed_requests_disable_redirects_for_discovery_and_inference():
    calls = []

    class Response:
        status = 302

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            return "redirect"

    def request(url, **kwargs):
        calls.append(kwargs)
        return Response()

    with patch.dict("os.environ", {"VLLM_API_KEY": "test-key", "GEODML_INFERENCE_AUTH_REQUIRED": "1"}):
        instance = client()
    instance.session = SimpleNamespace(get=request, post=request)
    with pytest.raises(RuntimeError, match="302"):
        asyncio.run(instance.verify_server_identity())
    with pytest.raises(RuntimeError, match="302"):
        asyncio.run(instance.complete(prompt="query", schema_name="test", schema={}, temperature=0, max_tokens=1, seed=1))
    assert len(calls) == 2
    assert all(row["allow_redirects"] is False for row in calls)


def test_discovery_exception_redacts_credential_and_closes_session():
    secret = "test-discovery-secret"
    instance = client(api_key=secret)
    session = SimpleNamespace(close=AsyncMock())
    aiohttp = SimpleNamespace(ClientTimeout=lambda **kw: None, ClientSession=lambda **kw: session)
    with (
        patch.dict("sys.modules", {"aiohttp": aiohttp}),
        patch.object(instance, "verify_server_identity", AsyncMock(side_effect=RuntimeError(secret))),
        pytest.raises(RuntimeError) as failure,
    ):
        asyncio.run(instance.__aenter__())
    assert secret not in str(failure.value)
    session.close.assert_awaited_once()


def test_transport_exception_cannot_write_credential_to_audit_or_raise_it():
    secret = "test-transport-secret"
    events = []

    def fail(*args, **kwargs):
        raise RuntimeError("Authorization: Bearer " + secret)

    instance = client(api_key=secret, audit_callback=events.append)
    instance.session = SimpleNamespace(post=fail)
    with pytest.raises(RuntimeError) as failure:
        asyncio.run(instance.complete(prompt="query", schema_name="test", schema={}, temperature=0, max_tokens=1, seed=1))
    assert secret not in str(failure.value)
    assert secret not in json.dumps(events)


@pytest.mark.parametrize("status", [401, 500])
@pytest.mark.parametrize("operation", ["discovery", "inference"])
def test_error_response_bodies_cannot_publish_encoded_credentials(status, operation):
    secret = "test-error-secret"
    encoded = "".join(f"\\u{ord(c):04x}" for c in secret)
    events = []

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            return '{"error":"' + encoded + '"}'

    Response.status = status
    instance = client(api_key=secret, audit_callback=events.append)
    instance.session = SimpleNamespace(get=lambda *a, **kw: Response(), post=lambda *a, **kw: Response())
    with pytest.raises(RuntimeError) as failure:
        if operation == "discovery":
            asyncio.run(instance.verify_server_identity())
        else:
            asyncio.run(instance.complete(prompt="query", schema_name="test", schema={}, temperature=0, max_tokens=1, seed=1))
    assert encoded not in str(failure.value)
    assert secret not in str(failure.value)
    if events:
        assert events[-1]["response_body"] is None


def test_successful_model_identity_cannot_publish_json_escaped_credential():
    secret = "test-model-secret"
    encoded = "".join(f"\\u{ord(c):04x}" for c in secret)

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def text(self):
            return '{"data":[{"id":"' + encoded + '"}]}'

    instance = client(api_key=secret)
    instance.session = SimpleNamespace(get=lambda *a, **kw: Response())
    with pytest.raises(RuntimeError) as failure:
        asyncio.run(instance.verify_server_identity())
    assert secret not in str(failure.value)
    assert encoded not in str(failure.value)

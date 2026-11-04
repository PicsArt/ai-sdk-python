from typing import Any

import httpx
import pytest
import respx

from picsart_ai_sdk.abort import AbortController, AbortError
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.client.errors import to_api_error
from picsart_ai_sdk.client.fetch import FetchInit
from picsart_ai_sdk.client.transport import build_transport, create_workflows_client, maybe_fetch
from picsart_ai_sdk.client.workflows import WorkflowsError
from picsart_ai_sdk.core.errors import ApiError
from picsart_ai_sdk.core.workflow import TransportPollOptions, WorkflowJobHandle, WorkflowSubmitRequest, has_method

BASE = "https://api.test"
WF = f"{BASE}/workflows/v1/models"


def ok(response: object) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "response": response})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(workflows, "_sleep", fake_sleep)


@pytest.fixture
async def http() -> Any:
    client = httpx.AsyncClient()
    yield client
    await client.aclose()


@respx.mock
async def test_api_key_fetch_sets_auth_and_gateway_headers(http: httpx.AsyncClient) -> None:
    route = respx.post(f"{WF}/submit").mock(return_value=ok({"id": "t"}))
    fetch = maybe_fetch(fetch=None, api_key="Bearer abc", http=http)
    assert fetch is not None
    await build_transport(create_workflows_client(BASE, fetch)).submit(WorkflowSubmitRequest("v1/models", {}))
    headers = route.calls[0].request.headers
    assert headers["authorization"] == "Bearer abc"
    assert headers["platform"] == "api" and headers["x-touchpoint"] == "sdk"
    assert headers["x-config-id"] == ""


@respx.mock
async def test_caller_gateway_header_wins(http: httpx.AsyncClient) -> None:
    route = respx.get(f"{BASE}/x").mock(return_value=httpx.Response(200))
    fetch = maybe_fetch(fetch=None, api_key="k", http=http)
    assert fetch is not None
    await fetch(f"{BASE}/x", {"method": "GET", "headers": {"platform": "test"}})
    assert route.calls[0].request.headers["platform"] == "test"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_custom_fetch_owns_headers(http: httpx.AsyncClient) -> None:
    seen: list[FetchInit] = []

    async def mine(url: str, init: FetchInit) -> httpx.Response:
        seen.append(init)
        return ok({"id": "t"})

    assert maybe_fetch(fetch=mine, api_key="ignored", http=http) is mine
    assert maybe_fetch(fetch=None, api_key=None, http=http) is None
    await build_transport(create_workflows_client(BASE, mine)).submit(WorkflowSubmitRequest("v1/models", {}))
    assert "authorization" not in seen[0]["headers"] and "platform" not in seen[0]["headers"]


async def fetch_returning(response: httpx.Response) -> Any:
    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        return response

    return build_transport(create_workflows_client(BASE, fetch))


async def test_submit_without_id_is_invalid_response() -> None:
    transport = await fetch_returning(ok({"id": ""}))
    with pytest.raises(ApiError) as exc:
        await transport.submit(WorkflowSubmitRequest("v1/models", {}))
    assert (exc.value.message, exc.value.status, exc.value.code) == ("No task id in response", 502, "invalid_response")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (ok({"credits": 5, "toolId": "t"}), {"credits": 5, "toolId": "t"}),
        (ok({"credits": "5"}), None),
        (ok({"credits": True}), None),
        (httpx.Response(500, json={}), None),
    ],
)
async def test_options_is_best_effort(response: httpx.Response, expected: object) -> None:
    transport = await fetch_returning(response)
    assert await transport.options("v1/models", {"model": "m"}) == expected


@respx.mock
async def test_poll_maps_options_and_progress(http: httpx.AsyncClient) -> None:
    respx.get(f"{WF}/t/result").mock(
        side_effect=[
            ok({"status": "IN_PROGRESS", "updated": "2026-01-01T00:00:00Z", "progress": {"percent": 1}}),
            ok(
                {
                    "status": "COMPLETED",
                    "updated": "2026-01-01T00:00:01Z",
                    "result": {"url": "u"},
                    "usage": {"credits": 3},
                }
            ),
        ]
    )
    fetch = maybe_fetch(fetch=None, api_key="k", http=http)
    assert fetch is not None
    progress: list[Any] = []
    res = await build_transport(create_workflows_client(BASE, fetch)).poll(
        WorkflowJobHandle("v1/models", "t"), TransportPollOptions(interval_ms=1, on_progress=progress.append)
    )
    assert (res.result, res.usage, res.raw) == ({"url": "u"}, {"credits": 3}, {"url": "u"})
    assert progress == [{"percent": 1}]


async def test_poll_client_timeout_maps_to_408() -> None:
    transport = await fetch_returning(ok({"status": "FAILED", "result": None}))
    with pytest.raises(ApiError) as exc:
        await transport.poll(WorkflowJobHandle("v1/models", "t1"))
    assert (exc.value.message, exc.value.status, exc.value.code) == (
        "Timed out waiting for workflow v1/models:t1",
        408,
        "timeout",
    )


async def test_status_after_abort() -> None:
    transport = await fetch_returning(ok({}))
    c = AbortController()
    c.abort()
    with pytest.raises(ApiError) as exc:
        await transport.status(WorkflowJobHandle("v1/models", "t"), c.signal)
    assert (exc.value.status, exc.value.code) == (499, "aborted")


async def test_transport_methods_are_detected() -> None:
    transport = await fetch_returning(ok({}))
    assert all(has_method(transport, m) for m in ("execute", "submit", "poll", "status", "options"))
    assert not has_method(object(), "submit")


@pytest.mark.parametrize(
    ("err", "status", "code", "message"),
    [
        (WorkflowsError(reason="request_failed", message="m", http_status_code=503), 503, "service_unavailable", "m"),
        (WorkflowsError(reason="unknown_error", message="m"), 502, "bad_gateway", "m"),
        (WorkflowsError(reason="nsfw", message="flagged", http_status_code=422), 422, "nsfw", "flagged"),
        (WorkflowsError(reason="connection_error", message="down"), 502, "connection_error", "down"),
        (RuntimeError("boom"), 502, "generation_failed", "boom"),
    ],
)
def test_to_api_error(err: BaseException, status: int, code: str, message: str) -> None:
    mapped = to_api_error(err, "v1/models")
    assert isinstance(mapped, ApiError)
    assert (mapped.status, mapped.code, mapped.message) == (status, code, message)


def test_to_api_error_passthrough_and_timeout() -> None:
    api = ApiError("x", status=400, code="validation_error")
    abort = AbortError()
    assert to_api_error(api, "w") is api and to_api_error(abort, "w") is abort
    timeout = to_api_error(WorkflowsError(reason="client_timeout", message="", http_status_code=408), "v1/models")
    assert isinstance(timeout, ApiError) and timeout.message == "Timed out waiting for workflow v1/models"

"""Regressions found by the stage-1 review — each reproduces a divergence from the TS SDK."""

import asyncio
import json
import pickle
import re
from typing import Any

import httpx
import pytest
import respx

from picsart_ai_sdk import ApiError, GenerationEventType, GenerationOptions, create_client
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.client.workflows import _parse_updated
from picsart_ai_sdk.core.js import js_json_dumps

API = "https://api.test"
WF = f"{API}/workflows/v1/models"


def ok(response: object) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "response": response})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(workflows, "_sleep", fake_sleep)


@respx.mock
async def test_request_body_is_json_stringify_output() -> None:
    route = respx.post(f"{WF}/submit").mock(return_value=ok({"id": "t"}))
    ai = create_client(api_url=API, api_key="k")
    await ai.submit("m", {"prompt": "x", "duration": 5.0, "g": float("nan"), "s": 1e-7, "big": 1e21})
    body = route.calls[0].request.content.decode()
    assert '"duration":5,' in body and '"g":null' in body and '"s":1e-7' in body and '"big":1e+21' in body
    await ai.aclose()


def test_js_number_formatting() -> None:
    assert js_json_dumps([1e-7, 1.5e-10, 1e21, 1.2e22, 0.1, 100.0, -0.0, 123456789.123]) == (
        "[1e-7,1.5e-10,1e+21,1.2e+22,0.1,100,0,123456789.123]"
    )


@respx.mock
async def test_subscribe_break_without_aclose_stops_polling_with_context_manager() -> None:
    counter = iter(range(10_000))

    def frame(_: httpx.Request) -> httpx.Response:
        return ok(
            {"status": "IN_PROGRESS", "updated": f"2026-01-01T00:00:{next(counter) % 60:02d}Z", "progress": {"p": 1}}
        )

    route = respx.get(url__regex=rf"{re.escape(WF)}/t/result").mock(side_effect=frame)
    ai = create_client(api_url=API, api_key="k")
    async with ai.subscribe("m", "t", GenerationOptions(interval_ms=1)) as events:
        async for event in events:
            assert event.type == GenerationEventType.Progress
            break
    polls = route.call_count
    for _ in range(30):
        await asyncio.sleep(0)
    assert route.call_count == polls
    await ai.aclose()


@respx.mock
async def test_empty_progress_object_is_still_progress() -> None:
    respx.get(url__regex=rf"{re.escape(WF)}/t/result").mock(
        side_effect=[
            ok({"status": "IN_PROGRESS", "updated": "2026-01-01T00:00:01Z", "progress": {}}),
            ok({"status": "COMPLETED", "updated": "2026-01-01T00:00:02Z", "result": {"url": "u"}}),
        ]
    )
    ai = create_client(api_url=API, api_key="k")
    events = [e async for e in ai.subscribe("m", "t", GenerationOptions(interval_ms=1))]
    assert [e.type for e in events] == [GenerationEventType.Progress, GenerationEventType.Completed]
    await ai.aclose()


@respx.mock
async def test_submit_without_id_is_invalid_response() -> None:
    respx.post(f"{WF}/submit").mock(return_value=ok({}))
    ai = create_client(api_url=API, api_key="k")
    with pytest.raises(ApiError) as exc:
        await ai.submit("m", {"prompt": "x"})
    assert (exc.value.message, exc.value.status, exc.value.code) == ("No task id in response", 502, "invalid_response")
    await ai.aclose()


@respx.mock
async def test_keyboard_interrupt_is_not_swallowed() -> None:
    respx.post(f"{WF}/submit").mock(return_value=ok({"id": "t"}))
    respx.get(url__regex=rf"{re.escape(WF)}/t/result").mock(
        return_value=ok({"status": "IN_PROGRESS", "updated": "2026-01-01T00:00:01Z", "progress": {"p": 1}})
    )

    def stop(_: Any) -> None:
        raise KeyboardInterrupt

    wc = create_client(api_url=API, api_key="k")._workflows
    assert wc is not None
    with pytest.raises(KeyboardInterrupt):
        await wc.run_polling("v1/models", "t", on_progress=stop)


@respx.mock
async def test_read_timeout_is_retried_as_a_connection_error() -> None:
    respx.get(url__regex=rf"{re.escape(WF)}/t/result").mock(
        side_effect=[httpx.ReadTimeout("slow")] * 3 + [ok({"status": "COMPLETED", "result": {"url": "u"}})]
    )
    wc = create_client(api_url=API, api_key="k")._workflows
    assert wc is not None
    assert (await wc.run_polling("v1/models", "t"))["result"] == {"url": "u"}


def test_default_http_client_has_undici_like_timeouts() -> None:
    ai = create_client(api_url=API, api_key="k")
    assert ai._http is not None
    assert (ai._http.timeout.read, ai._http.timeout.connect) == (300.0, None)


def test_updated_parsing_matches_js_dates() -> None:
    a = _parse_updated("2026-01-01T00:00:01.12Z")
    b = _parse_updated("2026-01-01T00:00:01.120Z")
    assert a == b and a == a  # not NaN
    assert _parse_updated("2026-01-01T00:00:01.1234567Z") == _parse_updated("2026-01-01T00:00:01.123Z")
    assert _parse_updated(1767225601000) == _parse_updated("2026-01-01T00:00:01Z")  # epoch ms, like new Date(n)


def test_api_error_pickles() -> None:
    e = pickle.loads(pickle.dumps(ApiError("boom", status=429, code="rate_limited")))
    assert (e.message, e.status, e.code, e.reason) == ("boom", 429, "rate_limited", "rate_limited")
    assert json.dumps({"x": 1})

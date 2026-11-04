import json
from typing import Any

import httpx
import pytest
import respx

from picsart_ai_sdk.abort import AbortController, AbortError
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.client.fetch import FetchInit, send
from picsart_ai_sdk.client.workflows import WorkflowsClient, WorkflowsError, polling_delay

BASE = "https://api.test"
WF = f"{BASE}/workflows/v1/models"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(workflows, "_sleep", fake_sleep)
    return calls


@pytest.fixture
async def client() -> Any:
    http = httpx.AsyncClient()
    seen: list[tuple[str, FetchInit]] = []

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        seen.append((url, init))
        return await send(http, url, init)

    wc = WorkflowsClient(BASE, fetch)
    wc.seen = seen  # type: ignore[attr-defined]
    yield wc
    await http.aclose()


def ok(response: object) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "response": response})


@respx.mock
async def test_submit_posts_params_and_returns_id(client: Any) -> None:
    route = respx.post(f"{WF}/submit").mock(return_value=ok({"id": "task-1"}))
    assert await client.submit("v1/models", {"model": "m", "params": {"prompt": "p"}}) == "task-1"
    assert route.calls[0].request.content == b'{"params":{"model":"m","params":{"prompt":"p"}}}'
    url, init = client.seen[0]
    assert url == f"{WF}/submit"
    assert init["headers"] == {"x-config-id": "", "accept": "application/json", "content-type": "application/json"}


@respx.mock
async def test_result_get_has_no_config_id(client: Any) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=ok({"id": "t", "status": "COMPLETED", "result": {"url": "u"}}))
    await client.get_result("v1/models", "t")
    assert client.seen[0][1]["headers"] == {"accept": "application/json", "content-type": "application/json"}


def test_base_url_gets_trailing_slash() -> None:
    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        raise AssertionError

    assert WorkflowsClient("https://x", fetch).workflows_api_base_url == "https://x/workflows"
    assert WorkflowsClient("https://x/", fetch).workflows_api_base_url == "https://x/workflows"


@respx.mock
async def test_run_polling_until_completed_with_progress_once_per_update(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/task-1/result").mock(
        side_effect=[
            ok(
                {
                    "id": "task-1",
                    "status": "IN_PROGRESS",
                    "updated": "2026-01-01T00:00:01Z",
                    "progress": {"percent": 10},
                }
            ),
            ok(
                {
                    "id": "task-1",
                    "status": "IN_PROGRESS",
                    "updated": "2026-01-01T00:00:01Z",
                    "progress": {"percent": 10},
                }
            ),
            ok(
                {
                    "id": "task-1",
                    "status": "IN_PROGRESS",
                    "updated": "2026-01-01T00:00:02Z",
                    "progress": {"percent": 50},
                }
            ),
            ok({"id": "task-1", "status": "COMPLETED", "updated": "2026-01-01T00:00:03Z", "result": {"url": "u"}}),
        ]
    )
    progress: list[Any] = []
    res = await client.run_polling("v1/models", "task-1", polling_interval=7, on_progress=progress.append)
    assert res["result"] == {"url": "u"}
    assert progress == [{"percent": 10}, {"percent": 50}]
    assert sleeps == [0.007] * 4  # sleep BEFORE every poll


@respx.mock
async def test_zero_interval_and_retries_fall_back_to_defaults(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=ok({"status": "COMPLETED", "result": 1}))
    await client.run_polling("v1/models", "t", polling_interval=0, retries_count=0)
    assert sleeps == [0.3]


def test_polling_delay_backoff() -> None:
    assert [polling_delay(300, n) for n in range(5)] == [300, 600, 1200, 2400, 4800]
    assert polling_delay(300, 6) == 5000
    assert polling_delay(8000, 3) == 8000


@respx.mock
async def test_ten_connection_errors_then_connection_error(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(WorkflowsError) as exc:
        await client.run_polling("v1/models", "t")
    assert (exc.value.reason, exc.value.http_status_code, exc.value.message) == ("connection_error", None, "boom")
    assert len(sleeps) == 11


@respx.mock
async def test_nine_connection_errors_then_success(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(
        side_effect=[httpx.ConnectError("x")] * 9 + [ok({"status": "COMPLETED", "result": {"ok": True}})]
    )
    assert (await client.run_polling("v1/models", "t"))["result"] == {"ok": True}


@respx.mock
async def test_http_error_on_poll_is_raised_at_once(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=httpx.Response(404, json={"reason": "not_found", "message": "gone"}))
    with pytest.raises(WorkflowsError) as exc:
        await client.run_polling("v1/models", "t")
    assert (exc.value.reason, exc.value.http_status_code, exc.value.message) == ("not_found", 404, "gone")
    assert len(sleeps) == 1


@respx.mock
async def test_terminal_without_result_is_client_timeout(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=ok({"status": "FAILED", "result": None}))
    with pytest.raises(WorkflowsError) as exc:
        await client.run_polling("v1/models", "t")
    assert (exc.value.reason, exc.value.http_status_code) == ("client_timeout", 408)


@respx.mock
async def test_retries_exhausted_is_client_timeout(client: Any, sleeps: list[float]) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=ok({"status": "IN_PROGRESS"}))
    with pytest.raises(WorkflowsError) as exc:
        await client.run_polling("v1/models", "t", retries_count=3)
    assert exc.value.reason == "client_timeout" and len(sleeps) == 3


@respx.mock
async def test_non_json_error_body(client: Any) -> None:
    respx.post(f"{WF}/submit").mock(return_value=httpx.Response(502, text="<html>Bad gateway</html>"))
    with pytest.raises(WorkflowsError) as exc:
        await client.submit("v1/models", {})
    assert (exc.value.reason, exc.value.http_status_code, exc.value.message) == (
        "request_failed",
        502,
        "Non json response was returned from server",
    )


@respx.mock
async def test_json_error_body_without_message(client: Any) -> None:
    respx.post(f"{WF}/submit").mock(return_value=httpx.Response(500, json={"status": "error"}))
    with pytest.raises(WorkflowsError) as exc:
        await client.submit("v1/models", {})
    assert (exc.value.reason, exc.value.message) == ("request_failed", "Unknown error")


@respx.mock
async def test_ok_non_json_body_is_invalid_response(client: Any) -> None:
    respx.post(f"{WF}/execute").mock(return_value=httpx.Response(200, text="nope"))
    with pytest.raises(WorkflowsError) as exc:
        await client.execute_sync("v1/models", {})
    assert (exc.value.reason, exc.value.http_status_code) == ("invalid_response", 200)


@respx.mock
async def test_submit_network_failure_becomes_unknown_error(client: Any) -> None:
    respx.post(f"{WF}/submit").mock(side_effect=httpx.ConnectError("offline"))
    with pytest.raises(WorkflowsError) as exc:
        await client.submit("v1/models", {})
    assert (exc.value.reason, exc.value.message, exc.value.http_status_code) == ("unknown_error", "offline", None)


@respx.mock
async def test_abort_between_polls(client: Any, sleeps: list[float]) -> None:
    controller = AbortController()

    def poll(_: httpx.Request) -> httpx.Response:
        controller.abort()
        return ok({"status": "IN_PROGRESS"})

    respx.get(f"{WF}/t/result").mock(side_effect=poll)
    with pytest.raises(AbortError):
        await client.run_polling("v1/models", "t", signal=controller.signal)
    assert len(sleeps) == 1


@respx.mock
async def test_options_and_execute_unwrap_response(client: Any) -> None:
    respx.post(f"{WF}/options").mock(return_value=ok({"credits": 5}))
    respx.post(f"{WF}/execute").mock(return_value=ok({"result": {"text": "hi"}, "usage": {"credits": 1}}))
    assert await client.options("v1/models", {"model": "m"}) == {"credits": 5}
    assert (await client.execute_sync("v1/models", {}))["result"] == {"text": "hi"}


@respx.mock
async def test_stream_events(client: Any) -> None:
    body = (
        'data: {"type":"event.metrics","n":1}\n\n'
        'data: {"type":"task.partial-result","result":"pa"}\r\r'
        'data: {"type":"task.completed","result":{"text":"done"},"usage":{"credits":2}}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx.post(f"{WF}/stream").mock(return_value=httpx.Response(200, text=body))
    events: list[Any] = []
    partial: list[Any] = []
    res = await client.stream("v1/models", {}, on_event=events.append, on_partial_result=partial.append)
    assert res == {"result": {"text": "done"}, "usage": {"credits": 2}, "status": "COMPLETED"}
    assert events == [{"type": "metrics", "n": 1}]
    assert partial == [{"status": "IN_PROGRESS", "result": "pa"}]
    assert route.calls[0].request.headers["accept"] == "text/event-stream"


@respx.mock
async def test_stream_failed_event(client: Any) -> None:
    respx.post(f"{WF}/stream").mock(
        return_value=httpx.Response(200, text='data: {"type":"task.failed","result":{"reason":"nsfw"}}\n\n')
    )
    with pytest.raises(WorkflowsError) as exc:
        await client.stream("v1/models", {}, on_event=lambda e: None)
    assert (exc.value.reason, exc.value.message, exc.value.http_status_code) == ("nsfw", "Workflow failed", 500)


@respx.mock
async def test_run_async_submits_then_polls(client: Any, sleeps: list[float]) -> None:
    respx.post(f"{WF}/submit").mock(return_value=ok({"id": "t9"}))
    respx.get(f"{WF}/t9/result").mock(return_value=ok({"id": "t9", "status": "COMPLETED", "result": {"a": 1}}))
    accepted: list[str] = []
    res = await client.run("v1/models", {"x": 1}, on_accepted=accepted.append)
    assert res["result"] == {"a": 1} and accepted == ["t9"]


def test_from_body_falls_back_per_field() -> None:
    e = WorkflowsError.from_body(
        {"message": "m"}, http_status_code=500, reason="workflow_failed", message="Workflow failed"
    )
    assert (e.reason, e.message, e.http_status_code) == ("workflow_failed", "m", 500)
    e = WorkflowsError.from_body({"statusCode": 422, "reason": "r"}, http_status_code=500, reason="x", message="y")
    assert (e.reason, e.http_status_code) == ("r", 422)
    assert json.dumps({"params": {"é": 1}}, ensure_ascii=False)  # sanity for the body encoding


@pytest.mark.parametrize("empty", [{}, []])
@respx.mock
async def test_polling_returns_empty_container_result(client: Any, sleeps: list[float], empty: object) -> None:
    respx.get(f"{WF}/t/result").mock(return_value=ok({"id": "t", "status": "COMPLETED", "result": empty}))
    res = await client.run_polling("v1/models", "t")
    assert res["result"] == empty


@respx.mock
async def test_run_async_does_not_submit_when_already_aborted(client: Any, sleeps: list[float]) -> None:
    route = respx.post(f"{WF}/submit").mock(return_value=ok({"id": "t"}))
    controller = AbortController()
    controller.abort()
    with pytest.raises(AbortError):
        await client.run("v1/models", {}, signal=controller.signal)
    assert not route.called


@respx.mock
async def test_stream_abort_from_on_event_raises(client: Any) -> None:
    body = 'data: {"type":"event.metrics","n":1}\n\ndata: {"type":"task.completed","result":{"text":"done"}}\n\n'
    respx.post(f"{WF}/stream").mock(return_value=httpx.Response(200, text=body))
    controller = AbortController()
    with pytest.raises(AbortError):
        await client.stream("v1/models", {}, on_event=lambda e: controller.abort(), signal=controller.signal)


async def test_stream_abort_while_waiting_for_body_raises() -> None:
    import asyncio

    controller = AbortController()
    closed = asyncio.Event()

    class Hanging(httpx.AsyncByteStream):
        async def __aiter__(self) -> Any:
            yield b'data: {"type":"event.metrics","n":1}\n\n'
            await asyncio.Event().wait()  # the server goes quiet
            yield b""

        async def aclose(self) -> None:
            closed.set()

    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Hanging())))
    wc = WorkflowsClient(BASE, lambda url, init: send(http, url, init))
    events: list[Any] = []

    def on_event(e: Any) -> None:
        events.append(e)
        asyncio.get_running_loop().call_later(0.01, controller.abort)

    with pytest.raises(AbortError):
        await asyncio.wait_for(wc.stream("v1/models", {}, on_event=on_event, signal=controller.signal), 2)
    assert events == [{"type": "metrics", "n": 1}] and closed.is_set()
    await http.aclose()

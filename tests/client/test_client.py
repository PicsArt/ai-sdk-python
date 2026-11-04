"""Port of ``__tests__/unit/dynamic-mode.test.ts`` (+ ``helpers/mock-platform.ts`` as a respx fixture)."""

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import respx

from picsart_ai_sdk import (
    AbortController,
    AiClient,
    ApiError,
    DriveConfig,
    GenerateOptions,
    GenerationEventType,
    GenerationOptions,
    InputsTransformation,
    TransportResult,
    WorkflowSubmitRequest,
    create_client,
)
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.model_types import Flux2ProInput

API = "https://api.test"
WF = f"{API}/workflows/v1/models"
FAST = GenerateOptions(interval_ms=1)
FAST_POLL = GenerationOptions(interval_ms=1)


@dataclass
class Platform:
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    submits: list[dict[str, Any]] = field(default_factory=list)
    polls: int = 0
    options_calls: list[dict[str, Any]] = field(default_factory=list)
    headers: list[httpx.Headers] = field(default_factory=list)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(workflows, "_sleep", fake_sleep)


@pytest.fixture
def platform() -> Any:
    p = Platform()

    def submit(request: httpx.Request) -> httpx.Response:
        p.headers.append(request.headers)
        p.submits.append(json.loads(request.content)["params"])
        return httpx.Response(200, json={"status": "success", "response": {"id": "task-1"}})

    def result(request: httpx.Request, task_id: str) -> httpx.Response:
        p.polls += 1
        frame = p.results.get(task_id, {"status": "COMPLETED", "result": {"url": "https://cdn.test/out.jpg"}})
        return httpx.Response(200, json={"status": "success", "response": {"id": task_id, **frame}})

    def options(request: httpx.Request) -> httpx.Response:
        p.options_calls.append(json.loads(request.content)["params"])
        return httpx.Response(200, json={"status": "success", "response": {"credits": 5, "toolId": "t"}})

    with respx.mock(assert_all_called=False) as router:
        router.post(f"{WF}/submit").mock(side_effect=submit)
        router.get(url__regex=rf"{re.escape(WF)}/(?P<task_id>[^/]+)/result").mock(side_effect=result)
        router.post(f"{WF}/options").mock(side_effect=options)
        yield p


@pytest.fixture
async def ai() -> Any:
    client = create_client(api_url=API, api_key="key", drive=DriveConfig(folder="AI Tests"))
    yield client
    await client.aclose()


async def test_generate_without_catalog_sends_the_models_command(ai: Any, platform: Platform) -> None:
    res = await ai.generate("backend-only-model", {"prompt": "a dog", "someNewParam": 42}, FAST)
    submit = platform.submits[-1]
    assert submit["model"] == "backend-only-model"
    assert submit["params"] == {"prompt": "a dog", "someNewParam": 42}
    assert submit["options"]["inputs_transformation"] == {"downscale_oversized_images": False}
    drive = submit["options"]["drive"]
    assert drive["folder"] == {"path": "AI Tests"}
    assert re.fullmatch(r"a-dog-\d+", drive["name"])
    assert drive["attributes"]["model"] == "backend-only-model"
    assert json.loads(drive["attributes"]["aiSDKPayload"]) == {"prompt": "a dog", "someNewParam": 42}
    assert (res.url, res.generation_id) == ("https://cdn.test/out.jpg", "task-1")
    headers = platform.headers[-1]
    assert (headers["authorization"], headers["platform"], headers["x-touchpoint"]) == ("Bearer key", "api", "sdk")


async def test_format_param_names_the_file(ai: Any, platform: Platform) -> None:
    await ai.generate("backend-only-model", {"prompt": "a dog", "outputFormat": "mp4"}, FAST)
    assert re.fullmatch(r"a-dog-\d+\.mp4", platform.submits[-1]["options"]["drive"]["name"])


async def test_items_metadata_and_usage_pass_through(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {
        "status": "COMPLETED",
        "result": {
            "items": [
                {"url": "https://cdn.test/a.jpg", "metadata": {"exploreImageId": "e1"}},
                {"url": "https://cdn.test/b.jpg"},
            ],
            "model": "backend-only-model",
        },
        "usage": {"credits": 3},
    }
    res = await ai.generate("backend-only-model", {"prompt": "x"}, FAST)
    assert len(res.items) == 2 and res.url == "https://cdn.test/a.jpg"
    assert res.items[0].metadata == {"exploreImageId": "e1"} and res.items[1].metadata is None
    assert res.usage == {"credits": 3}


async def test_missing_url_is_invalid_response(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {"status": "COMPLETED", "result": {"model": "x"}}
    with pytest.raises(ApiError) as exc:
        await ai.generate("backend-only-model", {"prompt": "x"}, FAST)
    assert (exc.value.status, exc.value.code) == (502, "invalid_response")


async def test_generate_text_sends_no_options(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {"status": "COMPLETED", "result": {"text": "hello", "model": "claude-opus-5"}}
    res = await ai.generate_text("backend-only-llm", {"prompt": "hi"}, FAST)
    assert (res.text, res.model) == ("hello", "claude-opus-5")
    assert "options" not in platform.submits[-1]


async def test_run(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {"status": "COMPLETED", "result": {"answer": 42}, "usage": {"credits": 1}}
    res = await ai.run("json-model", {"prompt": "x"}, FAST)
    assert (res.result, res.status, res.id, res.usage) == ({"answer": 42}, "COMPLETED", "task-1", {"credits": 1})
    assert "options" not in platform.submits[-1]


async def test_submit_result_subscribe(ai: Any, platform: Platform) -> None:
    task_id = await ai.submit("backend-only-model", {"prompt": "x"})
    assert task_id == "task-1"
    before = platform.polls
    res = await ai.result("backend-only-model", task_id, FAST_POLL)
    assert res.generation_id == "task-1" and platform.polls == before + 1
    events = [e async for e in ai.subscribe("backend-only-model", task_id, FAST_POLL)]
    assert [e.type for e in events] == [GenerationEventType.Completed]
    assert events[0].result.url == "https://cdn.test/out.jpg"


async def test_subscribe_progress_then_completed(ai: Any, platform: Platform) -> None:
    frames = iter(
        [
            {"status": "IN_PROGRESS", "updated": "2026-01-01T00:00:01Z", "progress": {"percent": 30}},
            {"status": "COMPLETED", "updated": "2026-01-01T00:00:02Z", "result": {"url": "https://cdn.test/v.mp4"}},
        ]
    )

    class Seq(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            return next(frames)

    platform.results = Seq()
    events = [e async for e in ai.subscribe("video-model", "task-7", FAST_POLL)]
    assert [e.type for e in events] == [GenerationEventType.Progress, GenerationEventType.Completed]
    assert events[0].progress == {"percent": 30}


async def test_subscribe_failure_is_an_event(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {
        "status": "FAILED",
        "result": {"status": "FAILED", "reason": "nsfw", "message": "flagged"},
    }
    events = [e async for e in ai.subscribe("m", "task-1", FAST_POLL)]
    assert [e.type for e in events] == [GenerationEventType.Failed]
    assert (events[0].error.code, events[0].error.message) == ("nsfw", "m failed: flagged")


async def test_leaving_subscribe_early_stops_polling(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {"status": "IN_PROGRESS", "progress": {"percent": 1}}
    # a changing `updated` makes every poll a progress event
    counter = iter(range(10_000))

    class Changing(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            return {
                "status": "IN_PROGRESS",
                "updated": f"2026-01-01T00:00:{next(counter) % 60:02d}Z",
                "progress": {"percent": 1},
            }

    platform.results = Changing()
    stream = ai.subscribe("m", "task-1", FAST_POLL)
    async for event in stream:
        assert event.type == GenerationEventType.Progress
        break
    await stream.aclose()
    polls = platform.polls
    for _ in range(20):
        await asyncio.sleep(0)
    assert platform.polls == polls


async def test_failed_task_without_result_is_a_timeout(ai: Any, platform: Platform) -> None:
    platform.results["task-1"] = {"status": "FAILED", "result": None}
    with pytest.raises(ApiError) as exc:
        await ai.generate("m", {"prompt": "x"}, FAST)
    assert (exc.value.status, exc.value.code, exc.value.message) == (
        408,
        "timeout",
        "Timed out waiting for workflow v1/models:task-1",
    )


async def test_abort_before_submit(ai: Any, platform: Platform) -> None:
    controller = AbortController()
    controller.abort()
    with pytest.raises(ApiError) as exc:
        await ai.generate("m", {"prompt": "x"}, GenerateOptions(signal=controller.signal))
    assert (exc.value.status, exc.value.code, exc.value.message) == (499, "aborted", "Operation aborted")
    assert platform.submits == []


async def test_get_credits_and_options(ai: Any, platform: Platform) -> None:
    assert await ai.get_credits("backend-only-model", {"prompt": "x"}) == 5
    assert platform.options_calls[-1] == {"model": "backend-only-model", "params": {"prompt": "x"}}
    assert await ai.options("backend-only-model", {"prompt": "x"}) == {"credits": 5, "toolId": "t"}


async def test_downscale_and_explicit_drive(platform: Platform) -> None:
    ai = create_client(
        api_url=API, api_key="k", inputs_transformation=InputsTransformation(downscale_oversized_images=True)
    )
    await ai.generate("m", {"prompt": "p"}, FAST)
    assert platform.submits[-1]["options"] == {"inputs_transformation": {"downscale_oversized_images": True}}
    await ai.generate(
        "m",
        {"prompt": "p"},
        GenerateOptions(
            interval_ms=1,
            inputs_transformation=InputsTransformation(downscale_oversized_images=False),
            drive={"name": "keep-me.png", "attributes": {"model": "spoof", "tag": "x"}, "folder": {"id": "f1"}},
        ),
    )
    drive = platform.submits[-1]["options"]["drive"]
    assert platform.submits[-1]["options"]["inputs_transformation"] == {"downscale_oversized_images": False}
    assert (drive["name"], drive["folder"], drive["attributes"]["tag"], drive["attributes"]["model"]) == (
        "keep-me.png",
        {"id": "f1"},
        "x",
        "m",
    )  # SDK provenance wins over custom attributes
    await ai.aclose()


def test_config_errors() -> None:
    with pytest.raises(Exception, match=r"createClient config requires `apiUrl` \(or a custom `transport`\)\."):
        create_client()
    with pytest.raises(Exception, match=r"requires either `fetch` or `apiKey`"):
        create_client(api_url=API)
    with pytest.raises(Exception, match=r"`drive` requires `apiUrl` plus `fetch` or `apiKey`"):
        create_client(transport=ExecuteOnly(), drive=DriveConfig(folder="x"))


class ExecuteOnly:
    def __init__(self) -> None:
        self.requests: list[WorkflowSubmitRequest] = []

    async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
        self.requests.append(request)
        return TransportResult(result={"url": "https://cdn.test/sync.png"})


async def test_execute_only_transport() -> None:
    transport = ExecuteOnly()
    ai = create_client(transport=transport)
    res = await ai.generate("m", {"prompt": "p"})
    assert res.url == "https://cdn.test/sync.png" and res.generation_id is None
    assert transport.requests[0].workflow == "v1/models"
    with pytest.raises(ApiError) as exc:
        await ai.submit("m", {"prompt": "p"})
    assert (exc.value.status, exc.value.code, exc.value.message) == (
        400,
        "unsupported_transport",
        "Transport does not support submit (execute-only transport)",
    )
    assert await ai.get_credits("m", {"prompt": "p"}) is None


async def test_typed_model_inputs_are_accepted(platform: Platform) -> None:
    # Statically checked by mypy (the generated TypedDicts are Mappings, not dict[str, Any]); the call itself
    # passes the caller's params through unchanged.
    client: AiClient = create_client(api_url=API, api_key="key")
    params: Flux2ProInput = {"prompt": "a cat", "aspectRatio": "16:9"}
    typed: Mapping[str, Any] = params
    try:
        result = await client.generate("flux-2-pro", params, FAST)
        assert result.url == "https://cdn.test/out.jpg"
        assert platform.submits[-1]["params"] == {"prompt": "a cat", "aspectRatio": "16:9"}
        await client.run("flux-2-pro", params, FAST)
        with pytest.raises(ApiError, match="no text"):  # an image result; the call itself type-checks
            await client.generate_text("flux-2-pro", params, FAST)
        await client.submit("flux-2-pro", typed, FAST)
        assert await client.get_credits("flux-2-pro", params) == 5
        await client.options("flux-2-pro", params)
    finally:
        await client.aclose()

"""``ai.apis`` (``src/client/apis.ts``) and the small core modules of step 3."""

import asyncio
import json

import httpx
import pytest
import respx

from picsart_ai_sdk import (
    AbortController,
    AbortError,
    ApiError,
    ApiRunMode,
    CatalogsConfig,
    TransportResult,
    WorkflowSubmitRequest,
    create_client,
)
from picsart_ai_sdk.client import workflows
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, install_catalog
from picsart_ai_sdk.core.catalogs import clear_hydrated_catalogs
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from picsart_ai_sdk.core.helpers import badge_priority, resolve_image_size
from picsart_ai_sdk.core.providers import get_provider_color, get_provider_label
from picsart_ai_sdk.core.voices import DEFAULT_VOICE_ID, SEEDAUDIO_DEFAULT_VOICE_ID
from tests.fixtures.catalog import make_model

API = "https://api.test"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(workflows, "_sleep", fake_sleep)


async def test_run_async_polls_and_returns_the_workflow_response() -> None:
    accepted: list[str] = []
    with respx.mock() as router:
        submit = router.post(f"{API}/workflows/openai/v1/chat/submit").mock(
            return_value=httpx.Response(200, json={"status": "success", "response": {"id": "t-9"}})
        )
        router.get(f"{API}/workflows/openai/v1/chat/t-9/result").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "success",
                    "response": {"id": "t-9", "status": "COMPLETED", "result": {"text": "hi"}, "usage": {"credits": 2}},
                },
            )
        )
        ai = create_client(api_url=API, api_key="k")
        res = await ai.apis.run("openai/v1/chat", {"prompt": "x"}, on_accepted=accepted.append, polling_interval=1)
        await ai.aclose()
    assert json.loads(submit.calls[0].request.content)["params"] == {"prompt": "x"}
    assert (res["result"], res["usage"], accepted) == ({"text": "hi"}, {"credits": 2}, ["t-9"])


async def test_run_sync_mode() -> None:
    with respx.mock() as router:
        router.post(f"{API}/workflows/x/v1/y/execute").mock(
            return_value=httpx.Response(200, json={"status": "success", "response": {"result": {"ok": True}}})
        )
        ai = create_client(api_url=API, api_key="k")
        res = await ai.apis.run("x/v1/y", {}, mode=ApiRunMode.SYNC)
        await ai.aclose()
    assert res["result"] == {"ok": True}


async def test_failures_arrive_as_api_error() -> None:
    with respx.mock() as router:
        router.post(f"{API}/workflows/x/v1/y/submit").mock(
            return_value=httpx.Response(403, json={"status": "error", "reason": "forbidden", "message": "nope"})
        )
        ai = create_client(api_url=API, api_key="k")
        with pytest.raises(ApiError) as exc:
            await ai.apis.run("x/v1/y", {})
        await ai.aclose()
    assert (exc.value.status, exc.value.code, str(exc.value)) == (403, "forbidden", "nope")


async def test_abort_passes_through() -> None:
    controller = AbortController()
    controller.abort()
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{API}/workflows/x/v1/y/submit").mock(
            return_value=httpx.Response(200, json={"status": "success", "response": {"id": "t"}})
        )
        router.get(f"{API}/workflows/x/v1/y/t/result").mock(
            return_value=httpx.Response(200, json={"status": "success", "response": {"status": "IN_PROGRESS"}})
        )
        ai = create_client(api_url=API, api_key="k")
        with pytest.raises((AbortError, ApiError)) as exc:
            await ai.apis.run("x/v1/y", {}, signal=controller.signal)
        await ai.aclose()
    assert not isinstance(exc.value, ApiError) or exc.value.code == "aborted"


async def test_custom_transport_has_no_apis() -> None:
    class T:
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            return TransportResult(result={})

    ai = create_client(transport=T())
    with pytest.raises(ApiError) as exc:
        await ai.apis.run("x/v1/y", {})
    assert (exc.value.status, exc.value.code) == (400, "unsupported_transport")


async def test_client_catalogs_preload_from_config() -> None:
    calls: list[WorkflowSubmitRequest] = []

    class T:
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            calls.append(request)
            return TransportResult(result={"items": [{"id": "v1", "name": "V", "tags": []}], "version": "1"})

    install_catalog(
        [
            make_model(
                id="tts",
                name="TTS",
                mode="audio",
                param_config={
                    "voiceId": ParamEntry(
                        {"kind": "catalog", "source": {"workflow": "el/v1/catalog/voices"}, "default": ""}
                    )
                },
            )
        ]
    )
    try:
        ai = create_client(transport=T(), catalogs=CatalogsConfig(preload=True))
        await asyncio.sleep(0.01)
        assert [c.workflow for c in calls] == ["el/v1/catalog/voices"]
        page = await ai.catalogs.voices("tts")
        assert [i["id"] for i in page.items] == ["v1"] and len(calls) == 1
    finally:
        clear_hydrated_catalogs()
        _reset_active_catalog_for_tests()


def test_core_helpers() -> None:
    assert resolve_image_size("16:9", "1024x1024", {"16:9": "1344x768"}) == "1344x768"
    assert resolve_image_size("4:3", "1024x1024", {"16:9": "1344x768"}) == "1024x1024"
    assert resolve_image_size(None, None, {}) is None
    assert [
        badge_priority(make_model(id="m", name="M", mode="image", badge=b))
        for b in (["hot", "new"], ["new", "popular"], ["new"], ["popular"], None)
    ] == [0, 1, 2, 3, 4]
    assert (get_provider_color("kling"), get_provider_label("kling")) == ("#8B5CF6", "K")
    assert (get_provider_color("nope"), get_provider_label("nope")) == ("#666", "?")
    assert (DEFAULT_VOICE_ID, SEEDAUDIO_DEFAULT_VOICE_ID) == ("JBFqnCBsd6RMkjVDRZzb", "en_male_tim_uranus_bigtts")

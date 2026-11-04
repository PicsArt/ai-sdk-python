"""Port of ``__tests__/unit/catalogs.test.ts``."""

import asyncio
from typing import Any

import pytest

from picsart_ai_sdk import AbortController, AbortError, ApiError
from picsart_ai_sdk.client import catalogs as catalogs_module
from picsart_ai_sdk.client.catalogs import create_catalogs
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, install_catalog
from picsart_ai_sdk.core.catalogs import (
    CatalogItem,
    clear_hydrated_catalogs,
    get_hydrated_catalog,
    get_voice_by_id,
    install_hydrated_catalog,
)
from picsart_ai_sdk.core.descriptors.model_accessor import Model
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from picsart_ai_sdk.core.workflow import TransportResult, WorkflowSubmitRequest
from tests.fixtures.catalog import FLUX, make_model

PROMPT = ParamEntry({"kind": "text"}, label="Prompt")


def cat_param(label: str, workflow: str, default: str = "", model_id: str | None = None) -> ParamEntry:
    source = {"workflow": workflow, "modelId": model_id} if model_id else {"workflow": workflow}
    return ParamEntry({"kind": "catalog", "source": source, "default": default}, label=label, catalog_options=[])


MODELS = [
    make_model(
        id="heygen-video-avatar",
        name="HeyGen Video Avatar",
        mode="video",
        provider="heygen",
        param_config={
            "prompt": PROMPT,
            "voiceId": cat_param("Voice", "heygen/v1/catalog/voices"),
            "videoId": cat_param("Avatar", "heygen/v1/catalog/avatars"),
        },
    ),
    make_model(
        id="heygen-talking-photo",
        name="HeyGen Talking Photo",
        mode="video",
        param_config={"prompt": PROMPT, "voiceId": cat_param("Voice", "heygen/v1/catalog/voices")},
    ),
    make_model(
        id="seed-audio-1.0",
        name="Seed Audio",
        mode="audio",
        param_config={
            "prompt": PROMPT,
            "voiceId": cat_param("Voice", "bytedance/v1/catalog/voices", "", "seed-audio-1.0"),
        },
    ),
    make_model(
        id="seed-audio-1.0-multilingual",
        name="Seed Audio ML",
        mode="audio",
        param_config={
            "prompt": PROMPT,
            "voiceId": cat_param("Voice", "bytedance/v1/catalog/voices", "", "seed-audio-1.0-multilingual"),
        },
    ),
    make_model(
        id="kling-video-effects",
        name="Kling Video Effects",
        mode="video",
        param_config={
            "prompt": PROMPT,
            "templateId": cat_param("Template", "kling/v1/catalog/templates", "korean_baseball"),
        },
    ),
    make_model(
        id="picsart-flow",
        name="Picsart Flow",
        mode="image",
        param_config={
            "prompt": PROMPT,
            "templateId": cat_param("Template", "picsart-flow/v1/catalog/templates", "", "picsart-flow"),
        },
    ),
    make_model(
        id="picsart-flow-video",
        name="Picsart Flow Video",
        mode="video",
        param_config={
            "prompt": PROMPT,
            "templateId": cat_param("Template", "picsart-flow/v1/catalog/templates", "", "picsart-flow-video"),
        },
    ),
    FLUX,
]

VOICE_ITEMS: list[CatalogItem] = [
    {
        "id": f"hg-voice-{i}",
        "name": f"Voice {i}",
        "tags": ["female", "english"],
        "preview": {"audioUrl": f"https://cdn/voice-{i}.mp3"},
        "meta": {"language": "English"},
    }
    for i in range(150)
]
AVATAR_ITEMS: list[CatalogItem] = [
    {
        "id": f"hg-avatar-{i}",
        "name": f"Avatar {i}",
        "tags": ["male"],
        "preview": {"imageUrl": f"https://cdn/avatar-{i}.webp", "videoUrl": f"https://cdn/avatar-{i}.mp4"},
        "meta": {"gender": "male", "defaultVoiceId": "hg-voice-1"},
    }
    for i in range(30)
]


def items_for(workflow: str) -> list[CatalogItem]:
    if workflow == "heygen/v1/catalog/voices":
        return VOICE_ITEMS
    if workflow == "heygen/v1/catalog/avatars":
        return AVATAR_ITEMS
    return [{"id": f"{workflow}-item-{i}", "name": f"Item {i}", "tags": []} for i in range(5)]


class FakeTransport:
    """Serves paginated catalog envelopes."""

    def __init__(self) -> None:
        self.calls: list[WorkflowSubmitRequest] = []
        self.fail_next: str | None = None
        self.error_envelope_next = False

    async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
        self.calls.append(request)
        await asyncio.sleep(0)
        if self.fail_next:
            message, self.fail_next = self.fail_next, None
            raise Exception(message)  # noqa: TRY002
        if self.error_envelope_next:
            self.error_envelope_next = False
            return TransportResult(result={"status": "error", "message": "workspace quota exceeded"})
        items = items_for(request.workflow)
        payload = request.payload
        limit = payload.get("limit", 100)
        cursor = payload.get("cursor")
        start = next(i for i, item in enumerate(items) if item["id"] == cursor) + 1 if cursor else 0
        page = items[start : start + limit]
        next_cursor = page[-1]["id"] if start + limit < len(items) else None
        return TransportResult(
            result={"items": page, "version": "2026-08-07", "ttlSeconds": 21600, "nextCursor": next_cursor}
        )


@pytest.fixture(autouse=True)
def catalog() -> Any:
    install_catalog(MODELS)
    clear_hydrated_catalogs()
    yield
    clear_hydrated_catalogs()
    _reset_active_catalog_for_tests()


def options(model: str, key: str) -> list[Any]:
    entry = Model(model).params().catalog(key)
    assert entry is not None
    return entry["catalogOptions"]  # type: ignore[no-any-return]


async def test_pages_cache_accumulate_and_hydrate() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    page1 = await c.voices("heygen-video-avatar")
    assert (len(page1.items), len(t.calls), page1.items[0]["id"], page1.next_cursor) == (
        100,
        1,
        "hg-voice-0",
        "hg-voice-99",
    )
    assert t.calls[0].payload == {"limit": 100}
    page2 = await c.voices("heygen-video-avatar", cursor=page1.next_cursor)
    assert (len(page2.items), page2.next_cursor, len(t.calls)) == (50, None, 2)

    # page cache: repeat calls do not hit the transport
    await c.voices("heygen-video-avatar")
    await c.voices("heygen-video-avatar", cursor="hg-voice-99")
    assert len(t.calls) == 2

    # returned pages are copies
    page1.items.clear()
    assert len((await c.voices("heygen-video-avatar")).items) == 100

    # loaded pages accumulate into the model's options
    entry = Model("heygen-video-avatar").params().catalog("voiceId")
    assert entry is not None and entry["kind"] == "catalog"
    voices = options("heygen-video-avatar", "voiceId")
    assert len(voices) == 150 and voices[0]["previewUrl"] == "https://cdn/voice-0.mp3"

    avatars = await c.avatars("heygen-video-avatar")
    assert (len(avatars.items), avatars.next_cursor) == (30, None)
    avatar = options("heygen-video-avatar", "videoId")[0]
    assert (avatar["previewImageUrl"], avatar["defaultVoiceId"], avatar["gender"], avatar["provider"]) == (
        "https://cdn/avatar-0.webp",
        "hg-voice-1",
        "male",
        "heygen",
    )

    # catalog params are not enums
    assert Model("heygen-video-avatar").params().enum("videoId") is None
    assert Model("heygen-video-avatar").params().enum("voiceId") is None
    # models sharing a source hydrate together
    assert len(options("heygen-talking-photo", "voiceId")) == 150
    schema = Model("heygen-video-avatar").params().to_schema()
    assert schema["videoId"]["type"] == "string" and "enum" not in schema["videoId"]

    voice = get_voice_by_id("hg-voice-7")
    assert voice is not None and voice["name"] == "Voice 7"
    assert get_voice_by_id("not-loaded-anywhere") is None

    # force_refresh drops the cached pages and refetches the first page only
    t.calls.clear()
    assert len((await c.voices("heygen-video-avatar", force_refresh=True)).items) == 100
    assert len(t.calls) == 1

    # validation: catalog params are free strings
    result = Model("heygen-video-avatar").validate(
        {
            "prompt": "A long enough script for the avatar to speak.",
            "videoId": "not-in-the-loaded-list",
            "voiceId": "also-not-in-the-list",
        }
    )
    assert result.valid, result.errors

    # transfer_values: catalog ids reset to the target default
    moved = Model("heygen-video-avatar").params().transfer_values({"voiceId": "hg-voice-3", "videoId": "x"})
    assert (moved["voiceId"], moved["videoId"]) == ("", "")


async def test_concurrent_requests_for_one_page_collapse() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    a, b = await asyncio.gather(c.avatars("heygen-video-avatar"), c.avatars("heygen-video-avatar"))
    assert (len(a.items), len(b.items), len(t.calls)) == (30, 30, 1)


async def test_models_without_a_bound_catalog_reject() -> None:
    c = create_catalogs(FakeTransport())
    with pytest.raises(Exception, match='has no runtime catalog on param "voiceId"'):
        await c.voices("flux-2-pro")
    with pytest.raises(Exception, match='has no runtime catalog on param "templateId"'):
        await c.templates("heygen-video-avatar")
    with pytest.raises(ApiError, match="Unknown model"):
        await c.voices("nope")


async def test_error_paths() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    t.error_envelope_next = True
    with pytest.raises(ApiError, match="workspace quota exceeded"):
        await c.voices("heygen-video-avatar")
    # a transport failure rejects and cleans the in-flight slot — retry works
    t.fail_next = "gateway timeout"
    with pytest.raises(ApiError) as exc:
        await c.avatars("heygen-video-avatar")
    assert (str(exc.value), exc.value.status, exc.value.code) == (
        "heygen/v1/catalog/avatars failed: gateway timeout",
        502,
        "bad_gateway",
    )
    t.calls.clear()
    assert len((await c.avatars("heygen-video-avatar")).items) == 30 and len(t.calls) == 1


async def test_invalid_results() -> None:
    class Shaped:
        def __init__(self, result: Any, raw: Any = None) -> None:
            self.result, self.raw = result, raw

        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            return TransportResult(result=self.result, raw=self.raw)

    with pytest.raises(ApiError) as exc:
        await create_catalogs(Shaped({"items": "x"})).voices("heygen-video-avatar")
    assert (str(exc.value), exc.value.code) == (
        "heygen/v1/catalog/voices returned no catalog result",
        "invalid_response",
    )
    # an envelope the transport kept is unwrapped: raw.response.result
    env = {"response": {"result": {"items": [VOICE_ITEMS[0]], "version": "v", "ttlSeconds": 60}}}
    page = await create_catalogs(Shaped(None, env)).voices("heygen-video-avatar")
    assert (len(page.items), page.next_cursor) == (1, None)


async def test_one_callers_abort_never_rejects_the_others() -> None:
    c = create_catalogs(FakeTransport())
    controller = AbortController()
    first = asyncio.ensure_future(c.voices("heygen-video-avatar", signal=controller.signal))
    second = asyncio.ensure_future(c.voices("heygen-video-avatar"))
    controller.abort()  # synchronously, before the shared fetch settles (as the TS test does)
    with pytest.raises(AbortError):
        await first
    assert len((await second).items) == 100


async def test_cancelling_one_caller_keeps_the_shared_fetch() -> None:
    c = create_catalogs(FakeTransport())
    first = asyncio.ensure_future(c.voices("heygen-video-avatar"))
    second = asyncio.ensure_future(c.voices("heygen-video-avatar"))
    await asyncio.sleep(0)
    first.cancel()
    assert len((await second).items) == 100 and first.cancelled()


async def test_an_already_aborted_signal_rejects_even_a_cached_page() -> None:
    c = create_catalogs(FakeTransport())
    await c.voices("heygen-video-avatar")
    controller = AbortController()
    controller.abort("stop")
    with pytest.raises(AbortError, match="stop"):
        await c.voices("heygen-video-avatar", signal=controller.signal)


async def test_force_refresh_during_an_in_flight_fetch_drops_the_stale_write() -> None:
    futures: list[asyncio.Future[TransportResult]] = []

    class Deferred:
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            future: asyncio.Future[TransportResult] = asyncio.get_running_loop().create_future()
            futures.append(future)
            return await future

    def page(version: str, items: list[CatalogItem]) -> TransportResult:
        return TransportResult(result={"items": items, "version": version, "ttlSeconds": 21600, "nextCursor": None})

    c = create_catalogs(Deferred())
    slow = asyncio.ensure_future(c.avatars("heygen-video-avatar"))
    while len(futures) < 1:  # noqa: ASYNC110 — a handful of ticks until the fetch starts
        await asyncio.sleep(0)
    fast = asyncio.ensure_future(c.avatars("heygen-video-avatar", force_refresh=True))
    while len(futures) < 2:  # noqa: ASYNC110
        await asyncio.sleep(0)
    futures[1].set_result(page("new", AVATAR_ITEMS[:5]))
    futures[0].set_result(page("old", AVATAR_ITEMS[:2]))
    slow_page, fast_page = await asyncio.gather(slow, fast)
    assert (len(fast_page.items), len(slow_page.items)) == (5, 2)
    assert len(options("heygen-video-avatar", "videoId")) == 5


async def test_model_id_sources_are_separate() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    await c.voices("seed-audio-1.0")
    assert t.calls[0].workflow == "bytedance/v1/catalog/voices"
    assert t.calls[0].payload == {"modelId": "seed-audio-1.0", "limit": 100}
    await c.voices("seed-audio-1.0-multilingual")
    assert len(t.calls) == 2


async def test_ttl_floor_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class ZeroTtl:
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            nonlocal calls
            calls += 1
            return TransportResult(result={"items": AVATAR_ITEMS[:3], "version": "v", "ttlSeconds": 0})

    now = [1_000_000.0]
    monkeypatch.setattr(catalogs_module, "_now_ms", lambda: now[0])
    c = create_catalogs(ZeroTtl())
    await c.avatars("heygen-video-avatar")
    await c.avatars("heygen-video-avatar")
    assert calls == 1  # ttlSeconds 0 still caches for the 60 s floor
    now[0] += 59_000
    await c.avatars("heygen-video-avatar")
    assert calls == 1
    now[0] += 1_000
    await c.avatars("heygen-video-avatar")
    assert calls == 2  # expired → refetch

    t = FakeTransport()
    c2 = create_catalogs(t)
    await c2.voices("heygen-video-avatar")
    now[0] += 22_000 * 1000  # past the 21600 s ttl
    t.calls.clear()
    await c2.voices("heygen-video-avatar")
    assert len(t.calls) == 1


async def test_preload_first_page_of_every_bound_catalog() -> None:
    t = FakeTransport()
    create_catalogs(t, preload=True)
    await asyncio.sleep(0.01)
    requested = [f"{c.workflow} {c.payload.get('modelId', '')}" for c in t.calls]
    assert len(set(requested)) == len(requested)
    assert "heygen/v1/catalog/voices " in requested
    assert "bytedance/v1/catalog/voices seed-audio-1.0" in requested
    assert len(options("heygen-video-avatar", "voiceId")) == 100


def test_preload_without_a_running_loop_is_a_no_op() -> None:
    t = FakeTransport()
    create_catalogs(t, preload=True)
    assert t.calls == []


async def test_templates() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    entry = Model("kling-video-effects").params().catalog("templateId")
    assert entry is not None and (entry["kind"], entry["default"], entry["source"]["workflow"]) == (
        "catalog",
        "korean_baseball",
        "kling/v1/catalog/templates",
    )
    page = await c.templates("kling-video-effects")
    assert t.calls[0].workflow == "kling/v1/catalog/templates" and len(page.items) == 5
    t.calls.clear()
    await c.templates("picsart-flow")
    await c.templates("picsart-flow-video")
    assert [x.payload.get("modelId") for x in t.calls] == ["picsart-flow", "picsart-flow-video"]


def test_raw_items_for_non_voice_avatar_keys() -> None:
    source = {"workflow": "kling/v1/catalog/templates"}
    effect_items: list[CatalogItem] = [
        {
            "id": "hug_pro",
            "name": "Hug Pro",
            "tags": ["Dual Image"],
            "preview": {"videoUrl": "https://cdn/hug_pro.mp4"},
            "meta": {"imageSlots": 2},
        }
    ]
    install_hydrated_catalog(source, "templateId", effect_items, "kling", "2026-08-10")  # type: ignore[arg-type]
    hydrated = get_hydrated_catalog(source)  # type: ignore[arg-type]
    assert hydrated is not None and hydrated.param_key == "templateId"
    assert hydrated.catalog_options == effect_items
    assert get_voice_by_id("hug_pro") is None
    clear_hydrated_catalogs()
    assert options("heygen-video-avatar", "videoId") == []


async def test_abort_while_waiting_on_a_shared_fetch() -> None:
    gate = asyncio.Event()

    class Gated(FakeTransport):
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            await gate.wait()
            return await super().execute(request)

    t = Gated()
    c = create_catalogs(t)
    controller = AbortController()
    first = asyncio.ensure_future(c.voices("heygen-video-avatar", signal=controller.signal))
    second = asyncio.ensure_future(c.voices("heygen-video-avatar"))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not first.done() and not second.done()
    controller.abort()
    with pytest.raises(AbortError):
        await first
    gate.set()
    assert len((await second).items) == 100 and len(t.calls) == 1
    # the fetch the aborted caller started still fed the cache
    assert len(options("heygen-video-avatar", "voiceId")) == 100


async def test_a_cancelled_caller_of_a_failed_fetch_logs_nothing() -> None:
    gate = asyncio.Event()
    reported: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, ctx: reported.append(ctx))

    class Failing(FakeTransport):
        async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
            await gate.wait()
            raise Exception("boom")  # noqa: TRY002

    try:
        c = create_catalogs(Failing())
        waiter = asyncio.ensure_future(c.voices("heygen-video-avatar"))
        for _ in range(3):
            await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        import gc

        gc.collect()
        await asyncio.sleep(0)
        assert waiter.cancelled() and reported == []
    finally:
        loop.set_exception_handler(None)


async def test_limit_is_part_of_the_page_cache_and_inflight_keys() -> None:
    t = FakeTransport()
    c = create_catalogs(t)
    assert len((await c.voices("heygen-video-avatar")).items) == 100
    small = await c.voices("heygen-video-avatar", limit=1)
    assert ([i["id"] for i in small.items], small.next_cursor, len(t.calls)) == (["hg-voice-0"], "hg-voice-0", 2)
    # the default limit and an explicit limit=100 share one cache entry
    await c.voices("heygen-video-avatar", limit=100)
    assert len(t.calls) == 2
    # concurrent callers with different limits do not share one request
    a, b = await asyncio.gather(c.avatars("heygen-video-avatar", limit=2), c.avatars("heygen-video-avatar"))
    assert (len(a.items), len(b.items), len(t.calls)) == (2, 30, 4)


async def test_returned_items_are_deep_copies() -> None:
    c = create_catalogs(FakeTransport())
    page = await c.templates("kling-video-effects")
    page.items[0]["name"] = "mutated"
    page.items[0]["tags"].append("mutated")
    again = await c.templates("kling-video-effects")
    assert (again.items[0]["name"], again.items[0]["tags"]) == ("Item 0", [])
    first = options("kling-video-effects", "templateId")[0]
    assert first["name"] == "Item 0" and "mutated" not in (first.get("tags") or [])

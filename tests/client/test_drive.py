"""``ai.drive`` — port of ``drive-generation-save.test.ts`` (Drive parts) plus the client's request shapes."""

import asyncio
import json
import re
from collections.abc import Callable
from types import MappingProxyType
from typing import Any

import httpx
import pytest

from picsart_ai_sdk import DriveConfig, DriveFolder, SaveParams, create_client, parse_generation
from picsart_ai_sdk.client.drive import DriveClient, create_drive_client
from picsart_ai_sdk.client.drive_naming import AppIdentity, GenerationProvenance
from picsart_ai_sdk.client.fetch import FetchInit

API = "https://api.test"
FOLDER = DriveFolder(uid="folder-1", name="AI")
SAVE = SaveParams(url="https://cdn.test/video.mp4", name="video.mp4", resource_type="VIDEO")

Handler = Callable[[str, str, Any], httpx.Response]


class FakeDrive:
    """Records requests; ``handler(method, path_and_query, body)`` answers them."""

    def __init__(self, handler: Handler | None = None) -> None:
        self.requests: list[tuple[str, str, Any, dict[str, str]]] = []
        self.handler = handler or (lambda m, p, b: httpx.Response(200, json={"response": {"uid": "saved-1"}}))

    async def fetch(self, url: str, init: FetchInit) -> httpx.Response:
        await asyncio.sleep(0)
        path = url.removeprefix(API)
        body = json.loads(init["body"]) if init.get("body") else None
        method = init.get("method", "GET")
        self.requests.append((method, path, body, dict(init.get("headers") or {})))
        return self.handler(method, path, body)

    def client(self, root: str = "AI") -> DriveClient:
        return create_drive_client(self.fetch, API, root)

    def attributes(self) -> dict[str, str]:
        body = next(b for m, p, b, _ in reversed(self.requests) if p == "/cloud-storage/v1/me/files" and m == "POST")
        return {a["property"]: a["value"] for a in body["attributes"]}


# ── drive-generation-save.test.ts ──────────────────────────────────────
async def test_explicit_generation_save_persists_all_inputs_and_protects_them() -> None:
    fake = FakeDrive()
    params = {
        "prompt": "a cat",
        "duration": 5,
        "generateAudio": False,
        "seed": 0,
        "imageUrls": ["https://cdn.test/r.png"],
    }
    await fake.client().save(
        SaveParams(
            url=SAVE.url,
            name=SAVE.name,
            resource_type="VIDEO",
            generation=GenerationProvenance(model_id="video-model", params=params),
            attributes={"model": "wrong", "aiSDKPayload": "{}", "custom": "kept"},
        ),
        FOLDER,
    )
    attrs = fake.attributes()
    assert (attrs["model"], attrs["custom"]) == ("video-model", "kept")
    assert json.loads(attrs["aiSDKPayload"]) == params


async def test_legacy_saves_persist_canonical_metadata_generic_uploads_do_not() -> None:
    fake = FakeDrive()
    drive = fake.client()
    await drive.save(
        SaveParams(
            url=SAVE.url,
            name=SAVE.name,
            resource_type="VIDEO",
            attributes={
                "model": "legacy-model",
                "prompt": "hello",
                "duration": "5",
                "textScript": json.dumps({"resolution": "720p", "referenceImageUrls": ["https://cdn.test/r.png"]}),
            },
        ),
        FOLDER,
    )
    assert json.loads(fake.attributes()["aiSDKPayload"]) == {
        "prompt": "hello",
        "resolution": "720p",
        "duration": 5,
        "imageUrls": ["https://cdn.test/r.png"],
    }
    await drive.save(SAVE, FOLDER)
    assert fake.attributes() == {}


def test_build_save_params_takes_full_params_or_a_prompt() -> None:
    drive = FakeDrive().client()
    full = drive.build_save_params(SAVE.url, "model", "Model", "video", {"prompt": "hello", "duration": 5, "seed": 0})
    assert json.loads(full.attributes["aiSDKPayload"]) == {"prompt": "hello", "duration": 5, "seed": 0}  # type: ignore[index]
    legacy = drive.build_save_params(SAVE.url, "model", "Model", "video", "hello")
    assert legacy.attributes == {
        "tool": "ai-sdk",
        "model": "model",
        "aiSDKPayload": '{"prompt":"hello"}',
        "prompt": "hello",
        "service": "Model",
    }
    # drive-filename.test.ts: named after the prompt, extension from the result URL
    saved = drive.build_save_params(
        "https://cdn.test/b0f5897a-1c2d.jpg",
        "picsart-sana-sprint-v1",
        "Picsart SANA-Sprint",
        "image",
        "A red paper boat on a calm blue lake at sunrise",
    )
    assert re.fullmatch(r"a-red-paper-boat-on-a-calm-blue-lake-at-sunrise-\d{6}\.jpg", saved.name)
    assert saved.resource_type == "PHOTO"


def test_build_save_params_cuts_the_prompt_attribute_to_the_drive_limit() -> None:
    drive = FakeDrive().client()
    params = drive.build_save_params(SAVE.url, "model", "Model", "image", "a" * 25_000)
    attributes = params.attributes or {}
    assert len(attributes["prompt"]) == 18_000
    assert all(len(value) <= 20_000 for value in attributes.values())
    assert params.name.startswith("a" * 50 + "-")  # the filename still slugs the full prompt


async def test_legacy_normalization_keeps_app_identity_on_read_back() -> None:
    fake = FakeDrive()
    drive = fake.client()
    await drive.save(
        SaveParams(
            url=SAVE.url,
            name=SAVE.name,
            resource_type="VIDEO",
            attributes={"model": "legacy-model", "prompt": "hello", "tool": "ai-playground"},
        ),
        FOLDER,
    )
    generation = parse_generation({"uid": "saved-1", "attributes": fake.attributes()})
    assert (generation.app_id, generation.app_type) == ("com.picsart.ai-playground", "miniapp")
    await drive.save(
        SaveParams(
            url=SAVE.url,
            name=SAVE.name,
            resource_type="VIDEO",
            attributes={"model": "legacy-model", "tool": "ai-playground", "appId": "custom", "appType": "native"},
        ),
        FOLDER,
    )
    assert (fake.attributes()["appId"], fake.attributes()["appType"]) == ("custom", "native")


# ── save request shape ─────────────────────────────────────────────────
async def test_save_body_and_result() -> None:
    fake = FakeDrive()
    result = await fake.client().save(
        SaveParams(url="https://cdn.test/a.png", name="a.png", resource_type="PHOTO", preview_url="https://cdn.test/p"),
        FOLDER,
    )
    assert result is not None and (result.uid, result.folder) == ("saved-1", FOLDER)
    method, path, body, headers = fake.requests[-1]
    assert (method, path, headers) == ("POST", "/cloud-storage/v1/me/files", {"Content-Type": "application/json"})
    assert body == {
        "name": "a.png",
        "sourceUrl": "https://cdn.test/a.png",
        "parentFolderUid": "folder-1",
        "content": {"type": "STANDALONE", "resourceType": "PHOTO", "sourcePlatform": "WEB"},
        "preview": {"url": "https://cdn.test/p", "width": 1024, "height": 1024},
        "attributes": [],
    }


async def test_restricted_keywords_retry_under_a_safe_name() -> None:
    def handler(method: str, path: str, body: Any) -> httpx.Response:
        if body["name"].startswith("ai-generation-"):
            return httpx.Response(200, json={"response": {"uid": "u2"}})
        return httpx.Response(400, text='{"reason":"restricted_keywords"}')

    fake = FakeDrive(handler)
    result = await fake.client().save(SaveParams(url="u", name="bad word.mp4", resource_type="VIDEO"), FOLDER)
    assert result is not None and result.uid == "u2"
    assert re.fullmatch(r"ai-generation-\d{13}\.mp4", fake.requests[-1][2]["name"])
    other = FakeDrive(lambda m, p, b: httpx.Response(400, text="nope"))
    assert await other.client().save(SAVE, FOLDER) is None and len(other.requests) == 1
    no_ext = FakeDrive(handler)
    await no_ext.client().save(SaveParams(url="u", name="noext", resource_type="PHOTO"), FOLDER)
    assert no_ext.requests[-1][2]["name"].endswith(".noext")  # `'noext'.split('.').pop()` is the whole name


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(500, json={}),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json=None),
        httpx.Response(200, json={"response": {}}),
        httpx.Response(200, json={"response": {"uid": ""}}),
    ],
)
async def test_save_failures_are_none(answer: httpx.Response) -> None:
    assert await FakeDrive(lambda m, p, b: answer).client().save(SAVE, FOLDER) is None


async def test_save_swallows_a_network_error() -> None:
    async def broken(url: str, init: FetchInit) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert await create_drive_client(broken, API, "AI").save(SAVE, FOLDER) is None


# ── root folder resolution ─────────────────────────────────────────────
def folder_api(by_path: Any = None, listing: list[Any] | None = None, created: str | None = "new-root") -> Handler:
    def handler(method: str, path: str, body: Any) -> httpx.Response:
        if path.startswith("/cloud-storage/v1/me/files-by-path"):
            return httpx.Response(200, json={"status": "success", "response": by_path} if by_path else {"status": "x"})
        if path.startswith("/cloud-storage/v1/me/files?") and "fileTypes=FOLDER" in path:
            return httpx.Response(200, json={"response": listing or []})
        if path == "/cloud-storage/v1/me/folders":
            return httpx.Response(200, json={"response": {"uid": created}}) if created else httpx.Response(500)
        return httpx.Response(200, json={"response": {"uid": "saved-1"}})

    return handler


async def test_root_folder_found_by_path_and_cached() -> None:
    fake = FakeDrive(folder_api(by_path=[{"uid": "root-1"}]))
    drive = fake.client("AI Playground")
    assert await drive.ensure_folder() == DriveFolder(name="AI Playground", uid="root-1")
    assert await drive.ensure_folder() == DriveFolder(name="AI Playground", uid="root-1")
    assert [p for _, p, _, _ in fake.requests] == ["/cloud-storage/v1/me/files-by-path?path=AI%20Playground"]


async def test_root_folder_resolution_order_and_creation() -> None:
    fake = FakeDrive(folder_api(listing=[{"name": "other", "uid": "x"}]))
    assert await fake.client("AI").ensure_folder() == DriveFolder(name="AI", uid="new-root")
    assert [(m, p, b) for m, p, b, _ in fake.requests] == [
        ("GET", "/cloud-storage/v1/me/files-by-path?path=AI", None),
        ("GET", "/cloud-storage/v1/me/files?fileTypes=FOLDER&limit=100", None),
        ("GET", "/cloud-storage/v1/me/files-by-path?path=AI", None),
        ("POST", "/cloud-storage/v1/me/folders", {"name": "AI"}),
    ]
    found = FakeDrive(folder_api(listing=[{"name": "ai", "uid": "listed"}]))
    assert (await found.client("AI").ensure_folder()) == DriveFolder(name="AI", uid="listed")  # case-insensitive


async def test_concurrent_callers_share_one_resolve_and_a_failure_is_retried() -> None:
    fake = FakeDrive(folder_api(by_path={"uid": "root-1"}))
    drive = fake.client()
    a, b = await asyncio.gather(drive.ensure_folder(), drive.ensure_folder())
    assert a == b == DriveFolder(name="AI", uid="root-1") and len(fake.requests) == 1

    failing = FakeDrive(folder_api(created=None))
    drive2 = failing.client()
    assert await drive2.ensure_folder() is None
    assert await drive2.save(SAVE) is None
    assert sum(1 for _, p, _, _ in failing.requests if p == "/cloud-storage/v1/me/folders") == 2  # retried


async def test_subfolders_found_or_created_in_the_root() -> None:
    fake = FakeDrive(folder_api(by_path={"uid": "root-1"}, listing=[{"name": "Board", "uid": "b-1"}]))
    drive = fake.client()
    assert await drive.ensure_folder("board") == DriveFolder(name="board", uid="b-1")
    assert fake.requests[-1][1] == "/cloud-storage/v1/me/files?parentFolderUid=root-1&fileTypes=FOLDER&limit=100"
    assert await drive.ensure_folder("New") == DriveFolder(name="New", uid="new-root")
    assert fake.requests[-1][1:3] == ("/cloud-storage/v1/me/folders", {"name": "New", "parentFolderUid": "root-1"})


async def test_folders_all_folders_find_folder() -> None:
    def handler(method: str, path: str, body: Any) -> httpx.Response:
        if path.startswith("/cloud-storage/v1/me/files-by-path"):
            return httpx.Response(200, json={"status": "success", "response": {"uid": "root-1"}})
        if "parentFolderUid=root-1" in path:
            return httpx.Response(
                200, json={"response": [{"name": "Board", "uid": "b-1"}, {"name": "", "uid": "skip"}, {"uid": "x"}]}
            )
        return httpx.Response(
            200, json={"response": [{"name": "AI", "uid": "root-1"}, {"name": "Board", "uid": "b-1"}]}
        )

    drive = FakeDrive(handler).client()
    assert await drive.folders() == [DriveFolder("Board", "b-1")]
    assert await drive.all_folders() == [DriveFolder("AI", "root-1"), DriveFolder("Board", "b-1")]
    assert await drive.find_folder("board") == DriveFolder("Board", "b-1")
    assert await drive.find_folder("ai") == DriveFolder("AI", "root-1")
    assert await drive.find_folder("missing") is None


# ── listing and reading ────────────────────────────────────────────────
FILES = [
    {
        "uid": "f1",
        "name": "clip.MP4",
        "sourceUrl": " https://cdn.test/clip.mp4 ",
        "preview": {"url": "https://cdn.test/p.jpg"},
        "updatedAt": 1700000000000,
        "createdAt": "2026-01-01T00:00:00Z",
        "attributes": [
            {"property": "model", "value": "kling"},
            {"property": "prompt", "value": ""},
            {"property": "userReaction", "value": "like"},
            {"property": "seedance~draftTask", "value": '{"id":"d-1","extra":1}'},
            {"property": "seedance~draftExpiresAt", "value": "2026-02-01T00:00:00Z"},
            {"property": "textScript", "value": '{"aspectRatio":"16:9","referenceImageUrls":["https://r"]}'},
        ],
    },
    {"uid": "f2", "name": "__hidden.png", "sourceUrl": "https://cdn.test/h.png"},
    {"uid": "f3", "name": "no-url.png", "sourceUrl": "   "},
    {
        "uid": 4,
        "name": "voice",
        "sourceUrl": "https://cdn.test/v",
        "content": {"resourceType": "audio"},
        "createdAt": "x",
    },
]


async def test_list_and_list_detailed() -> None:
    fake = FakeDrive(lambda m, p, b: httpx.Response(200, json={"response": FILES}))
    drive = fake.client()
    items = await drive.list()
    assert fake.requests[-1][1] == (
        "/cloud-storage/v1/me/flattened-files?limit=100&sortType=UPDATED&sortOrder=DESC&fileTypes=FILE"
        "&contentResourceTypes=PHOTO,VIDEO,AUDIO"
    )
    assert [(i.uid, i.url, i.type, i.preview_url) for i in items] == [
        ("f1", "https://cdn.test/clip.mp4", "video", "https://cdn.test/p.jpg"),
        ("4", "https://cdn.test/v", "audio", None),
    ]
    assert items[0].timestamp == 1700000000000 and items[1].timestamp != items[1].timestamp  # Number("x") is NaN

    await drive.list(folder=FOLDER, type="image")
    assert fake.requests[-1][1] == (
        "/cloud-storage/v1/me/files?parentFolderUid=folder-1&limit=100&sortType=UPDATED&sortOrder=DESC"
        "&fileTypes=FILE&contentResourceTypes=PHOTO"
    )
    detailed = (await drive.list_detailed())[0]
    assert (detailed.model, detailed.prompt, detailed.user_reaction, detailed.created_at) == (
        "kling",
        None,
        "like",
        "2026-01-01T00:00:00Z",
    )
    assert (detailed.aspect_ratio, detailed.reference_image_urls) == ("16:9", ["https://r"])
    assert (detailed.draft_task, detailed.draft_expires_at) == ({"id": "d-1"}, "2026-02-01T00:00:00Z")


async def test_a_listing_error_is_empty_a_null_file_raises_like_ts() -> None:
    assert await FakeDrive(lambda m, p, b: httpx.Response(503)).client().list() == []
    assert await FakeDrive(lambda m, p, b: httpx.Response(200, json={"response": "x"})).client().list() == []
    with pytest.raises(TypeError):  # TS: `file.sourceUrl` on null throws out of list()
        await FakeDrive(lambda m, p, b: httpx.Response(200, json={"response": [None]})).client().list()


async def test_get_generation() -> None:
    attrs = {
        "model": "flux",
        "aiSDKPayload": '{"prompt":"a cat","seed":1}',
        "appId": "com.x",
        "appType": "web",
        "userReaction": "meh",
    }
    fake = FakeDrive(lambda m, p, b: httpx.Response(200, json={"response": {"uid": "g", "attributes": attrs}}))
    generation = await fake.client().get_generation("g")
    assert fake.requests[-1][1] == "/drive/v1/files/g"
    assert generation is not None
    assert (generation.model, generation.ai_sdk_payload, generation.app_id, generation.app_type) == (
        "flux",
        {"prompt": "a cat", "seed": 1},
        "com.x",
        None,
    )
    assert generation.user_reaction is None
    for answer in (httpx.Response(404), httpx.Response(200, json={"response": []})):

        def reply(m: str, p: str, b: Any, a: httpx.Response = answer) -> httpx.Response:
            return a

        assert await FakeDrive(reply).client().get_generation("g") is None


def test_parse_generation_reads_attributes_from_any_mapping() -> None:
    attributes = MappingProxyType({"model": "flux", "aiSDKPayload": '{"prompt":"a cat"}', "appId": "com.app"})
    generation = parse_generation({"attributes": attributes})
    assert (generation.model, generation.app_id, generation.ai_sdk_payload) == ("flux", "com.app", {"prompt": "a cat"})


def test_parse_generation_legacy_adapter() -> None:
    generation = parse_generation(
        {
            "attributes": {
                "model": "old",
                "prompt": "p",
                "duration": " 7 ",
                "textScript": json.dumps(
                    {
                        "aspectRatio": " ",
                        "resolution": "1080p",
                        "referenceVideoUrl": "https://v",
                        "referenceImageUrls": ["a", 1],
                        "style": "anime",
                        "exploreImageId": "e",
                    }
                ),
                "userReaction": "dislike",
            }
        }
    )
    assert generation.model == "old" and generation.user_reaction == "dislike"
    assert generation.ai_sdk_payload == {
        "prompt": "p",
        "resolution": "1080p",
        "duration": 7,
        "videoUrl": "https://v",
        "style": "anime",
        "exploreImageId": "e",
    }
    assert parse_generation({"attributes": {"duration": "abc"}}).ai_sdk_payload["duration"] != 0  # type: ignore[index]
    assert parse_generation({}).ai_sdk_payload == {"prompt": ""}
    assert parse_generation({"attributes": [{"property": "model", "value": None}]}).model == "null"


async def test_reactions() -> None:
    fake = FakeDrive(lambda m, p, b: httpx.Response(200, json={}))
    drive = fake.client()
    assert await drive.add_reaction("f1", "like") is True
    assert fake.requests[-1][:3] == ("PATCH", "/drive/v1/files/f1", {"attributes": {"userReaction": "like"}})
    assert await drive.remove_reaction("f1") is True
    assert fake.requests[-1][2] == {"attributes": {"userReaction": None}}
    assert await FakeDrive(lambda m, p, b: httpx.Response(403)).client().add_reaction("f", "like") is False


# ── client wiring ──────────────────────────────────────────────────────
async def test_client_exposes_drive_only_when_configured() -> None:
    assert create_client(api_url=API, api_key="k").drive is None
    ai = create_client(api_url=API, api_key="k", drive=DriveConfig(folder="AI"))
    assert isinstance(ai.drive, DriveClient)
    await ai.aclose()


def test_app_identity_rides_build_generation_attributes() -> None:
    fake = FakeDrive()
    params = SaveParams(
        url="u",
        name="n",
        resource_type="PHOTO",
        generation=GenerationProvenance(model_id="m", params={"prompt": "x"}, app=AppIdentity(id="a", type="native")),
    )
    from picsart_ai_sdk.client.drive import _attributes_for_save

    assert _attributes_for_save(params) == {
        "model": "m",
        "aiSDKPayload": '{"prompt":"x"}',
        "appId": "a",
        "appType": "native",
    }
    assert fake.requests == []


# ── generation surface (drive-generation-save.test.ts) ─────────────────
class Capturing:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def execute(self, request: Any) -> Any:
        from picsart_ai_sdk import TransportResult

        self.payloads.append(request.payload)
        return TransportResult(result={"url": SAVE.url})

    async def submit(self, request: Any) -> str:
        self.payloads.append(request.payload)
        return "job-1"

    async def poll(self, handle: Any, options: Any) -> Any:
        from picsart_ai_sdk import TransportResult

        return TransportResult(result={"url": SAVE.url})


async def test_generate_and_submit_cannot_lose_provenance_through_custom_attributes() -> None:
    from picsart_ai_sdk import GenerateOptions

    t = Capturing()
    client = create_client(transport=t)
    drive: dict[str, Any] = {
        "name": "asset.png",
        "attributes": {"model": "wrong", "aiSDKPayload": "{}", "custom": "kept"},
    }
    options = GenerateOptions(drive=drive)  # type: ignore[arg-type]
    await client.generate("flux-2-pro", {"prompt": "a cat"}, options)
    await client.submit("flux-2-pro", {"prompt": "a dog"}, options)
    assert len(t.payloads) == 2
    for payload in t.payloads:
        attrs = payload["options"]["drive"]["attributes"]
        assert (attrs["model"], attrs["custom"]) == ("flux-2-pro", "kept")
        assert json.loads(attrs["aiSDKPayload"])["prompt"] in ("a cat", "a dog")
    assert drive["attributes"]["model"] == "wrong"  # caller options are not mutated


async def test_explicit_drive_generation_records_the_given_inputs() -> None:
    from picsart_ai_sdk import GenerateOptions

    t = Capturing()
    client = create_client(transport=t)
    original = {"prompt": "a cat at dusk", "resolution": "1080p", "duration": 5}
    options = GenerateOptions(
        app=AppIdentity(id="com.test.app", type="miniapp"),
        drive={
            "name": "final.mp4",
            "attributes": {"model": "wrong", "aiSDKPayload": "{}", "custom": "kept"},
            "generation": GenerationProvenance(model_id="flux-2-pro", params=original),
        },
    )
    await client.submit("flux-2-pro", {"prompt": "render job-1"}, options)
    drive = t.payloads[0]["options"]["drive"]
    assert json.loads(drive["attributes"]["aiSDKPayload"]) == original
    assert (drive["attributes"]["model"], drive["attributes"]["custom"]) == ("flux-2-pro", "kept")
    assert drive["attributes"]["appId"] == "com.test.app"  # the call-level app identity still applies
    assert "generation" not in drive  # the override is never sent to the workflow


async def test_raw_apis_pass_vendor_payloads_through_unchanged() -> None:
    from picsart_ai_sdk.client.apis import create_apis

    calls: list[Any] = []

    class Wc:
        async def run(self, api: str, payload: Any, **options: Any) -> dict[str, Any]:
            calls.append(payload)
            return {"result": {}}

    apis = create_apis(Wc())  # type: ignore[arg-type]
    payload = {"prompt": "a cat", "image_url": "https://cdn.test/ref.png", "options": {"drive": {"name": "clip.mp4"}}}
    await apis.run("vendor/model/video", payload)
    assert calls[0] is payload and "attributes" not in payload["options"]["drive"]
    no_drive = {"prompt": "hello"}
    await apis.run("vendor/model/video", no_drive)
    assert calls[1] is no_drive


async def test_a_null_root_is_retried_under_an_eager_task_factory() -> None:
    import sys

    if sys.version_info < (3, 12):
        pytest.skip("eager_task_factory is 3.12+")
    loop = asyncio.get_running_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        ok = [False]

        async def fetch(url: str, init: FetchInit) -> httpx.Response:  # answers without suspending
            if ok[0] and "files-by-path" in url:
                return httpx.Response(200, json={"status": "success", "response": {"uid": "R"}})
            return httpx.Response(404)

        drive = create_drive_client(fetch, API, "AI")
        assert await drive.ensure_folder() is None
        ok[0] = True
        assert await drive.ensure_folder() == DriveFolder(name="AI", uid="R")
    finally:
        loop.set_task_factory(None)

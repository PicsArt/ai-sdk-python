"""Port of ``__tests__/unit/catalog-store.test.ts``."""

import asyncio
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from picsart_ai_sdk import create_client
from picsart_ai_sdk.client.fetch import FetchInit
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, active_models
from picsart_ai_sdk.core.catalog_store import (
    _reset_model_catalog_store_for_tests,
    build_query,
    is_model_catalog_loaded,
    load_model_catalog,
    refresh_model_catalog,
    when_model_catalog_loaded,
)
from picsart_ai_sdk.core.catalog_wire import serialize_catalog
from picsart_ai_sdk.core.errors import ApiError
from tests.fixtures.catalog import FIXTURE_CATALOG, make_model

API_URL = "https://x.test"
FIXTURE = serialize_catalog(FIXTURE_CATALOG)
FIXTURE_PLUS = serialize_catalog([*FIXTURE_CATALOG, make_model(id="fresh-model", name="Fresh Model", mode="image")])


@pytest.fixture(autouse=True)
def reset() -> Any:
    _reset_model_catalog_store_for_tests()
    _reset_active_catalog_for_tests()
    yield
    _reset_model_catalog_store_for_tests()
    _reset_active_catalog_for_tests()


class Mock:
    def __init__(self, body: Callable[[], Any], status: int = 200) -> None:
        self.body, self.status, self.urls = body, status, []  # type: ignore[var-annotated]

    async def fetch(self, url: str, init: FetchInit) -> httpx.Response:
        self.urls.append(url)
        return httpx.Response(self.status, json=self.body())


def ids() -> list[str]:
    return sorted(m.id for m in active_models())


async def test_query_string() -> None:
    mock = Mock(lambda: FIXTURE)
    await load_model_catalog(
        mock.fetch,
        API_URL,
        {
            "mode": ["video", "image"],
            "provider": "flux",
            "release": ["production", "general-availability"],
            "includeDeprecated": True,
            "search": "flux and more",
        },
    )
    url = mock.urls[0]
    assert url.startswith(
        "https://x.test/gw-v2/v1/models-catalog?include=schema%2Cconstraints%2Crouting&mode=video%2Cimage"
    )
    assert url.endswith("&search=flux+and+more")
    q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    assert q == {
        "include": "schema,constraints,routing",
        "mode": "video,image",
        "provider": "flux",
        "release": "production,general-availability",
        "includeDeprecated": "true",
        "search": "flux and more",
    }
    assert is_model_catalog_loaded()


def test_js_truthiness_of_filters() -> None:
    assert (
        build_query({"mode": [], "provider": "", "includeDeprecated": False})
        == "include=schema%2Cconstraints%2Crouting&mode="
    )


async def test_success_installs() -> None:
    await load_model_catalog(Mock(lambda: FIXTURE).fetch, API_URL + "///")
    assert ids() == ["claude-opus-5", "flux-2-pro", "kling-v3"]


async def test_concurrent_and_repeat_share_one_fetch() -> None:
    mock = Mock(lambda: FIXTURE)
    # Each caller gets its own shielded view of the one shared load (TS: the same Promise), so the share is
    # asserted by the single fetch, not by identity.
    a, b = load_model_catalog(mock.fetch, API_URL), load_model_catalog(mock.fetch, API_URL)
    await asyncio.gather(a, b)
    await load_model_catalog(mock.fetch, API_URL)
    loaded = when_model_catalog_loaded()
    assert loaded is not None
    await loaded
    assert len(mock.urls) == 1


async def test_conflict() -> None:
    mock = Mock(lambda: FIXTURE)
    await load_model_catalog(mock.fetch, API_URL)
    for args in ((API_URL, {"mode": "video"}), ("https://other.test", None)):
        with pytest.raises(ApiError) as exc:
            await load_model_catalog(mock.fetch, *args)
        assert (exc.value.status, exc.value.code) == (409, "catalog_conflict")
    assert exc.value.message == 'Model catalog already loaded from "https://x.test" — one catalog source per process.'


async def test_failure_is_retryable() -> None:
    calls = {"n": 0}

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                503, json={"status": "error", "reason": "service_unavailable", "message": "catalog down"}
            )
        return httpx.Response(200, json=FIXTURE)

    with pytest.raises(ApiError) as exc:
        await load_model_catalog(fetch, API_URL)
    assert (exc.value.status, exc.value.code, exc.value.message) == (503, "service_unavailable", "catalog down")
    assert not is_model_catalog_loaded() and when_model_catalog_loaded() is None
    await load_model_catalog(fetch, API_URL)
    assert is_model_catalog_loaded() and ids()


async def test_non_json_error_and_malformed_body() -> None:
    async def html(url: str, init: FetchInit) -> httpx.Response:
        return httpx.Response(502, text="<html>")

    with pytest.raises(ApiError) as exc:
        await load_model_catalog(html, API_URL)
    assert (exc.value.status, exc.value.code, exc.value.message) == (
        502,
        "bad_gateway",
        "Model catalog request failed with status 502",
    )
    with pytest.raises(ApiError) as exc:
        await load_model_catalog(Mock(lambda: {"status": "success", "response": {}}).fetch, API_URL)
    assert (exc.value.status, exc.value.code) == (502, "invalid_response")


async def test_refresh_refetches_and_accepts_new_filters() -> None:
    body = {"v": FIXTURE}
    mock = Mock(lambda: body["v"])
    await load_model_catalog(mock.fetch, API_URL)
    body["v"] = FIXTURE_PLUS
    await refresh_model_catalog(mock.fetch, API_URL)
    assert len(mock.urls) == 2 and "fresh-model" in ids() and is_model_catalog_loaded()
    await refresh_model_catalog(mock.fetch, API_URL, {"mode": "video"})
    assert "mode=video" in mock.urls[-1]


async def test_failed_refresh_keeps_previous_catalog() -> None:
    state = {"fail": False}

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        if state["fail"]:
            return httpx.Response(503, json={"reason": "service_unavailable", "message": "catalog down"})
        return httpx.Response(200, json=FIXTURE)

    await load_model_catalog(fetch, API_URL)
    loaded = when_model_catalog_loaded()
    state["fail"] = True
    with pytest.raises(ApiError):
        await refresh_model_catalog(fetch, API_URL)
    assert when_model_catalog_loaded() is loaded and is_model_catalog_loaded() and ids()


async def test_scheduled_refresh_and_default_never_refreshes() -> None:
    mock = Mock(lambda: FIXTURE)
    await load_model_catalog(mock.fetch, API_URL, None, 0.05)
    await asyncio.sleep(0.18)
    assert len(mock.urls) >= 3
    _reset_model_catalog_store_for_tests()
    mock2 = Mock(lambda: FIXTURE)
    await load_model_catalog(mock2.fetch, API_URL)
    await asyncio.sleep(0.1)
    assert len(mock2.urls) == 1


async def test_init_model_catalog_on_the_client() -> None:
    seen: list[str] = []

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        seen.append(url)
        return httpx.Response(200, json=FIXTURE)

    ai = create_client(api_url=API_URL, fetch=fetch)
    await ai.init_model_catalog({"mode": "image"})
    assert "mode=image" in seen[0] and ids()

    class T:
        async def execute(self, request: Any) -> Any: ...

    with pytest.raises(ApiError) as exc:
        await create_client(transport=T()).init_model_catalog()
    assert (exc.value.status, exc.value.code) == (400, "validation_error")
    assert json.dumps(FIXTURE)  # fixture stays JSON-serializable


async def test_trailing_slash_urls_are_one_source() -> None:
    mock = Mock(lambda: FIXTURE)
    await load_model_catalog(mock.fetch, API_URL)
    await load_model_catalog(mock.fetch, f"{API_URL}/")
    await load_model_catalog(mock.fetch, f"{API_URL}//")
    assert len(mock.urls) == 1


async def test_readers_during_a_failed_refresh_keep_the_previous_catalog() -> None:
    gate = asyncio.Event()
    state = {"fail": False}

    async def fetch(url: str, init: FetchInit) -> httpx.Response:
        if state["fail"]:
            await gate.wait()
            return httpx.Response(503, json={"reason": "service_unavailable", "message": "catalog down"})
        return httpx.Response(200, json=FIXTURE)

    await load_model_catalog(fetch, API_URL)
    state["fail"] = True
    refresh = asyncio.ensure_future(refresh_model_catalog(fetch, API_URL))
    await asyncio.sleep(0)
    reader = when_model_catalog_loaded()
    same_source = load_model_catalog(fetch, API_URL)
    assert reader is not None
    gate.set()
    with pytest.raises(ApiError):
        await refresh
    await reader  # the installed catalog is still valid: readers resolve
    await same_source
    assert is_model_catalog_loaded() and ids()

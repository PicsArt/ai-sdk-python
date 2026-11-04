"""The one model-catalog load per process — port of ``src/core/catalog-store.ts``.

``load_model_catalog`` fetches ``/gw-v2/v1/models-catalog`` once and installs it
(repeat or concurrent calls with the same url and filters share the load; a
different source is a 409 ``catalog_conflict``); an optional background task
refreshes it, keeping the previous catalog when a refresh fails.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any, TypedDict

from .active_catalog import install_catalog
from .catalog_wire import deserialize_catalog
from .errors import ApiError, code_for_status
from .js import js_json_dumps, truthy

if TYPE_CHECKING:
    from ..client.fetch import AuthenticatedFetch


class ModelCatalogFilters(TypedDict, total=False):
    mode: str | list[str]
    provider: str | list[str]
    release: str | list[str]
    includeDeprecated: bool
    search: str


CATALOG_PATH = "/gw-v2/v1/models-catalog"

_load: asyncio.Future[None] | None = None
_loaded = False
_key: str | None = None
_refresh_task: asyncio.Task[None] | None = None


def _key_of(api_url: str, filters: ModelCatalogFilters | None) -> str:
    """Normalized as ``_fetch_and_install`` builds the request: ``x`` and ``x/`` are one source."""
    return f"{re.sub(r'/+$', '', api_url)}::{js_json_dumps(filters if filters is not None else {})}"


def _form_encode(value: str) -> str:
    """``application/x-www-form-urlencoded`` as ``URLSearchParams`` writes it (``*-._`` kept, space → ``+``)."""
    out = []
    for byte in value.encode("utf-8"):
        char = chr(byte)
        if re.fullmatch(r"[A-Za-z0-9*\-._]", char):
            out.append(char)
        elif char == " ":
            out.append("+")
        else:
            out.append(f"%{byte:02X}")
    return "".join(out)


def build_query(filters: ModelCatalogFilters | None) -> str:
    f: dict[str, Any] = dict(filters or {})

    def join(v: Any) -> str:
        return ",".join(str(x) for x in v) if isinstance(v, list) else str(v)

    query = [("include", "schema,constraints,routing")]
    # JS truthiness: an empty list is truthy (sends `mode=`), an empty string is not.
    for key in ("mode", "provider", "release"):
        if truthy(f.get(key)):
            query.append((key, join(f[key])))
    if truthy(f.get("includeDeprecated")):
        query.append(("includeDeprecated", "true"))
    if truthy(f.get("search")):
        query.append(("search", str(f["search"])))
    return "&".join(f"{_form_encode(k)}={_form_encode(v)}" for k, v in query)


async def _fetch_and_install(fetch: AuthenticatedFetch, api_url: str, filters: ModelCatalogFilters | None) -> None:
    url = f"{re.sub(r'/+$', '', api_url)}{CATALOG_PATH}?{build_query(filters)}"
    res = await fetch(url, {"method": "GET"})
    if not res.is_success:
        reason: str | None = None
        message = f"Model catalog request failed with status {res.status_code}"
        try:
            body = json.loads(await res.aread())
            if isinstance(body, dict):
                reason = body.get("reason")
                if truthy(body.get("message")):
                    message = body["message"]
        except ValueError:
            pass
        raise ApiError(
            message, status=res.status_code, code=reason if reason is not None else code_for_status(res.status_code)
        )
    install_catalog(deserialize_catalog(json.loads(await res.aread())))


def _clear_refresh_schedule() -> None:
    global _refresh_task
    if _refresh_task is not None:
        _refresh_task.cancel()
        _refresh_task = None


def _schedule_refresh(
    fetch: AuthenticatedFetch, api_url: str, filters: ModelCatalogFilters | None, seconds: float
) -> None:
    global _refresh_task
    _clear_refresh_schedule()
    if not (seconds > 0):
        return

    async def loop() -> None:
        while True:
            await asyncio.sleep(seconds)
            with contextlib.suppress(Exception):  # nobody to surface a failed tick to
                await refresh_model_catalog(fetch, api_url, filters)

    _refresh_task = asyncio.ensure_future(loop())


def load_model_catalog(
    fetch: AuthenticatedFetch,
    api_url: str,
    filters: ModelCatalogFilters | None = None,
    refresh_interval_seconds: float = -1,
) -> Awaitable[None]:
    """Load and install the catalog once per process; share the in-flight or settled load.

    Every caller gets a shielded view of the one shared task (a JS Promise cannot be cancelled): cancelling one
    waiter — a timeout, a disconnected request — never aborts the load the others wait on.
    """
    global _load, _key
    key = _key_of(api_url, filters)
    if _load is not None:
        if _key != key:
            source = (_key or "").split("::")[0]
            failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            failed.set_exception(
                ApiError(
                    f'Model catalog already {"loaded" if _loaded else "loading"} from "{source}" '
                    "— one catalog source per process.",
                    status=409,
                    code="catalog_conflict",
                )
            )
            return failed
        if _loaded:
            _schedule_refresh(fetch, api_url, filters, refresh_interval_seconds)
        return asyncio.shield(_load)

    async def run() -> None:
        global _load, _key, _loaded
        try:
            await _fetch_and_install(fetch, api_url, filters)
        except BaseException:
            _load = None
            _key = None
            raise
        _loaded = True
        _schedule_refresh(fetch, api_url, filters, refresh_interval_seconds)

    _key = key
    _load = asyncio.ensure_future(run())
    return asyncio.shield(_load)


def refresh_model_catalog(
    fetch: AuthenticatedFetch, api_url: str, filters: ModelCatalogFilters | None = None
) -> Awaitable[None]:
    """Re-fetch and install now (never dedups, never conflicts); a failure restores the previous state.

    Readers that join while the refresh is in flight (``when_model_catalog_loaded``, a same-source
    ``load_model_catalog``) wait on a reader view: it settles with the refresh on success, and with the previous
    load when the refresh fails — the catalog that stays installed is still valid for them. Only the caller of
    this function sees the refresh's own failure.
    """
    global _load, _loaded, _key
    prev = (_load, _loaded, _key)
    prev_load = prev[0]

    async def run() -> None:
        global _load, _loaded, _key
        try:
            await _fetch_and_install(fetch, api_url, filters)
        except BaseException:
            if _load is reader:
                _load, _loaded, _key = prev
            raise
        _loaded = True

    async def read() -> None:
        await asyncio.wait({task})
        if not task.cancelled() and task.exception() is None:
            return
        if prev_load is not None:
            await asyncio.shield(prev_load)
            return
        task.result()  # nothing to fall back to: re-raise the refresh failure

    task: asyncio.Future[None] = asyncio.ensure_future(run())
    reader: asyncio.Future[None] = asyncio.ensure_future(read())
    # Nobody may be reading: a failure is still retrieved, never logged.
    reader.add_done_callback(lambda f: f.cancelled() or f.exception())
    _load, _loaded, _key = reader, False, _key_of(api_url, filters)
    return asyncio.shield(task)


def when_model_catalog_loaded() -> Awaitable[None] | None:
    """The current (shielded) load, or None before the first ``load_model_catalog``."""
    return asyncio.shield(_load) if _load is not None else None


def is_model_catalog_loaded() -> bool:
    return _loaded


def _reset_model_catalog_store_for_tests() -> None:
    global _load, _loaded, _key
    _clear_refresh_schedule()
    _load, _loaded, _key = None, False, None

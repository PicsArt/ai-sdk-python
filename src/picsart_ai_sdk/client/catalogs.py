"""``ai.catalogs`` — runtime access to the platform catalog tasks — port of ``src/client/catalogs.ts``.

The tasks (``<vendor>/v1/catalog/<voices|avatars|templates>``) serve every catalog-bound param
(``kind: "catalog"``): voices, avatars, effect templates. Each call fetches ONE page (a UI loads more via
``next_cursor``). Pages are cached for the ``ttlSeconds`` the platform returns (at least 60 s), concurrent
fetches of one page share a request, and every fetched page is appended to the options installed on the
model, so ``Model(id).params()`` — and every picker built on it — sees what has been loaded so far. With
``create_client(catalogs=CatalogsConfig(preload=True))`` the first page of every bound catalog loads in the
background (when the client is created inside a running event loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, cast

from ..abort import AbortError, AbortSignal
from ..core.active_catalog import peek_active_models
from ..core.catalogs import CatalogItem, CatalogSource, install_hydrated_catalog
from ..core.errors import ApiError
from ..core.js import truthy
from ..core.model_registry import resolve_model
from ..core.response import throw_if_error_result
from ..core.types import ModelDefinition
from ..core.workflow import SdkTransport, WorkflowSubmitRequest

DEFAULT_LIMIT = 100
"""Every catalog task accepts up to 100."""

MIN_TTL_SECONDS = 60
"""Serve-stale floor, so a worker sending ``ttlSeconds: 0`` cannot disable caching."""


@dataclass
class CatalogPage:
    """One fetched page. Pass ``next_cursor`` back to load the next one; ``None`` when the list is complete."""

    items: list[CatalogItem]
    next_cursor: str | None


@dataclass
class _SourceStore:
    # key: request cursor ("" = first page) and effective limit — a page of 1 is not a page of 100
    pages: dict[str, CatalogPage] = field(default_factory=dict)
    version: str = ""
    expires_at: float = 0
    gen: int = 0  # bumped on reset: a fetch started before it must not write back


def _now_ms() -> float:
    return time.time() * 1000


def _copy(page: CatalogPage) -> CatalogPage:
    """Callers get deep copies — the cached page (and the options hydrated from it) must survive caller
    mutation of the list, an item or its nested tags/preview/meta."""
    return CatalogPage(items=copy.deepcopy(page.items), next_cursor=page.next_cursor)


def _field(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _abort_error(signal: AbortSignal) -> BaseException:
    reason = signal.reason
    if isinstance(reason, BaseException):
        return reason
    return AbortError("The catalog load was aborted." if reason is None else str(reason))


async def _abortable(shared: asyncio.Future[CatalogPage], signal: AbortSignal | None) -> CatalogPage:
    """Wait for a shared fetch; this caller's abort or cancellation rejects only this caller."""
    if signal is None:
        # Not `shield`: since 3.12 a cancelled shield logs the shared fetch's later failure. `wait` neither
        # cancels the shared task nor reports it.
        await asyncio.wait({shared})
        return _copy(shared.result())
    if signal.aborted:
        raise _abort_error(signal)
    aborted: asyncio.Future[Any] = asyncio.ensure_future(signal.wait())
    racers: set[asyncio.Future[Any]] = {shared, aborted}
    try:
        await asyncio.wait(racers, return_when=asyncio.FIRST_COMPLETED)
    finally:
        aborted.cancel()
    if not shared.done():
        raise _abort_error(signal)
    return _copy(shared.result())


def _ttl_seconds(value: Any) -> float:
    """``Math.max(MIN_TTL_SECONDS, ttlSeconds || 0)`` (NaN when the value is not a number)."""
    value = value if truthy(value) else 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if math.isnan(number) else max(MIN_TTL_SECONDS, number)


class CatalogsClient:
    """``ai.catalogs``: pages of a model's voice, avatar and effect-template catalogs."""

    def __init__(self, transport: SdkTransport, *, preload: bool = False) -> None:
        self._transport = transport
        self._stores: dict[str, _SourceStore] = {}
        self._inflight: dict[str, asyncio.Future[CatalogPage]] = {}
        self._background: set[asyncio.Future[Any]] = set()
        if preload:
            self._preload()

    async def voices(
        self,
        model: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
        signal: AbortSignal | None = None,
    ) -> CatalogPage:
        """One page of the model's voice catalog (``voiceId``). Pages accumulate into
        ``Model(id).params().catalog("voiceId")["catalogOptions"]``."""
        return await self._load_param(model, "voiceId", cursor, limit, force_refresh, signal)

    async def avatars(
        self,
        model: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
        signal: AbortSignal | None = None,
    ) -> CatalogPage:
        """One page of the model's avatar catalog (``videoId``) — same semantics as :meth:`voices`."""
        return await self._load_param(model, "videoId", cursor, limit, force_refresh, signal)

    async def templates(
        self,
        model: str,
        *,
        cursor: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
        signal: AbortSignal | None = None,
    ) -> CatalogPage:
        """One page of the model's effect-template catalog (``templateId``) — same semantics."""
        return await self._load_param(model, "templateId", cursor, limit, force_refresh, signal)

    # ── internals ───────────────────────────────────────────────────────
    @staticmethod
    def _key_of(source: CatalogSource) -> str:
        return f"{source['workflow']} {source.get('modelId') or ''}"

    async def _fetch_page(
        self, workflow: str, model_id: str | None, cursor: str | None, limit: int | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if truthy(model_id):
            payload["modelId"] = model_id
        if truthy(cursor):
            payload["cursor"] = cursor
        if truthy(limit):
            payload["limit"] = limit
        try:
            res = await self._transport.execute(WorkflowSubmitRequest(workflow, payload))
        except (ApiError, AbortError, asyncio.CancelledError):
            raise
        except Exception as err:
            # A custom transport may raise anything; callers only ever see ApiError.
            raise ApiError(f"{workflow} failed: {err}", status=502, code="bad_gateway") from err
        # The transport hands back the task result; a transport that keeps the platform envelope is unwrapped.
        body = res.result if res.result is not None else res.raw
        container = _field(body, "response")
        container = body if container is None else container
        result = _field(container, "result")
        result = container if result is None else result
        # A failed catalog task resolves with its error payload in place of the result.
        throw_if_error_result(result, workflow)
        if not truthy(result) or not isinstance(_field(result, "items"), list):
            raise ApiError(f"{workflow} returned no catalog result", status=502, code="invalid_response")
        next_cursor = result.get("nextCursor")
        return {**result, "nextCursor": next_cursor}

    def _store_for(self, source: CatalogSource, force_refresh: bool) -> _SourceStore:
        key = self._key_of(source)
        store = self._stores.get(key)
        if store is None:
            store = self._stores[key] = _SourceStore()
            return store
        # Expired or forced: drop the pages but keep the store; the gen bump turns any in-flight fetch from
        # the old snapshot into a no-op.
        if force_refresh or (store.expires_at != 0 and store.expires_at <= _now_ms()):
            store.pages.clear()
            store.version = ""
            store.expires_at = 0
            store.gen += 1
        return store

    @staticmethod
    def _accumulated(store: _SourceStore) -> list[CatalogItem]:
        """Everything fetched so far, in fetch order, deduped by id (a later copy replaces an earlier one)."""
        by_id: dict[str, CatalogItem] = {}
        for page in store.pages.values():
            for item in page.items:
                by_id[item["id"]] = item
        return list(by_id.values())

    async def _load_page(
        self,
        definition: ModelDefinition,
        param_key: str,
        source: CatalogSource,
        cursor: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
        signal: AbortSignal | None = None,
    ) -> CatalogPage:
        store = self._store_for(source, force_refresh)
        effective_limit = limit if limit is not None else DEFAULT_LIMIT
        cursor_key = f"{cursor if cursor is not None else ''} {effective_limit}"

        cached = store.pages.get(cursor_key)
        if cached is not None:
            if signal is not None and signal.aborted:
                raise _abort_error(signal)
            return _copy(cached)

        inflight_key = f"{self._key_of(source)} {cursor_key}"
        if not force_refresh:  # a forced refresh must not join a fetch that predates the reset
            pending = self._inflight.get(inflight_key)
            if pending is not None:
                return await _abortable(pending, signal)

        gen = store.gen

        async def run() -> CatalogPage:
            try:
                res = await self._fetch_page(source["workflow"], source.get("modelId"), cursor, effective_limit)
                page = CatalogPage(items=res["items"], next_cursor=res["nextCursor"])
                if store.gen == gen:
                    store.pages[cursor_key] = page
                    store.version = res.get("version")  # type: ignore[assignment]
                    if store.expires_at == 0:
                        store.expires_at = _now_ms() + _ttl_seconds(res.get("ttlSeconds")) * 1000
                    install_hydrated_catalog(
                        source, param_key, self._accumulated(store), definition.provider, store.version
                    )
                return page
            finally:
                if self._inflight.get(inflight_key) is task:
                    del self._inflight[inflight_key]

        # The fetch itself takes no signal: it is shared by every caller of this page, and its result feeds
        # the cache either way.
        task: asyncio.Future[CatalogPage] = asyncio.ensure_future(run())
        # Every caller may have given up (aborted, cancelled): the failure is still retrieved, never logged.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        self._inflight[inflight_key] = task
        return await _abortable(task, signal)

    @staticmethod
    def _require_source(definition: ModelDefinition, key: str) -> CatalogSource:
        entry = definition.param_config.get(key)
        descriptor = entry.descriptor if entry is not None else None
        source = descriptor.get("source") if descriptor is not None and descriptor.get("kind") == "catalog" else None
        if not truthy(source):
            raise Exception(  # noqa: TRY002 — TS throws a plain Error
                f'Model "{definition.id}" has no runtime catalog on param "{key}" — its options are static.'
            )
        return cast(CatalogSource, source)

    async def _load_param(
        self,
        model: str,
        key: str,
        cursor: str | None,
        limit: int | None,
        force_refresh: bool,
        signal: AbortSignal | None,
    ) -> CatalogPage:
        definition = resolve_model(model)
        source = self._require_source(definition, key)
        return await self._load_page(definition, key, source, cursor, limit, force_refresh, signal)

    def _preload(self) -> None:
        """First page of every bound catalog, in the background; failures are silent by design (a picker
        opened later just loads normally). No catalog installed yet, or no running loop: nothing to do."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        seen: set[str] = set()
        for definition in peek_active_models() or []:
            for key, entry in definition.param_config.items():
                descriptor = entry.descriptor
                source = descriptor.get("source") if descriptor.get("kind") == "catalog" else None
                if not truthy(source) or self._key_of(source) in seen:  # type: ignore[arg-type]
                    continue
                seen.add(self._key_of(source))  # type: ignore[arg-type]
                task = asyncio.ensure_future(self._quiet(self._load_page(definition, key, source)))  # type: ignore[arg-type]
                self._background.add(task)
                task.add_done_callback(self._background.discard)

    @staticmethod
    async def _quiet(load: Any) -> None:
        with contextlib.suppress(Exception):
            await load


def create_catalogs(transport: SdkTransport, *, preload: bool = False) -> CatalogsClient:
    """Build ``ai.catalogs`` over the client's transport."""
    return CatalogsClient(transport, preload=preload)


__all__ = ["DEFAULT_LIMIT", "MIN_TTL_SECONDS", "CatalogPage", "CatalogsClient", "create_catalogs"]

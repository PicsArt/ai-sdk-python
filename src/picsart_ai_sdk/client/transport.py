"""Authenticated fetch and the built-in :class:`SdkTransport` — port of ``src/client/transport.ts``."""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast

import httpx

from ..abort import AbortSignal
from ..core.errors import ApiError
from ..core.workflow import (
    OptionsResult,
    TransportPollOptions,
    TransportResult,
    WorkflowJobHandle,
    WorkflowSubmitRequest,
)
from .errors import to_api_error
from .fetch import AuthenticatedFetch, FetchInit, send
from .workflows import ExecutionMode, WorkflowsClient

# Attribution headers the gateway requires on every request; set only when absent.
GATEWAY_HEADERS: dict[str, str] = {"platform": "api", "X-Touchpoint": "sdk"}


def maybe_fetch(
    *, fetch: AuthenticatedFetch | None, api_key: str | None, http: httpx.AsyncClient | None
) -> AuthenticatedFetch | None:
    """The caller's ``fetch``, else one built from ``api_key`` on ``http``, else ``None``."""
    if fetch is not None:
        return fetch
    if api_key and http is not None:
        token = re.sub(r"^Bearer\s+", "", api_key, flags=re.IGNORECASE)

        async def authed(url: str, init: FetchInit) -> httpx.Response:
            headers = {**(init.get("headers") or {})}
            present = {k.lower() for k in headers}
            headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
            headers["Authorization"] = f"Bearer {token}"
            for name, value in GATEWAY_HEADERS.items():
                if name.lower() not in present:
                    headers[name] = value
            return await send(http, url, {**init, "headers": headers})

        return authed
    return None


def create_workflows_client(api_url: str, fetch: AuthenticatedFetch) -> WorkflowsClient:
    return WorkflowsClient(api_url, fetch)


def _raise_mapped(err: BaseException, workflow: str, task_id: str | None = None) -> None:
    mapped = to_api_error(err, workflow, task_id)
    if mapped is err:
        raise err
    raise mapped from err


def _as_result(res: dict[str, Any]) -> TransportResult:
    return TransportResult(result=res.get("result"), usage=res.get("usage"), raw=res.get("result"))


class BuiltinTransport:
    """The default transport over the workflows client: execute (SYNC), submit, poll, status, options."""

    def __init__(self, wc: WorkflowsClient) -> None:
        self._wc = wc

    async def execute(self, request: WorkflowSubmitRequest) -> TransportResult:
        try:
            res = await self._wc.run(request.workflow, request.payload, mode=ExecutionMode.SYNC, signal=request.signal)
            return _as_result(res)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_mapped(err, request.workflow)
            raise

    async def submit(self, request: WorkflowSubmitRequest) -> str:
        try:
            task_id = await self._wc.submit(request.workflow, request.payload)
            if not task_id:
                raise ApiError("No task id in response", status=502, code="invalid_response")
            return task_id
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_mapped(err, request.workflow)
            raise

    async def poll(self, handle: WorkflowJobHandle, options: TransportPollOptions | None = None) -> TransportResult:
        o = options or TransportPollOptions()
        try:
            res = await self._wc.run_polling(
                handle.workflow,
                handle.id,
                polling_interval=o.interval_ms,
                retries_count=o.max_attempts,
                signal=o.signal,
                on_progress=o.on_progress,  # type: ignore[arg-type]
            )
            return _as_result(res)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_mapped(err, handle.workflow, handle.id)
            raise

    async def status(self, handle: WorkflowJobHandle, signal: AbortSignal | None = None) -> TransportResult:
        if signal is not None and signal.aborted:
            raise ApiError("Operation aborted", status=499, code="aborted")
        try:
            return _as_result(await self._wc.result(handle.workflow, handle.id))
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_mapped(err, handle.workflow, handle.id)
            raise

    async def options(self, workflow: str, payload: dict[str, Any]) -> OptionsResult | None:
        # Pricing is best-effort: an estimate never fails a generation flow.
        try:
            res = await self._wc.options(workflow, payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
        credits = res.get("credits") if isinstance(res, dict) else None
        if isinstance(credits, bool) or not isinstance(credits, (int, float)):
            return None
        return cast(OptionsResult, {**res, "credits": credits})


def build_transport(wc: WorkflowsClient) -> BuiltinTransport:
    return BuiltinTransport(wc)

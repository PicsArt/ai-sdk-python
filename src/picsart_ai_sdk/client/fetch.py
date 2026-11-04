"""The ``fetch`` the SDK makes every request through (the TS ``AuthenticatedFetch``).

A callable ``(url, init) -> httpx.Response``. ``init`` carries what the TS
``RequestInit`` does here: ``method``, ``headers`` (a plain dict — the workflows
client hands lower-cased names, as TS ``Object.fromEntries(headers)`` does),
``body`` (a string), ``signal`` and, for SSE, ``stream``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypedDict

import httpx

from ..abort import AbortError, AbortSignal


class FetchInit(TypedDict, total=False):
    method: str
    headers: dict[str, str]
    body: str
    signal: AbortSignal | None
    # Return before the body is read (Server-Sent Events); the caller closes the response.
    stream: bool


AuthenticatedFetch = Callable[[str, FetchInit], Awaitable[httpx.Response]]


class _AbortableStream(httpx.AsyncByteStream):
    """A streamed body that stays bound to ``signal``: an abort fails the pending read, as a fetch body does."""

    def __init__(self, inner: httpx.AsyncByteStream, signal: AbortSignal) -> None:
        self._inner = inner
        self._signal = signal

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunks = self._inner.__aiter__()
        while True:
            if self._signal.aborted:
                raise AbortError()
            next_task = asyncio.ensure_future(chunks.__anext__())
            abort_task = asyncio.ensure_future(self._signal.wait())
            try:
                done, _ = await asyncio.wait({next_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                abort_task.cancel()
                if not next_task.done():
                    next_task.cancel()
            if next_task not in done:
                raise AbortError()
            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                return
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


async def send(http: httpx.AsyncClient, url: str, init: FetchInit) -> httpx.Response:
    """Send one request on ``http``; an aborted ``signal`` raises :class:`AbortError` (TS DOMException)."""
    request = http.build_request(
        init.get("method", "GET"),
        url,
        headers=init.get("headers"),
        content=init.get("body"),
    )
    signal = init.get("signal")
    stream = init.get("stream", False)
    if signal is None:
        return await http.send(request, stream=stream)
    signal.throw_if_aborted()
    send_task = asyncio.ensure_future(http.send(request, stream=stream))
    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _ = await asyncio.wait({send_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        abort_task.cancel()
        if not send_task.done():
            send_task.cancel()  # aborted, or our caller was cancelled: drop the in-flight request
    if send_task not in done:
        raise AbortError()
    response = send_task.result()
    if stream:
        # Headers are in but the body is not: keep the abort live while the caller reads it.
        response.stream = _AbortableStream(response.stream, signal)  # type: ignore[arg-type]
    return response

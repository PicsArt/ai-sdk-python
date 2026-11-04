"""Workflows HTTP client — port of ``@picsart/workflows-client`` 2.2.1 (the parts the SDK uses).

The TS SDK talks to the platform only through this client, so the requests here
(paths, headers, bodies), the polling cadence and the errors are kept
byte-for-byte. SOCKET mode (socket.io) and remote settings are not ported: the
SDK passes no ``getRemoteSettings``, so ``x-config-id`` is always ``""``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from ..abort import AbortError, AbortSignal
from ..core.js import js_json_dumps, truthy
from .fetch import AuthenticatedFetch, FetchInit
from .sse import decode_sse

log = logging.getLogger("picsart_ai_sdk")

DEFAULT_REASON = "unknown_error"
DEFAULT_MESSAGE = "Unknown error"
NON_JSON_BODY_MESSAGE = "Non json response was returned from server"
DEFAULT_POLLING_INTERVAL = 300
DEFAULT_RETRIES_COUNT = 1000
NETWORK_RETRIES_COUNT = 10
MAX_POLLING_BACKOFF = 5000
TERMINAL_STATUSES = ("COMPLETED", "FAILED")

_sleep = asyncio.sleep  # patched in tests


class ExecutionMode(str, Enum):
    ASYNC = "ASYNC"
    SYNC = "SYNC"
    STREAM = "STREAM"
    SOCKET = "SOCKET"


def _as_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_status(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) else None


class WorkflowsError(Exception):
    """An error of the workflows API: ``reason``, ``message``, optional ``http_status_code``."""

    name = "WorkflowsError"

    def __init__(self, *, reason: str, message: str, http_status_code: int | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.http_status_code = http_status_code

    @classmethod
    def from_body(cls, body: object, *, http_status_code: int | None, reason: str, message: str) -> WorkflowsError:
        """From a parsed error payload; each field falls back individually."""
        parsed = body if isinstance(body, dict) else {}
        status = _as_status(parsed.get("statusCode"))
        return cls(
            reason=_as_text(parsed.get("reason")) or reason,
            message=_as_text(parsed.get("message")) or message,
            http_status_code=status if status is not None else http_status_code,
        )

    @classmethod
    async def from_response(cls, response: httpx.Response, *, reason: str | None = None) -> WorkflowsError:
        """From a non-OK response: status from the response, reason/message from its JSON body."""
        try:
            body: object = json.loads(await response.aread())
        except ValueError:
            body = None
        # JS `asRecord`: any object, arrays included
        parsed = body if isinstance(body, dict) else ({} if isinstance(body, list) else None)
        return cls(
            reason=_as_text((parsed or {}).get("reason")) or reason or DEFAULT_REASON,
            message=_as_text((parsed or {}).get("message"))
            or (DEFAULT_MESSAGE if parsed is not None else NON_JSON_BODY_MESSAGE),
            http_status_code=response.status_code,
        )

    @classmethod
    def from_unknown(cls, error: BaseException) -> WorkflowsError:
        return cls(reason=DEFAULT_REASON, message=str(error) or DEFAULT_MESSAGE)


def _wrap_error(error: BaseException) -> BaseException:
    if isinstance(error, (WorkflowsError, AbortError)):
        return error
    return WorkflowsError.from_unknown(error)


def _reraise_wrapped(error: BaseException) -> None:
    """``throw this.wrapError(name, err)``: re-raise as is, or as a WorkflowsError caused by it."""
    wrapped = _wrap_error(error)
    if wrapped is error:
        raise error
    raise wrapped from error


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?(Z|[+-]\d{2}:?\d{2})?$")


def _parse_updated(value: object) -> float:
    """``new Date(updated).getTime()`` in ms (JS keeps milliseconds); NaN when unparsable."""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return math.nan
    match = _ISO.match(value.strip())
    if not match:
        return math.nan
    year, month, day, hour, minute, second, fraction, zone = match.groups()
    millis = int((fraction or "0")[:3].ljust(3, "0"))
    try:
        stamp = datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0))
    except ValueError:
        return math.nan
    offset = 0
    if zone and zone != "Z":
        sign = 1 if zone[0] == "+" else -1
        hh, mm = zone[1:].replace(":", "")[:2], zone[1:].replace(":", "")[2:]
        offset = sign * (int(hh) * 60 + int(mm))
    epoch = (stamp - datetime(1970, 1, 1)).total_seconds() - offset * 60
    return epoch * 1000 + millis


class _UpdateAware:
    """Port of ``WorkflowResultUpdateAware``: progress / partial results / events per changed update."""

    def __init__(
        self,
        on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
        on_partial_result: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
        on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
    ) -> None:
        self.on_progress = on_progress
        self.on_partial_result = on_partial_result
        self.on_event = on_event
        self.updated: float | None = None
        self.last_event_id: object = None

    async def on_update(self, response: dict[str, Any]) -> None:
        if not truthy(response.get("updated")):
            return
        await self._deliver_events(response)
        if response.get("status") != "IN_PROGRESS":
            return
        new_updated = _parse_updated(response.get("updated"))
        if self.updated is None or not new_updated == self.updated:
            self.updated = new_updated
            await _call(self.on_partial_result, response)
            if truthy(response.get("progress")):
                await _call(self.on_progress, response["progress"])

    async def _deliver_events(self, response: dict[str, Any]) -> None:
        events = response.get("events") or []
        if not events or not self.on_event:
            return
        start = 0
        if self.last_event_id:
            for i, event in enumerate(events):
                if isinstance(event, dict) and event.get("id") == self.last_event_id:
                    start = i + 1
                    break
        new_events = events[start:]
        for event in new_events:
            await _call(self.on_event, event)
        if new_events:
            self.last_event_id = new_events[-1].get("id") if isinstance(new_events[-1], dict) else None


async def _call(fn: Callable[[Any], Awaitable[None] | None] | None, arg: Any) -> None:
    if fn is None:
        return
    result = fn(arg)
    if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
        await result


def _is_connection_error(error: BaseException) -> bool:
    """Anything the server never answered (not an abort, not a timeout, no HTTP status)."""
    # undici's header/body timeouts reach TS as `TypeError: fetch failed` and are retried;
    # only aborts and AbortSignal.timeout() (DOMException TimeoutError) are not.
    if isinstance(error, (AbortError, asyncio.TimeoutError)):
        return False
    if isinstance(error, WorkflowsError):
        return error.http_status_code is None
    return True


def polling_delay(interval: int, consecutive_failures: int) -> int:
    if consecutive_failures == 0:
        return interval
    return min(interval * (1 << consecutive_failures), max(interval, MAX_POLLING_BACKOFF))


class WorkflowsClient:
    """The workflows API at ``{base_url}workflows`` through an authenticated ``fetch``."""

    def __init__(self, base_url: str, fetch: AuthenticatedFetch) -> None:
        base = base_url or "https://api.picsart.com/"
        if not base.endswith("/"):
            base += "/"
        self.base_url = base
        self.workflows_api_base_url = f"{base}workflows"
        self._fetch_impl = fetch

    # ── requests ────────────────────────────────────────────────────────
    async def _fetch(self, url: str, init: FetchInit) -> httpx.Response:
        headers = {k.lower(): v for k, v in (init.get("headers") or {}).items()}
        headers.setdefault("accept", "application/json")
        headers.setdefault("content-type", "application/json")
        return await self._fetch_impl(url, {**init, "headers": headers})

    @staticmethod
    def _request_headers(headers: dict[str, str] | None = None) -> dict[str, str]:
        merged = {k.lower(): v for k, v in (headers or {}).items()}
        merged.setdefault("x-config-id", "")
        return merged

    @staticmethod
    def _body(params: object) -> str:
        return js_json_dumps({"params": params})

    @staticmethod
    async def _throw_if_error(response: httpx.Response) -> None:
        if not response.is_success:
            raise await WorkflowsError.from_response(response, reason="request_failed")

    async def _to_success_response(self, response: httpx.Response) -> dict[str, Any]:
        await self._throw_if_error(response)
        try:
            body: dict[str, Any] = json.loads(await response.aread())
        except ValueError:
            raise WorkflowsError(
                http_status_code=response.status_code, reason="invalid_response", message=NON_JSON_BODY_MESSAGE
            ) from None
        return body

    # ── API ─────────────────────────────────────────────────────────────
    async def submit(self, name: str, params: object, *, headers: dict[str, str] | None = None) -> str:
        try:
            return await self._post_task(name, params, headers)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _reraise_wrapped(err)
            raise

    async def _post_task(self, name: str, params: object, headers: dict[str, str] | None = None) -> str:
        # `notification` is undefined in the SDK, so JSON.stringify drops it.
        response = await self._fetch(
            f"{self.workflows_api_base_url}/{name}/submit",
            {"method": "POST", "headers": self._request_headers(headers), "body": self._body(params)},
        )
        body = await self._to_success_response(response)
        return (body.get("response") or {}).get("id")  # type: ignore[return-value]

    async def execute_sync(
        self, name: str, params: object, *, signal: AbortSignal | None = None, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        response = await self._fetch(
            f"{self.workflows_api_base_url}/{name}/execute",
            {"method": "POST", "headers": self._request_headers(headers), "body": self._body(params), "signal": signal},
        )
        return (await self._to_success_response(response)).get("response")  # type: ignore[return-value]

    async def get_result(self, name: str, task_id: str) -> dict[str, Any]:
        response = await self._fetch(f"{self.workflows_api_base_url}/{name}/{task_id}/result", {"method": "GET"})
        return await self._to_success_response(response)

    async def result(self, name: str, task_id: str) -> dict[str, Any]:
        """The task as it stands now (one request, no polling)."""
        try:
            return (await self.get_result(name, task_id)).get("response")  # type: ignore[return-value]
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _reraise_wrapped(err)
            raise

    async def options(self, name: str, params: object = None) -> dict[str, Any]:
        try:
            response = await self._fetch(
                f"{self.workflows_api_base_url}/{name}/options",
                {
                    "method": "POST",
                    "headers": self._request_headers(),
                    "body": self._body({} if params is None else params),
                },
            )
            return (await self._to_success_response(response)).get("response")  # type: ignore[return-value]
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _reraise_wrapped(err)
            raise

    async def run_polling(
        self,
        name: str,
        task_id: str,
        *,
        polling_interval: int | None = None,
        retries_count: int | None = None,
        signal: AbortSignal | None = None,
        on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_partial_result: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> dict[str, Any]:
        """Poll until COMPLETED/FAILED; returns the task ``response``. JS ``||`` defaults: 0 → default."""
        interval = polling_interval or DEFAULT_POLLING_INTERVAL
        retries = retries_count or DEFAULT_RETRIES_COUNT
        polling: dict[str, Any] | None = None
        network_failures = 0
        last_network_error: BaseException | None = None
        aware = _UpdateAware(on_progress, on_partial_result, on_event)
        while True:
            if signal is not None and signal.aborted:
                raise AbortError("Aborted")
            await _sleep(polling_delay(interval, network_failures) / 1000)
            retries -= 1
            try:
                polling = await self.get_result(name, task_id)
                network_failures = 0
                last_network_error = None
            except asyncio.CancelledError:
                raise
            except Exception as err:
                if not _is_connection_error(err):
                    raise
                network_failures += 1
                if network_failures > NETWORK_RETRIES_COUNT:
                    raise self._connection_error(err) from err
                last_network_error = err
                log.warning(
                    "workflows.runPolling - poll %s/%s of %s/%s did not reach the server, retrying",
                    network_failures,
                    NETWORK_RETRIES_COUNT,
                    name,
                    task_id,
                )
                if retries > 0 and not self._is_terminal(polling):
                    continue
                break
            await aware.on_update(polling["response"])
            if not (retries > 0 and not self._is_terminal(polling)):
                break
        if last_network_error is not None:
            raise self._connection_error(last_network_error)
        # JS `!response.result`: an empty `{}` / `[]` result is truthy there, so it completes the poll.
        if not self._is_terminal(polling) or not truthy((polling or {}).get("response", {}).get("result")):
            raise WorkflowsError(
                http_status_code=408,
                reason="client_timeout",
                message=(
                    "Polling timeout reached. Consider increasing polling interval or retries count "
                    "from execution options."
                ),
            )
        assert polling is not None
        return polling.get("response")  # type: ignore[return-value]

    @staticmethod
    def _connection_error(error: BaseException) -> WorkflowsError:
        return WorkflowsError(reason="connection_error", message=str(error) or "The request did not reach the server")

    @staticmethod
    def _is_terminal(polling: dict[str, Any] | None) -> bool:
        return bool(polling) and (polling or {}).get("response", {}).get("status") in TERMINAL_STATUSES

    async def stream(
        self,
        name: str,
        params: object,
        *,
        on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None,
        on_partial_result: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        signal: AbortSignal | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if on_event is None:
            raise WorkflowsError(
                http_status_code=400, reason="INVALID_ARGUMENTS", message="onEvent is required for streaming"
            )
        stream_headers = self._request_headers(headers)
        stream_headers["accept"] = "text/event-stream"
        response = await self._fetch(
            f"{self.workflows_api_base_url}/{name}/stream",
            {"method": "POST", "headers": stream_headers, "body": self._body(params), "signal": signal, "stream": True},
        )
        try:
            await self._throw_if_error(response)
            completed: dict[str, Any] = {}
            async for event in decode_sse(response.aiter_bytes()):
                if signal is not None and signal.aborted:
                    # TS breaks here and reports COMPLETED with no result; raise like an aborted body read does.
                    raise AbortError("Aborted")
                data = event.get("data") or {}
                kind = str(data.get("type", ""))
                if kind.startswith("event."):
                    await _call(on_event, {**data, "type": kind[len("event.") :].lstrip()})
                if kind == "task.partial-result":
                    await _call(on_partial_result, {"status": "IN_PROGRESS", "result": data.get("result")})
                if kind == "task.failed":
                    raise WorkflowsError.from_body(
                        data.get("result"), http_status_code=500, reason="workflow_failed", message="Workflow failed"
                    )
                if kind == "task.completed":
                    completed = data
            return {"result": completed.get("result"), "usage": completed.get("usage"), "status": "COMPLETED"}
        finally:
            await response.aclose()

    async def run(
        self,
        name: str,
        params: object,
        *,
        mode: ExecutionMode | str = ExecutionMode.ASYNC,
        signal: AbortSignal | None = None,
        headers: dict[str, str] | None = None,
        on_accepted: Callable[[str], Awaitable[None] | None] | None = None,
        on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_partial_result: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        polling_interval: int | None = None,
        retries_count: int | None = None,
    ) -> dict[str, Any]:
        """Run a workflow in ``mode``. As in TS only the submit step is wrapped into ``WorkflowsError``."""
        mode = ExecutionMode(mode or ExecutionMode.ASYNC)
        if mode is ExecutionMode.SYNC:
            return await self.execute_sync(name, params, signal=signal, headers=headers)
        if mode is ExecutionMode.STREAM:
            return await self.stream(
                name, params, on_event=on_event, on_partial_result=on_partial_result, signal=signal, headers=headers
            )
        if mode is ExecutionMode.SOCKET:
            raise WorkflowsError(
                http_status_code=400, reason="unsupported_mode", message="SOCKET mode is not supported"
            )
        if signal is not None and signal.aborted:
            # Never create a billable task for a caller that already canceled.
            raise AbortError("Aborted")
        try:
            task_id = await self._post_task(name, params, headers)
            if on_accepted is not None:
                await _call(on_accepted, task_id)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _reraise_wrapped(err)
            raise
        return await self.run_polling(
            name,
            task_id,
            polling_interval=polling_interval,
            retries_count=retries_count,
            signal=signal,
            on_progress=on_progress,
            on_partial_result=on_partial_result,
            on_event=on_event,
        )

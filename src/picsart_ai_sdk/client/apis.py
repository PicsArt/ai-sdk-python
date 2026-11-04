"""``ai.apis`` — direct, low-level access to the Picsart model APIs — port of ``src/client/apis.ts``.

The escape hatch beneath ``generate()`` / ``generate_text()``: run any model API by name over the client's
workflows client. Failures arrive as :class:`ApiError`, as on the generation surface.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..abort import AbortSignal
from ..core.errors import ApiError
from .errors import to_api_error
from .workflows import ExecutionMode, WorkflowsClient

ApiResponse = dict[str, Any]
"""Result of ``ai.apis.run()``: ``{"result", "status", "usage"?, "id"?, "updated"?, "progress"?}``."""


class ApisClient:
    """``ai.apis``. ``run`` takes the workflows client's execution options except ``remote_setting_name``,
    ``on_partial_result`` and ``notification_config``, which are not part of the SDK surface."""

    def __init__(self, client: WorkflowsClient | None) -> None:
        self._client = client

    async def run(
        self,
        api: str,
        payload: dict[str, Any],
        *,
        mode: ExecutionMode | str = ExecutionMode.ASYNC,
        signal: AbortSignal | None = None,
        headers: dict[str, str] | None = None,
        on_accepted: Callable[[str], Awaitable[None] | None] | None = None,
        on_progress: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        polling_interval: int | None = None,
        retries_count: int | None = None,
    ) -> ApiResponse:
        """Run an API by name (``WorkflowsClient.run``)."""
        if self._client is None:
            raise ApiError(
                "`ai.apis` requires `apiUrl` plus `fetch` or `apiKey` on createClient — the workflows APIs are "
                "not served by a custom transport.",
                status=400,
                code="unsupported_transport",
            )
        try:
            return await self._client.run(
                api,
                payload,
                mode=mode,
                signal=signal,
                headers=headers,
                on_accepted=on_accepted,
                on_progress=on_progress,
                on_event=on_event,
                polling_interval=polling_interval,
                retries_count=retries_count,
            )
        except asyncio.CancelledError:
            raise
        except Exception as err:
            # ApiError is the SDK's only error type: the workflows client's own error never reaches a caller.
            mapped = to_api_error(err, api)
            if mapped is err:
                raise
            raise mapped from err


def create_apis(client: WorkflowsClient | None) -> ApisClient:
    """``ai.apis`` over the shared workflows client (``None`` for a client on a custom transport only)."""
    return ApisClient(client)


__all__ = ["ApiResponse", "ApisClient", "create_apis"]

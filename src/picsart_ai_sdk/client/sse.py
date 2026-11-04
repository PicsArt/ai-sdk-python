"""Server-Sent Events decoding — port of the workflows-client ``decodeSSE`` / ``readSSE``."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger("picsart_ai_sdk")


def _double_newline_index(buffer: bytes) -> int:
    """End index of the first event separator (``\\n\\n``, ``\\r\\r`` or ``\\r\\n\\r\\n``), or -1."""
    for i in range(len(buffer) - 1):
        a, b = buffer[i], buffer[i + 1]
        if a == 10 and b == 10:
            return i + 2
        if a == 13 and b == 13:
            return i + 2
        # Same bound as TS (`i + 3 < length`): a trailing \r\n\r\n at the very end waits for more bytes.
        if a == 13 and b == 10 and i + 3 < len(buffer) and buffer[i + 2] == 13 and buffer[i + 3] == 10:
            return i + 4
    return -1


async def read_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Yield raw event chunks (separator included), then any trailing bytes."""
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while (index := _double_newline_index(buffer)) != -1:
            yield buffer[:index].decode("utf-8", errors="replace")
            buffer = buffer[index:]
    if buffer:
        yield buffer.decode("utf-8", errors="replace")


async def decode_sse(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Yield ``{"data": <parsed JSON>}`` per event (``{}`` when it has no parsable data); stop at ``[DONE]``."""
    async for chunk in read_sse(chunks):
        event: dict[str, Any] = {}
        for line in chunk.split("\n"):
            if line.startswith("data:"):
                # A CR-terminated line (CRLF or bare-CR events) keeps its ``\r``: drop it so ``[DONE]\r`` still stops.
                data = line[len("data:") :].lstrip().rstrip("\r")
                if data == "[DONE]":
                    return
                try:
                    event["data"] = json.loads(data)
                except ValueError:
                    log.warning("Failed to parse data JSON from OpenAI event stream: - %s", data)
        yield event

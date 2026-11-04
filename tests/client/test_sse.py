from collections.abc import AsyncIterator
from typing import Any

from picsart_ai_sdk.client.sse import decode_sse


async def chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def collect(*parts: bytes) -> list[dict[str, Any]]:
    return [event async for event in decode_sse(chunks(*parts))]


async def test_separators() -> None:
    assert await collect(b'data: {"a":1}\n\ndata: {"b":2}\r\rdata: {"c":3}\r\n\r\nx') == [
        {"data": {"a": 1}},
        {"data": {"b": 2}},
        {"data": {"c": 3}},
        {},
    ]


async def test_event_split_across_chunks() -> None:
    assert await collect(b'data: {"a"', b":1}\n", b'\ndata: {"b":2}') == [{"data": {"a": 1}}, {"data": {"b": 2}}]


async def test_done_stops() -> None:
    assert await collect(b'data: {"a":1}\n\ndata: [DONE]\n\ndata: {"b":2}\n\n') == [{"data": {"a": 1}}]


async def test_unparsable_data_yields_empty_event() -> None:
    assert await collect(b"data: {oops\n\n") == [{}]


async def test_crlf_done_stops() -> None:
    assert await collect(b'data: {"a":1}\r\n\r\ndata: [DONE]\r\n\r\ndata: {"b":2}\r\n\r\n') == [{"data": {"a": 1}}]

"""Compact JSON → raw DEFLATE → base64url — port of ``src/core/deeplink/codec.ts`` (fflate ``deflateSync``).

The compressed bytes may differ from fflate's; what matters is that each SDK decodes the other's links.
"""

from __future__ import annotations

import base64
import re
import zlib
from typing import Any

from ..js import js_json_dumps, js_json_parse

_ATOB_WHITESPACE = re.compile(r"[\t\n\f\r ]")
_BASE64 = re.compile(r"[A-Za-z0-9+/]*")


def _to_base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _atob(text: str) -> bytes:
    """``atob()``: forgiving base64 (ASCII whitespace dropped, padding optional); ``ValueError`` otherwise."""
    text = _ATOB_WHITESPACE.sub("", text)
    if len(text) % 4 == 0:
        text = text[:-2] if text.endswith("==") else text[:-1] if text.endswith("=") else text
    if len(text) % 4 == 1 or not _BASE64.fullmatch(text):
        raise ValueError("The string to be decoded is not correctly encoded.")
    return base64.b64decode(text + "=" * (-len(text) % 4))


def _from_base64url(text: str) -> bytes:
    return _atob(text.replace("-", "+").replace("_", "/"))


def encode(data: Any) -> str:
    """Compress and base64url-encode a JSON-serializable value."""
    deflate = zlib.compressobj(6, zlib.DEFLATED, -15)
    return _to_base64url(deflate.compress(js_json_dumps(data).encode("utf-8")) + deflate.flush())


def decode(encoded: str) -> Any:
    """Decode and decompress a base64url string back to a value; raises on any malformed input."""
    inflate = zlib.decompressobj(-15)
    raw = inflate.decompress(_from_base64url(encoded))
    if not inflate.eof:
        raise ValueError("unexpected EOF")
    # TextDecoder: invalid UTF-8 becomes U+FFFD and a leading BOM is dropped.
    return js_json_parse(raw.decode("utf-8-sig", errors="replace"))

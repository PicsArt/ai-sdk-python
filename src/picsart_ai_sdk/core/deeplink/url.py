"""Public deep-link entry points — port of ``src/core/deeplink/url.ts``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .deserialize import deserialize_payload
from .serialize import serialize_payload
from .types import DeepLinkResult


def encode_deep_link_payload(model_id: str, context: Mapping[str, Any]) -> str:
    """Encode a model id + context into a compressed base64url payload (attach it as e.g. ``?aistate=…``)."""
    return serialize_payload(model_id, context)


def decode_deep_link_payload(encoded: str) -> DeepLinkResult | None:
    """Decode a payload back into a :class:`DeepLinkResult`; ``None`` when it does not decode or validate."""
    return deserialize_payload(encoded)

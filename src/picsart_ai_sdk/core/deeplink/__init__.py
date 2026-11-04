"""Deep links — port of ``src/core/deeplink/*``."""

from .types import DeepLinkResult
from .url import decode_deep_link_payload, encode_deep_link_payload

__all__ = ["DeepLinkResult", "decode_deep_link_payload", "encode_deep_link_payload"]

"""Deep-link payload and result — port of ``src/core/deeplink/types.ts``.

Payload v1 keys: ``v`` (1), ``m`` model id, ``p`` prompt, ``np`` negative prompt, ``o`` non-default params,
``i`` image URLs, ``vi`` video URL, ``au`` audio URL, ``sf`` / ``ef`` start / end frame URLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeepLinkResult:
    """A decoded deep link."""

    model_id: str
    """Model id from the payload."""
    context: dict[str, Any] = field(default_factory=dict)
    """The partial generation context rebuilt from the payload (camelCase keys, as the catalog's params)."""
    model_known: bool = False
    """Whether the model id exists in the installed catalog."""
    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings (rejected URLs, unknown model)."""

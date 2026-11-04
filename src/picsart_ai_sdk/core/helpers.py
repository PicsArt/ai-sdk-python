"""Small model helpers — port of ``src/core/helpers.ts`` (internal)."""

from __future__ import annotations

from collections.abc import Mapping

from .js import truthy
from .types import ModelDefinition


def resolve_image_size(aspect_ratio: str | None, size: str | None, ar_map: Mapping[str, str]) -> str | None:
    """Pixel size from the aspect ratio (preferred) or the direct size: ``(ar && arMap[ar]) || size``."""
    mapped = ar_map.get(aspect_ratio) if truthy(aspect_ratio) else None  # type: ignore[arg-type]
    return mapped if truthy(mapped) else size


def badge_priority(model: ModelDefinition) -> int:
    """Sort priority from the badges: hot > new+popular > new > popular > rest."""
    badges = model.badge or []
    if "hot" in badges:
        return 0
    if "new" in badges and "popular" in badges:
        return 1
    if "new" in badges:
        return 2
    if "popular" in badges:
        return 3
    return 4

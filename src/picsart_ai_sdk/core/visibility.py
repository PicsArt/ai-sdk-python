"""Release visibility — port of ``src/core/visibility.ts``."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ModelDefinition

DEFAULT_VISIBLE_RELEASES: tuple[str, ...] = ("production", "general-availability")


def release_of(model: ModelDefinition) -> str:
    return model.release if model.release is not None else "production"


def is_visible_for_releases(model: ModelDefinition, releases: Sequence[str] = DEFAULT_VISIBLE_RELEASES) -> bool:
    """Deprecated models are always hidden; otherwise visible when the model's release is listed."""
    if model.deprecated:
        return False
    return release_of(model) in releases

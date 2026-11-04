"""The installed model catalog (process-global) — port of ``src/core/active-catalog.ts``."""

from __future__ import annotations

from .errors import ApiError
from .types import ModelDefinition

_models: list[ModelDefinition] | None = None
_generation = 0


def install_catalog(models: list[ModelDefinition]) -> None:
    global _models, _generation
    _models = models
    _generation += 1


def is_catalog_installed() -> bool:
    return _models is not None


def catalog_generation() -> int:
    return _generation


def active_models() -> list[ModelDefinition]:
    """The installed catalog; 409 ``catalog_not_loaded`` before ``init_model_catalog()``."""
    if _models is None:
        raise ApiError(
            "Model catalog not loaded — call ai.initModelCatalog() before using the model catalog.",
            status=409,
            code="catalog_not_loaded",
        )
    return _models


def peek_active_models() -> list[ModelDefinition] | None:
    """The installed catalog, or ``None`` while pending — never raises."""
    return _models


def _reset_active_catalog_for_tests() -> None:
    global _models, _generation
    _models = None
    _generation += 1

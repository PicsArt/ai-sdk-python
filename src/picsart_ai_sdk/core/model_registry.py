"""Model lookup over the installed catalog — port of ``src/core/model-registry.ts`` and ``resolve.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field

from .active_catalog import active_models, catalog_generation
from .errors import ApiError
from .types import ModelDefinition


@dataclass
class _Indexes:
    by_id: dict[str, ModelDefinition] = field(default_factory=dict)
    by_model_id: dict[str, ModelDefinition] = field(default_factory=dict)
    by_workflow: dict[str, ModelDefinition] = field(default_factory=dict)
    by_name: dict[str, ModelDefinition] = field(default_factory=dict)


_idx: _Indexes | None = None
_idx_gen = -1


def _build_indexes(models: list[ModelDefinition]) -> _Indexes:
    idx = _Indexes()
    for model in models:
        idx.by_id[model.id] = model  # `new Map(entries)`: the last duplicate id wins
    ambiguous: set[str] = set()
    for model in models:
        if model.model_id and model.model_id not in idx.by_model_id:
            idx.by_model_id[model.model_id] = model
        if model.workflow and model.workflow not in idx.by_workflow:
            idx.by_workflow[model.workflow] = model
        if model.edit_workflow and model.edit_workflow not in idx.by_workflow:
            idx.by_workflow[model.edit_workflow] = model
        name_key = (model.name or "").strip().lower()
        if not name_key or name_key in ambiguous:
            continue
        if name_key in idx.by_name:
            del idx.by_name[name_key]
            ambiguous.add(name_key)
            continue
        idx.by_name[name_key] = model
    return idx


def _indexes() -> _Indexes:
    global _idx, _idx_gen
    gen = catalog_generation()
    if _idx is None or _idx_gen != gen:
        _idx = _build_indexes(active_models())
        _idx_gen = gen
    return _idx


def get_model(model_id: str) -> ModelDefinition | None:
    """By catalog id, then by vendor ``modelId``."""
    idx = _indexes()
    return idx.by_id.get(model_id) or idx.by_model_id.get(model_id)


def find_model(ref: str) -> ModelDefinition | None:
    """By id / modelId, then by workflow (or edit workflow), then by a unique display name."""
    key = ref.strip()
    if not key:
        return None
    return get_model(key) or _indexes().by_workflow.get(key) or _indexes().by_name.get(key.lower())


def resolve_model(model_id: str) -> ModelDefinition:
    found = find_model(model_id)
    if found is None:
        raise ApiError(f'Unknown model: "{model_id}"', status=400, code="unknown_model")
    return found

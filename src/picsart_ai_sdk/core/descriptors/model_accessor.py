"""``Model(id)`` and ``catalog`` — port of ``src/core/descriptors/model-accessor.ts``.

Entries the accessors return are flat dicts ``{...meta, ...descriptor}`` with the TS keys
(``disabledReason``, ``requiredReason``, ``catalogOptions``…), as in TS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..active_catalog import active_models
from ..catalogs import get_hydrated_catalog
from ..constraints import ConditionalRequirement, NormalizedRestriction, evaluate_constraints, evaluate_requirements
from ..js import js_string
from ..model_registry import resolve_model
from ..types import ModelDefinition
from ..visibility import DEFAULT_VISIBLE_RELEASES, is_visible_for_releases
from .pricing import configure_pricing, get_credits_for_model, is_pricing_loaded, load_pricing
from .types import (
    CreditRange,
    CreditRangeContext,
    FlatParamEntry,
    ModelMeta,
    ParamEntry,
    ProviderInfo,
    ValidationResult,
)
from .utils import descriptors_to_schema, extract_defaults, transfer_values, validate_all

_META_KEYS = ("label", "required", "category", "catalogOptions")


def _flat(entry: ParamEntry, key: str | None = None) -> FlatParamEntry:
    meta: dict[str, Any] = {}
    if entry.label is not None:
        meta["label"] = entry.label
    if entry.required is not None:
        meta["required"] = entry.required
    if entry.category is not None:
        meta["category"] = entry.category
    if entry.catalog_options is not None:
        meta["catalogOptions"] = entry.catalog_options
    flat = {**({"key": key} if key is not None else {}), **meta, **entry.descriptor}
    if entry.descriptor.get("kind") == "catalog":
        hydrated = get_hydrated_catalog(entry.descriptor.get("source") or {})  # type: ignore[arg-type]
        if hydrated is not None:
            flat["catalogOptions"] = hydrated.catalog_options
    return flat


class ModelParamsAccessor:
    """Read a model's params: flat entries, kind-checked getters, defaults, schema, value transfer."""

    def __init__(self, definition: ModelDefinition) -> None:
        self._def = definition

    def param(self, key: str) -> FlatParamEntry | None:
        entry = self._def.param_config.get(key)
        return _flat(entry) if entry is not None else None

    def has_param(self, key: str) -> bool:
        return key in self._def.param_config

    def all(self) -> list[FlatParamEntry]:
        return [_flat(entry, key) for key, entry in self._def.param_config.items()]

    def _narrow(self, key: str, *kinds: str) -> FlatParamEntry | None:
        entry = self.param(key)
        return entry if entry is not None and entry.get("kind") in kinds else None

    def enum(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "enum")

    def catalog(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "catalog")

    def range(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "range")

    def boolean(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "boolean")

    def text(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "text")

    def file(self, key: str) -> FlatParamEntry | None:
        return self._narrow(key, "file")

    def prompt(self) -> FlatParamEntry | None:
        return self._narrow("prompt", "text")

    def aspect_ratio(self) -> FlatParamEntry | None:
        return self._narrow("aspectRatio", "enum")

    def duration(self) -> FlatParamEntry | None:
        return self._narrow("duration", "enum", "range")

    def resolution(self) -> FlatParamEntry | None:
        return self._narrow("resolution", "enum")

    def generate_audio(self) -> FlatParamEntry | None:
        return self._narrow("generateAudio", "boolean")

    def start_frame(self) -> FlatParamEntry | None:
        return self._narrow("startFrame", "file")

    def end_frame(self) -> FlatParamEntry | None:
        return self._narrow("endFrame", "file")

    def has_file_input(self) -> bool:
        return any(e.descriptor.get("kind") == "file" for e in self._def.param_config.values())

    def get_default(self, key: str) -> Any:
        entry = self._def.param_config.get(key)
        return entry.descriptor.get("default") if entry is not None else None

    def get_defaults(self) -> dict[str, Any]:
        return extract_defaults(self._def.param_config)

    def to_schema(self) -> dict[str, dict[str, Any]]:
        return descriptors_to_schema(self._def.param_config)

    def transfer_values(self, prev: dict[str, Any]) -> dict[str, Any]:
        return transfer_values(self._def.param_config, prev)


def _allowed_set(r: NormalizedRestriction) -> set[str]:
    return {js_string(v) for v in r.allowed or []}


class ConstrainedParamsAccessor(ModelParamsAccessor):
    """The params as the current values constrain them: disabled options / entries, conditional requirements."""

    def __init__(
        self,
        inner: ModelParamsAccessor,
        effects: dict[str, NormalizedRestriction],
        requirements: dict[str, ConditionalRequirement],
    ) -> None:
        super().__init__(inner._def)
        self._inner = inner
        self._effects = effects
        self._requirements = requirements

    def _require(self, key: str, entry: FlatParamEntry | None) -> FlatParamEntry | None:
        if entry is None:
            return None
        requirement = self._requirements.get(key)
        if requirement is None:
            return entry
        return {**entry, "required": True, "requiredReason": requirement.reason}

    def _decorate(self, key: str, entry: FlatParamEntry | None) -> FlatParamEntry | None:
        if entry is None:
            return None
        r = self._effects.get(key)
        decorated = entry
        if r is not None and entry.get("kind") == "enum":
            if r.kind == "disabled":
                options = [{**o, "disabled": True, "disabledReason": r.reason} for o in entry.get("options", [])]
                decorated = {**entry, "options": options, "disabled": True, "disabledReason": r.reason}
            else:
                allowed = _allowed_set(r)
                options = [
                    o if js_string(o.get("id")) in allowed else {**o, "disabled": True, "disabledReason": r.reason}
                    for o in entry.get("options", [])
                ]
                decorated = {**entry, "options": options}
        elif r is not None and r.kind == "disabled":
            decorated = {**entry, "disabled": True, "disabledReason": r.reason}
        return self._require(key, decorated)

    def param(self, key: str) -> FlatParamEntry | None:
        return self._decorate(key, self._inner.param(key))

    def all(self) -> list[FlatParamEntry]:
        return [self._decorate(e["key"], e) for e in self._inner.all()]  # type: ignore[misc]


class ModelDescriptor:
    """What ``Model(id)`` returns: params, constrained params, meta, validation and credit info."""

    def __init__(self, definition: ModelDefinition) -> None:
        self.id = definition.id
        self.name = definition.name
        self.api = ModelApi(definition.workflow, definition.edit_workflow)
        self._def = definition
        self._params: ModelParamsAccessor | None = None
        self._meta: ModelMeta | None = None

    def params(self) -> ModelParamsAccessor:
        if self._params is None:
            self._params = ModelParamsAccessor(self._def)
        return self._params

    def params_for(self, values: dict[str, Any]) -> ModelParamsAccessor:
        inner = self.params()
        effects = evaluate_constraints(self._def.constraints, values)
        requirements = evaluate_requirements(self._def.constraints, values)
        if not effects and not requirements:
            return inner
        return ConstrainedParamsAccessor(inner, effects, requirements)

    def validate(self, values: object) -> ValidationResult:
        if not isinstance(values, dict):
            return ValidationResult(valid=False, errors=[f'Invalid input for model "{self._def.id}"'])
        try:
            validate_all(self._def.param_config, values, self._def.constraints)
            return ValidationResult(valid=True)
        except Exception as err:  # TS catches anything thrown and reports its message
            return ValidationResult(valid=False, errors=[str(err)])

    def meta(self) -> ModelMeta:
        if self._meta is None:
            d = self._def
            self._meta = ModelMeta(
                mode=d.mode,
                input_type=d.input_type,
                description=d.description,
                features=list(d.features),  # type: ignore[arg-type]
                badges=d.badge if d.badge is not None else [],
                provider=ProviderInfo(d.provider, d.provider_name, d.provider_color, d.provider_label),
                added_at=d.added_at,
                release=d.release if d.release is not None else "production",
            )
        return self._meta

    def get_credits_info(self, ctx: CreditRangeContext | None = None) -> CreditRange | None:
        if self._def.model_id:
            by_model_id = get_credits_for_model(self._def.model_id, ctx)
            if by_model_id is not None:
                return by_model_id
        return get_credits_for_model(self._def.id, ctx)


@dataclass(frozen=True)
class ModelApi:
    workflow: str
    edit_workflow: str | None


def Model(model_id: str) -> ModelDescriptor:  # noqa: N802 — the TS name
    """The descriptor of a catalog model (409 ``catalog_not_loaded`` / 400 ``unknown_model``)."""
    return ModelDescriptor(resolve_model(model_id))


class _Pricing:
    configure = staticmethod(configure_pricing)
    load = staticmethod(load_pricing)
    is_loaded = staticmethod(is_pricing_loaded)


class _Catalog:
    """``catalog.all / find / search / pricing`` over the installed catalog (catalog order, visible releases)."""

    pricing = _Pricing()

    @staticmethod
    def all(*, release: Sequence[str] | None = None) -> list[ModelDescriptor]:
        releases = release if release is not None else DEFAULT_VISIBLE_RELEASES
        return [ModelDescriptor(m) for m in active_models() if is_visible_for_releases(m, releases)]

    @staticmethod
    def find(
        *, output: str | None = None, provider: str | None = None, release: Sequence[str] | None = None
    ) -> list[ModelDescriptor]:
        releases = release if release is not None else DEFAULT_VISIBLE_RELEASES
        return [
            ModelDescriptor(m)
            for m in active_models()
            if is_visible_for_releases(m, releases)
            and (not output or m.mode == output)
            and (not provider or m.provider == provider)
        ]

    @staticmethod
    def search(query: str, *, release: Sequence[str] | None = None) -> list[ModelDescriptor]:
        releases = release if release is not None else DEFAULT_VISIBLE_RELEASES
        q = query.lower()
        return [
            ModelDescriptor(m)
            for m in active_models()
            if is_visible_for_releases(m, releases)
            and (q in m.id.lower() or q in (m.name or "").lower() or q in (m.provider or "").lower())
        ]


catalog = _Catalog()

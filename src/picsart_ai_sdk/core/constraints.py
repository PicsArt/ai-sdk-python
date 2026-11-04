"""Conditional param constraints — port of ``src/core/constraints.ts``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .js import js_string, js_trim, strict_equal
from .types import Constraint


@dataclass(frozen=True)
class NormalizedRestriction:
    kind: Literal["allowed", "disabled"]
    allowed: list[Any] | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ConditionalRequirement:
    reason: str | None = None


def _in(key: str, obj: object) -> bool:
    """JS ``key in obj``: a TypeError on a non-object, as in TS."""
    if not isinstance(obj, dict):
        raise TypeError(f"Cannot use 'in' operator to search for '{key}' in {js_string(obj)}")
    return key in obj


def _normalize(restriction: dict[str, Any]) -> NormalizedRestriction | None:
    if _in("required", restriction):
        return None
    if "disabled" in restriction:
        return NormalizedRestriction("disabled", reason=restriction.get("reason"))
    return NormalizedRestriction("allowed", allowed=restriction.get("allowed"), reason=restriction.get("reason"))


def _match_operator(op: Any, actual: object) -> bool:
    if _in("exists", op):
        has = (
            actual is not None
            and (not isinstance(actual, list) or len(actual) > 0)
            and (not isinstance(actual, str) or len(js_trim(actual)) > 0)
        )
        return has if op["exists"] else not has
    if "is" in op:
        return strict_equal(actual, op["is"])
    return False


def _match_condition(when: dict[str, Any], values: dict[str, Any]) -> bool:
    return all(_match_operator(op, values.get(key)) for key, op in when.items())


def _merge(prev: NormalizedRestriction | None, nxt: NormalizedRestriction) -> NormalizedRestriction:
    if prev is None:
        return nxt
    if prev.kind == "disabled" or nxt.kind == "disabled":
        reason = nxt.reason if nxt.kind == "disabled" else prev.reason if prev.kind == "disabled" else None
        return NormalizedRestriction("disabled", reason=reason)
    allowed = {js_string(o) for o in nxt.allowed or []}
    return NormalizedRestriction(
        "allowed",
        allowed=[o for o in prev.allowed or [] if js_string(o) in allowed],
        reason=nxt.reason if nxt.reason is not None else prev.reason,
    )


def evaluate_constraints(
    constraints: list[Constraint] | None, values: dict[str, Any]
) -> dict[str, NormalizedRestriction]:
    """Per param, the merged ``allowed`` / ``disabled`` effect of every matching rule."""
    effects: dict[str, NormalizedRestriction] = {}
    for rule in constraints or []:
        if not _match_condition(rule["when"], values):
            continue
        for key, restriction in rule["then"].items():
            normalized = _normalize(restriction)
            if normalized is not None:
                effects[key] = _merge(effects.get(key), normalized)
    return effects


def evaluate_requirements(
    constraints: list[Constraint] | None, values: dict[str, Any]
) -> dict[str, ConditionalRequirement]:
    """Params a matching rule makes required (the first matching rule's reason wins)."""
    requirements: dict[str, ConditionalRequirement] = {}
    for rule in constraints or []:
        if not _match_condition(rule["when"], values):
            continue
        for key, restriction in rule["then"].items():
            if _in("required", restriction) and key not in requirements:
                requirements[key] = ConditionalRequirement(restriction.get("reason"))
    return requirements

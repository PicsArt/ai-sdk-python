#!/usr/bin/env python3
"""Generate ``src/picsart_ai_sdk/model_types.py`` from the models-catalog API (spec §6).

The Python counterpart of ``@picsart/model-types``: one ``TypedDict`` per model built from the same
param descriptors the SDK validates against at runtime, plus ``TypedModelId`` / ``TextModelId`` /
``JsonModelId`` and the ``ModelInputById`` / ``TextModelInputById`` maps.

Usage:
    scripts/generate_model_types.py                  # fetch the prod catalog, write the module
    scripts/generate_model_types.py --check          # exit 1 when the committed module is stale
    scripts/generate_model_types.py --from-file F    # read a saved `/v1/models-catalog` response instead

Auth (fetching): ``PICSART_TOKEN`` (Bearer, with the SDK's gateway headers), the token the e2e suite uses.
``PICSART_API_URL`` overrides the gateway (default ``https://api.picsart.com``).

An unknown descriptor kind or value shape fails the run (exit 2) instead of writing a wrong type.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "picsart_ai_sdk" / "model_types.py"
DEFAULT_API_URL = "https://api.picsart.com"
# Every model, as the TS @picsart/model-types covers: the API's default is only the public
# production / general-availability set, so preview and deprecated models are asked for explicitly.
CATALOG_PATH = (
    "/gw-v2/v1/models-catalog?include=schema&release=production,general-availability,preview&includeDeprecated=true"
)


class SchemaError(Exception):
    """A descriptor the generator does not know how to type."""


# ── naming ───────────────────────────────────────────────────────────


def _camel(text: str) -> str:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", text) if p]
    name = "".join(p[:1].upper() + p[1:] for p in parts)
    return f"M{name}" if not name or name[0].isdigit() else name


class _Names:
    """Unique, deterministic class names (a clash gets a numeric suffix in first-seen order)."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def take(self, wanted: str) -> str:
        name, n = wanted, 2
        while name in self._used:
            name, n = f"{wanted}{n}", n + 1
        self._used.add(name)
        return name


# ── descriptor → type ────────────────────────────────────────────────


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) or isinstance(value, float) and value.is_integer()


def _literal(values: list[Any]) -> str:
    return "Literal[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in values) + "]"


class _Generator:
    def __init__(self) -> None:
        self.names = _Names()
        self.nested: list[str] = []  # nested TypedDict definitions, in emission order

    def type_of(self, descriptor: Mapping[str, Any], owner: str, key: str) -> str:
        kind = descriptor.get("kind")
        if kind in ("text", "catalog"):
            return "str"
        if kind == "boolean":
            return "bool"
        if kind == "unknown":
            return "object"
        if kind == "file":
            return "list[str]" if "array" in descriptor else "str"
        if kind == "range":
            integral = all(_is_int(descriptor.get(k)) for k in ("min", "max")) and _is_int(descriptor.get("step"))
            return "int" if integral else "float"
        if kind == "enum":
            return self._enum(descriptor, owner, key)
        if kind == "object":
            base = self._object(descriptor, owner, key)
            return f"list[{base}]" if "array" in descriptor else base
        raise SchemaError(f"{owner}.{key}: unknown descriptor kind {kind!r}")

    @staticmethod
    def _enum(descriptor: Mapping[str, Any], owner: str, key: str) -> str:
        options = descriptor.get("options")
        if not isinstance(options, list) or not options or not all(isinstance(o, dict) and "id" in o for o in options):
            raise SchemaError(f"{owner}.{key}: enum without options")
        ids = list(dict.fromkeys(o["id"] for o in options))
        if all(isinstance(i, str) for i in ids):
            return _literal(ids)
        if all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
            return _literal(ids)
        if all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in ids):
            return "float"  # Literal cannot hold non-integral numbers
        raise SchemaError(f"{owner}.{key}: enum ids of mixed or unsupported types {ids!r}")

    def _object(self, descriptor: Mapping[str, Any], owner: str, key: str) -> str:
        fields = descriptor.get("fields")
        extra = descriptor.get("additionalProperties")
        if fields and not extra:
            return self.typed_dict(
                self.names.take(f"{owner.removesuffix('Input')}{_camel(key)}"), fields, f"{owner}.{key}"
            )
        if extra and not fields:
            return f"dict[str, {self.type_of(extra, owner, key + 'Value')}]"
        return "dict[str, Any]"  # both or neither: a TypedDict cannot express the open keys

    def typed_dict(self, name: str, params: Mapping[str, Any], where: str) -> str:
        entries = []
        for key, descriptor in params.items():
            if not isinstance(descriptor, dict):
                raise SchemaError(f"{where}.{key}: descriptor is not an object")
            annotation = self.type_of(descriptor, name, key)
            if not descriptor.get("required"):
                annotation = f"NotRequired[{annotation}]"
            entries.append(f"        {json.dumps(key)}: {annotation},")
        body = "\n".join(entries)
        self.nested.append(
            f'{name} = TypedDict(\n    "{name}",\n    {{\n{body}\n    }},\n)'
            if entries
            else f'{name} = TypedDict("{name}", {{}})'
        )
        return name


# ── module ───────────────────────────────────────────────────────────

HEADER = '''"""Typed model inputs — GENERATED by scripts/generate_model_types.py from the models-catalog API.

DO NOT EDIT. One ``TypedDict`` per model (the Python counterpart of ``@picsart/model-types``)::

    from picsart_ai_sdk.model_types import Flux2ProInput

    params: Flux2ProInput = {"prompt": "a cat", "aspectRatio": "16:9"}
    await ai.generate("flux-2-pro", params)
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import NotRequired, TypedDict

__all__ = ["JsonModelId", "ModelInputById", "TextModelId", "TextModelInputById", "TypedModelId"]
'''


def render(body: Any) -> str:
    """The module for a ``/v1/models-catalog`` response body."""
    response = body.get("response") if isinstance(body, dict) else None
    models = response.get("models") if isinstance(response, dict) else None
    if not isinstance(models, list) or not models:
        raise SchemaError('expected { "status": "success", "response": { "models": [...] } } with models')
    gen = _Generator()
    by_id: dict[str, str] = {}
    text_ids: list[str] = []
    json_ids: list[str] = []
    for model in sorted(models, key=lambda m: m["id"]):
        model_id = model["id"]
        params = model.get("params")
        if params is None:
            raise SchemaError(f"{model_id}: no params — request include=schema")
        by_id[model_id] = gen.typed_dict(gen.names.take(f"{_camel(model_id)}Input"), params, model_id)
        if model.get("mode") == "text":
            text_ids.append(model_id)
        elif model.get("mode") == "json":
            json_ids.append(model_id)

    def literal_alias(name: str, ids: list[str]) -> str:
        if not ids:
            return f"{name} = str  # no such models in the catalog"
        return f"{name} = Literal[\n" + "".join(f"    {json.dumps(i)},\n" for i in ids) + "]"

    def mapping(name: str, ids: list[str]) -> str:
        rows = "".join(f"        {json.dumps(i)}: {by_id[i]},\n" for i in ids)
        return f'{name} = TypedDict(\n    "{name}",\n    {{\n{rows}    }},\n)'

    exported = sorted(set(by_id.values()))
    return (
        HEADER.replace(
            '__all__ = ["JsonModelId", "ModelInputById", "TextModelId", "TextModelInputById", "TypedModelId"]',
            "__all__ = [\n"
            + "".join(
                f'    "{n}",\n'
                for n in sorted(
                    [*exported, "JsonModelId", "ModelInputById", "TextModelId", "TextModelInputById", "TypedModelId"]
                )
            )
            + "]",
        )
        + "\n\n"
        + "\n\n\n".join(gen.nested)
        + "\n\n\n"
        + literal_alias("TypedModelId", list(by_id))
        + '\n"""Every catalog model id."""\n\n'
        + literal_alias("TextModelId", text_ids)
        + '\n"""Text-generation (LLM) models — ``generate_text()``."""\n\n'
        + literal_alias("JsonModelId", json_ids)
        + '\n"""Structured-output (json-mode) models — ``run()``."""\n\n'
        + mapping("ModelInputById", list(by_id))
        + '\n"""Model id → its input ``TypedDict`` (read the types off ``__annotations__``)."""\n\n'
        + mapping("TextModelInputById", text_ids)
        + "\n"
    )


def fetch_catalog(api_url: str) -> Any:
    token = re.sub(r"^Bearer\s+", "", os.environ.get("PICSART_TOKEN", ""), flags=re.IGNORECASE)
    if not token:
        raise SystemExit("PICSART_TOKEN must be set to fetch the catalog (or pass --from-file)")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "platform": "api",
        "X-Touchpoint": "sdk",
    }
    request = urllib.request.Request(api_url.rstrip("/") + CATALOG_PATH, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as res:  # noqa: S310 — fixed https gateway URL
        return json.loads(res.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed module is stale")
    parser.add_argument("--from-file", type=Path, help="a saved /v1/models-catalog response")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    body = (
        json.loads(args.from_file.read_text("utf-8"))
        if args.from_file
        else fetch_catalog(os.environ.get("PICSART_API_URL", DEFAULT_API_URL))
    )
    try:
        module = render(body)
    except SchemaError as err:
        print(f"generate_model_types: {err}", file=sys.stderr)
        return 2
    current = args.output.read_text("utf-8") if args.output.exists() else None
    if args.check:
        if current != module:
            print(f"{args.output} is stale — run scripts/generate_model_types.py", file=sys.stderr)
            return 1
        return 0
    if current != module:
        args.output.write_text(module, "utf-8")
        print(f"generate_model_types: wrote {args.output}")
    else:
        print("generate_model_types: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

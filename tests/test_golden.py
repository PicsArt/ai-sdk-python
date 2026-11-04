"""Golden fixtures recorded from the TypeScript SDK (spec §8) — every case must come out the same here.

``tests/golden/ts-sdk.json`` is written by ``scripts/golden/generate_golden.mts`` (run it on a TS version
bump: the JSON diff is the behavior change to port). Results are compared as JSON: Python dataclasses are
turned back into the TS shape (camelCase keys, ``None`` fields left out, as ``JSON.stringify`` drops
``undefined``). A TS ``TypeError`` (a JS property read on ``undefined``) must be a Python ``TypeError``;
an ``ApiError`` must match in message, status and code.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from picsart_ai_sdk.client.drive import parse_generation
from picsart_ai_sdk.client.drive_naming import (
    AppIdentity,
    GenerationProvenance,
    build_filename,
    build_generation_attributes,
    expected_output_format,
    resolve_extension,
    to_sdk_payload,
)
from picsart_ai_sdk.client.prepare import (
    build_models_command,
    parse_dynamic_result,
    parse_dynamic_run_result,
    parse_dynamic_text_result,
)
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, install_catalog
from picsart_ai_sdk.core.catalog_wire import deserialize_catalog, serialize_catalog
from picsart_ai_sdk.core.deeplink import decode_deep_link_payload, encode_deep_link_payload
from picsart_ai_sdk.core.deeplink.codec import decode
from picsart_ai_sdk.core.descriptors.model_accessor import Model
from picsart_ai_sdk.core.errors import ApiError
from picsart_ai_sdk.core.model_registry import get_model
from picsart_ai_sdk.core.workflow import WorkflowJobHandle, WorkflowStatusResult

GOLDEN = json.loads((Path(__file__).parent / "golden" / "ts-sdk.json").read_text("utf-8"))
CASES: list[dict[str, Any]] = GOLDEN["cases"]
KEY_NAMES = {"ai_sdk_payload": "aiSDKPayload"}


def _camel(name: str) -> str:
    if name in KEY_NAMES:
        return KEY_NAMES[name]
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def as_ts(value: Any) -> Any:
    """A Python result in the TS JSON shape."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            v = getattr(value, f.name)
            if f.name == "extra":  # catch-all for fields the dataclass does not name
                out.update(as_ts(v))
            elif v is not None or (f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING):
                out[_camel(f.name)] = as_ts(v)  # a required field is always present (TS `null`)
        return out
    if isinstance(value, dict):
        return {k: as_ts(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_ts(v) for v in value]
    return value


def json_of(value: Any) -> Any:
    """What ``JSON.parse(JSON.stringify(value))`` gives for the Python result."""
    return json.loads(json.dumps(as_ts(value), allow_nan=False))


def _typed(value: Any) -> Any:
    """``value`` with every scalar tagged by whether it is a boolean, so ``True`` no longer equals ``1``."""
    if isinstance(value, dict):
        return {key: _typed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_typed(item) for item in value]
    return (isinstance(value, bool), value)


def same_json(actual: Any, expected: Any) -> bool:
    """JSON equality: ``1 == 1.0`` (one JSON number), but ``true`` is not ``1``."""
    return bool(_typed(actual) == _typed(expected))


@pytest.fixture(autouse=True, scope="module")
def catalog() -> Any:
    install_catalog(deserialize_catalog(GOLDEN["catalog"]))
    yield
    _reset_active_catalog_for_tests()


def _status(raw: dict[str, Any]) -> WorkflowStatusResult:
    handle = WorkflowJobHandle(raw["handle"]["workflow"], raw["handle"]["id"])
    return WorkflowStatusResult(
        handle=handle, status=raw["status"], result=raw.get("result"), usage=raw.get("usage"), raw=raw.get("raw")
    )


def _command(model: str, params: dict[str, Any], opts: dict[str, Any] | None) -> Any:
    if opts is None:
        return build_models_command(model, params)
    return build_models_command(
        model,
        params,
        with_options=True,
        drive=opts.get("drive"),
        downscale_oversized_images=opts.get("downscaleOversizedImages"),
    )


def _attributes(provenance: dict[str, Any]) -> Any:
    app = provenance.get("app")
    return build_generation_attributes(
        GenerationProvenance(
            model_id=provenance["modelId"],
            params=provenance["params"],
            app=AppIdentity(id=app["id"], type=app["type"]) if app else None,
        )
    )


def _filename(prompt: str | None, mode: str, hints: dict[str, str]) -> str:
    return re.sub(r"-\d+(?=(\.[^.]*)?$)", "-<id>", build_filename(prompt, mode, hints))  # type: ignore[arg-type]


CALLS: dict[str, Any] = {
    "validate": lambda m, values: Model(m).validate(values),
    "params.all": lambda m: Model(m).params().all(),
    "params.getDefaults": lambda m: Model(m).params().get_defaults(),
    "params.toSchema": lambda m: Model(m).params().to_schema(),
    "params.transferValues": lambda m, prev: Model(m).params().transfer_values(prev),
    "meta": lambda m: Model(m).meta(),
    "paramsFor": lambda m, values: Model(m).params_for(values).all(),
    "buildModelsCommand": _command,
    "parseDynamicResult": lambda s, ref: parse_dynamic_result(_status(s), ref),
    "parseDynamicTextResult": lambda s, ref: parse_dynamic_text_result(_status(s), ref),
    "parseDynamicRunResult": lambda s, ref: parse_dynamic_run_result(_status(s), ref),
    "toSdkPayload": to_sdk_payload,
    "buildGenerationAttributes": _attributes,
    "resolveExtension": resolve_extension,
    "expectedOutputFormat": lambda m, params: expected_output_format(get_model(m), params),  # type: ignore[arg-type]
    "buildFilename": _filename,
    "parseGeneration": parse_generation,
    "decodeDeepLink": decode_deep_link_payload,
}


def _id(case: dict[str, Any]) -> str:
    return f"{case['fn']}:{json.dumps(case['args'], ensure_ascii=False)[:80]}"


@pytest.mark.parametrize("case", [c for c in CASES if c["fn"] in CALLS], ids=_id)
def test_case(case: dict[str, Any]) -> None:
    call = CALLS[case["fn"]]
    expected_error = case.get("error")
    if expected_error is None:
        actual = json_of(call(*case["args"]))
        assert same_json(actual, case["output"]), (actual, case["output"])
        return
    error_type = (
        TypeError
        if expected_error["name"] == "TypeError"
        else ApiError
        if expected_error["name"] == "ApiError"
        else Exception
    )
    with pytest.raises(error_type) as caught:
        call(*case["args"])
    if isinstance(caught.value, ApiError):
        assert (str(caught.value), caught.value.status, caught.value.code) == (
            expected_error["message"],
            expected_error["status"],
            expected_error["code"],
        )
    elif error_type is Exception:
        assert str(caught.value) == expected_error["message"]


@pytest.mark.parametrize("case", [c for c in CASES if c["fn"] == "encodeDeepLink"], ids=_id)
def test_deep_link_payloads_match(case: dict[str, Any]) -> None:
    # The DEFLATE bytes may differ (spec §5.9); the payload inside must not.
    assert same_json(decode(encode_deep_link_payload(*case["args"])), decode(case["output"]))


def test_catalog_wire_round_trip() -> None:
    assert same_json(json_of(serialize_catalog(deserialize_catalog(GOLDEN["catalog"]))), GOLDEN["catalog"])


def test_every_recorded_function_is_replayed() -> None:
    recorded = Counter(c["fn"] for c in CASES)
    assert set(recorded) <= {*CALLS, "encodeDeepLink"}, set(recorded) - {*CALLS, "encodeDeepLink"}
    assert sum(recorded.values()) > 800


def test_json_comparison_tells_booleans_from_numbers() -> None:
    # Python's `True == 1` must not let a wrong JSON type pass a golden case; `1 == 1.0` is one JSON number.
    assert not same_json({"valid": 1}, {"valid": True})
    assert not same_json([0], [False])
    assert same_json({"a": [1, {"b": True}]}, {"a": [1.0, {"b": True}]})
    assert not same_json({"a": 1}, {"a": 1, "b": 2})

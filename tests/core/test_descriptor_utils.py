"""Ports of ``transfer-values.test.ts``, ``object-additional-properties.test.ts`` and the descriptor part of
``validate.test.ts``, plus constraint semantics."""

from typing import Any

import pytest

from picsart_ai_sdk.core.constraints import evaluate_constraints, evaluate_requirements
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from picsart_ai_sdk.core.descriptors.utils import (
    DescriptorError,
    descriptors_to_schema,
    extract_defaults,
    transfer_values,
    validate_all,
    validate_descriptor,
)


def array_file(maximum: int) -> dict[str, ParamEntry]:
    return {"imageUrls": ParamEntry({"kind": "file", "accept": "image", "array": {"max": maximum}})}


def single_file() -> dict[str, ParamEntry]:
    return {"imageUrls": ParamEntry({"kind": "file", "accept": "image"})}


PARAMS = {
    "caption": ParamEntry({"kind": "text", "maxLength": 10}),
    "free": ParamEntry({"kind": "text"}),
    "aspectRatio": ParamEntry(
        {"kind": "enum", "valueType": "string", "options": [{"id": "1:1"}, {"id": "16:9"}], "default": "1:1"}
    ),
    "steps": ParamEntry({"kind": "range", "min": 10, "max": 50, "default": 20}),
    "enhance": ParamEntry({"kind": "boolean", "default": False}),
    "voiceId": ParamEntry({"kind": "catalog", "source": {"workflow": "w"}, "default": "v0"}),
}


@pytest.mark.parametrize(
    ("params", "prev", "expected"),
    [
        (array_file(1), ["a", "b", "c", "d"], ["a"]),
        (array_file(3), ["a", "b", "c", "d"], ["a", "b", "c"]),
        (array_file(4), ["a", "b"], ["a", "b"]),
        (array_file(4), ["a", "", "b", ""], ["a", "b"]),
        (array_file(4), "solo.png", ["solo.png"]),
        (single_file(), ["first.png", "second.png"], "first.png"),
    ],
)
def test_file_shape_coercion(params: Any, prev: Any, expected: Any) -> None:
    assert transfer_values(params, {"imageUrls": prev})["imageUrls"] == expected


def test_empty_file_values_unset() -> None:
    assert "imageUrls" not in transfer_values(array_file(4), {"imageUrls": ["", ""]})
    assert "imageUrls" not in transfer_values(single_file(), {"imageUrls": ""})
    assert "imageUrls" not in transfer_values(single_file(), {"imageUrls": []})


def test_transfer_values_kinds() -> None:
    assert transfer_values(PARAMS, {"caption": "0123456789ABCDEF"})["caption"] == "0123456789"
    assert transfer_values(PARAMS, {"free": "unbounded value here"})["free"] == "unbounded value here"
    assert transfer_values(PARAMS, {"aspectRatio": "__nope__"})["aspectRatio"] == "1:1"
    assert transfer_values(PARAMS, {"aspectRatio": "16:9"})["aspectRatio"] == "16:9"
    assert transfer_values(PARAMS, {"steps": 9999})["steps"] == 50
    assert transfer_values(PARAMS, {"steps": 0})["steps"] == 10
    assert transfer_values(PARAMS, {"enhance": True})["enhance"] is True
    assert transfer_values(PARAMS, {"voiceId": "picked"})["voiceId"] == "v0"  # catalog resets to default
    out = transfer_values(PARAMS, {"caption": "hi", "removedField": "leftover"})
    assert out["caption"] == "hi" and "removedField" not in out


def test_extract_defaults_and_schema() -> None:
    assert extract_defaults(PARAMS) == {"aspectRatio": "1:1", "steps": 20, "enhance": False, "voiceId": "v0"}
    schema = descriptors_to_schema(
        {**PARAMS, "img": ParamEntry({"kind": "file", "accept": "image"}, label="Image", required=True)}
    )
    assert schema["aspectRatio"] == {"type": "string", "enum": ["1:1", "16:9"], "default": "1:1"}
    assert schema["steps"] == {
        "type": "number",
        "min": 10,
        "max": 50,
        "default": 20,
    }  # TS `step: undefined` is not written
    assert schema["img"] == {"type": "file", "required": True, "label": "Image", "accept": "image"}


@pytest.mark.parametrize(
    ("d", "val", "required", "message"),
    [
        ({"kind": "text"}, None, True, '"k" is required'),
        ({"kind": "text"}, "   ", True, '"k" is required'),
        ({"kind": "file", "accept": "image", "array": {}}, [], True, '"k" is required'),
        (
            {"kind": "enum", "valueType": "string", "options": [{"id": "a"}, {"id": "b"}]},
            "c",
            False,
            '"k" must be one of: a, b',
        ),
        (
            {"kind": "enum", "valueType": "number", "options": [{"id": 0}, {"id": 1}]},
            False,
            False,
            '"k" must be one of: 0, 1',
        ),
        ({"kind": "enum", "valueType": "number", "options": [{"id": 1}]}, True, False, '"k" must be one of: 1'),
        ({"kind": "catalog", "source": {}}, 5, False, '"k" must be a string'),
        ({"kind": "range", "min": 1, "max": 5}, "3", False, '"k" must be a number'),
        ({"kind": "range", "min": 1, "max": 5}, True, False, '"k" must be a number'),
        ({"kind": "range", "min": 1, "max": 5.5}, 9, False, '"k" must be between 1 and 5.5'),
        ({"kind": "boolean"}, 1, False, '"k" must be a boolean'),
        ({"kind": "text", "minLength": 3}, " ab ", False, '"k" must be at least 3 characters'),
        ({"kind": "text", "maxLength": 3}, "😀😀", False, '"k" exceeds max length of 3'),  # 4 UTF-16 units
        ({"kind": "file", "accept": "image", "array": {"min": 2}}, ["a"], False, '"k" needs at least 2 items'),
        ({"kind": "file", "accept": "image", "array": {"max": 1}}, ["a", "b"], False, '"k" allows at most 1 items'),
        ({"kind": "file", "accept": "image", "array": {}}, "a", False, '"k" must be an array of URLs'),
        ({"kind": "file", "accept": "image"}, ["a"], False, '"k" must be a string URL'),
    ],
)
def test_validate_descriptor_messages(d: dict[str, Any], val: Any, required: bool, message: str) -> None:
    with pytest.raises(DescriptorError) as exc:
        validate_descriptor("k", d, val, required)
    assert str(exc.value) == message


def test_valid_values_pass() -> None:
    validate_descriptor("k", {"kind": "enum", "valueType": "number", "options": [{"id": 0}, {"id": 1}]}, 1.0, False)
    validate_descriptor("k", {"kind": "text", "maxLength": 4}, "😀😀", False)
    validate_descriptor("k", {"kind": "unknown"}, object(), True)
    validate_descriptor("k", {"kind": "text"}, None, False)
    validate_descriptor("k", {"kind": "text"}, "", False)


QUESTIONS = {
    "questions": ParamEntry(
        required=True,
        descriptor={
            "kind": "object",
            "additionalProperties": {
                "kind": "object",
                "fields": {
                    "type": {
                        "kind": "enum",
                        "valueType": "string",
                        "options": [{"id": "noul"}, {"id": "choice"}, {"id": "score"}],
                        "default": "noul",
                        "required": True,
                    },
                    "instructions": {"kind": "unknown", "required": True},
                    "reasoning": {"kind": "boolean", "default": False, "required": False},
                },
                "additionalProperties": {"kind": "unknown"},
            },
        },
    )
}


def test_object_additional_properties() -> None:
    validate_all(QUESTIONS, {"questions": {"q1": {"type": "choice", "instructions": {"x": 1}, "extra": 3}}})
    cases: list[tuple[dict[str, Any], str]] = [
        (
            {"questions": {"q1": {"type": "nope", "instructions": 1}}},
            '"questions.q1.type" must be one of: noul, choice, score',
        ),
        ({"questions": {"q1": {"type": "choice"}}}, '"questions.q1.instructions" is required'),
        ({"questions": {"q1": "flat"}}, '"questions.q1" must be an object'),
        ({"questions": []}, '"questions" is required'),
        ({}, '"questions" is required'),
        ({"questions": {}}, '"questions" is required'),
    ]
    for bad, message in cases:
        with pytest.raises(DescriptorError) as exc:
            validate_all(QUESTIONS, bad)
        assert str(exc.value) == message
    tags = {"tags": ParamEntry({"kind": "object", "additionalProperties": {"kind": "text"}})}
    validate_all(tags, {"tags": {"a": "x"}})
    with pytest.raises(DescriptorError, match='"tags.a" must be a string'):
        validate_all(tags, {"tags": {"a": 1}})
    closed = {"item": ParamEntry({"kind": "object", "fields": {"id": {"kind": "text", "required": True}}})}
    validate_all(closed, {"item": {"id": "1", "free": "extras pass without additionalProperties"}})
    with pytest.raises(DescriptorError, match='"item" must be an array'):
        validate_all({"item": ParamEntry({"kind": "object", "array": {}})}, {"item": {}})


def test_constraints() -> None:
    rules = [
        {
            "when": {"resolution": {"is": "1080p"}},
            "then": {"duration": {"allowed": [5, 8, 10]}, "fps": {"disabled": True, "reason": "fixed"}},
        },
        {"when": {"resolution": {"is": "1080p"}}, "then": {"duration": {"allowed": ["8", 10], "reason": "1080p"}}},
        {"when": {"image": {"exists": True}}, "then": {"prompt": {"required": True, "reason": "describe the image"}}},
        {"when": {"flag": {"is": 1}}, "then": {"x": {"disabled": True}}},
    ]
    effects = evaluate_constraints(rules, {"resolution": "1080p", "flag": True})
    assert effects["duration"].allowed == [8, 10] and effects["duration"].reason == "1080p"
    assert (effects["fps"].kind, effects["fps"].reason) == ("disabled", "fixed")
    assert "x" not in effects  # True !== 1
    assert evaluate_requirements(rules, {"image": "   "}) == {}
    assert evaluate_requirements(rules, {"image": "u"})["prompt"].reason == "describe the image"
    with pytest.raises(DescriptorError, match='"prompt" is required: describe the image'):
        validate_all({"prompt": ParamEntry({"kind": "text"})}, {"image": "u"}, rules)

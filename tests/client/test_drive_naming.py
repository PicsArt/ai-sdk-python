import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from picsart_ai_sdk.client.drive_naming import (
    AppIdentity,
    GenerationProvenance,
    build_filename,
    build_generation_attributes,
    expected_output_format,
    infer_resource_type,
    resolve_extension,
    to_sdk_payload,
)
from picsart_ai_sdk.core.js import js_json_dumps, utf16_len


def ext(name: str) -> str:
    return name[name.rfind(".") + 1 :]


@dataclass
class Entry:
    descriptor: dict[str, Any]


@dataclass
class Model:
    param_config: dict[str, Any] = field(default_factory=dict)
    output_extension: str | None = None


def test_mode_defaults() -> None:
    assert [resolve_extension(m) for m in ("image", "video", "audio", "unknown")] == ["png", "mp4", "mp3", ""]
    assert ext(build_filename("a cat", "image")) == "png"
    assert ext(build_filename(None, "video")) == "mp4"


@pytest.mark.parametrize(
    ("mode", "mime", "expected"),
    [
        ("image", "image/jpeg", "jpg"),
        ("image", "image/webp", "webp"),
        ("image", "IMAGE/JPEG; charset=binary", "jpg"),
        ("video", "video/quicktime", "mov"),
        ("video", "video/webm", "webm"),
        ("audio", "audio/x-wav", "wav"),
        ("audio", "audio/mpeg", "mp3"),
    ],
)
def test_mime(mode: str, mime: str, expected: str) -> None:
    assert resolve_extension(mode, {"mimeType": mime}) == expected


def test_priority_and_urls() -> None:
    assert (
        resolve_extension("image", {"mimeType": "image/jpeg", "url": "https://cdn.test/x.png", "format": "webp"})
        == "jpg"
    )
    assert (
        resolve_extension("image", {"mimeType": "application/octet-stream", "url": "https://cdn.test/x.webp"}) == "webp"
    )
    assert resolve_extension("image", {"url": "https://cdn.picsart.test/b0f5897a-1c2d.jpg?sig=abc&exp=1"}) == "jpg"
    assert resolve_extension("image", {"url": "https://cdn.test/out.JPEG#frag"}) == "jpg"
    assert resolve_extension("video", {"url": "https://cdn.test/a/b/clip.mov"}) == "mov"
    assert resolve_extension("image", {"url": "https://cdn.test/files/abc123", "format": "webp"}) == "webp"
    assert resolve_extension("image", {"url": "https://cdn.test/v1.2/abc"}) == "png"
    assert resolve_extension("image", {"url": "not a url/pic.webp?x=1"}) == "webp"


@pytest.mark.parametrize(
    ("mode", "fmt", "expected"),
    [
        ("image", "jpeg", "jpg"),
        ("image", "PNG", "png"),
        ("image", "webp", "webp"),
        ("video", "mov", "mov"),
        ("video", "mp4_8bit", "mp4"),
        ("audio", "wav", "wav"),
        ("audio", "ogg_opus", "ogg"),
        ("image", "../etc", "png"),
    ],
)
def test_format(mode: str, fmt: str, expected: str) -> None:
    assert resolve_extension(mode, {"format": fmt}) == expected


def test_expected_output_format_order() -> None:
    gpt = Model(param_config={"outputFormat": Entry({"kind": "enum", "default": "png"})})
    assert expected_output_format(gpt, {"prompt": "x", "outputFormat": "jpeg"}) == "jpeg"
    assert expected_output_format(gpt, {"prompt": "x"}) == "png"
    assert expected_output_format(Model(), {"prompt": "x", "output_format": "webp"}) == "webp"
    assert expected_output_format(Model(), {"prompt": "x"}) is None
    assert expected_output_format(Model(output_extension="jpg"), {"prompt": "x"}) == "jpg"


def test_filename_slug() -> None:
    assert re.fullmatch(
        r"a-red-paper-boat-on-a-calm-blue-lake-at-sunrise-\d{6}\.jpg",
        build_filename("A red paper boat on a calm blue lake at sunrise!", "image", {"url": "https://c/x.jpg"}),
    )
    assert re.fullmatch(r"a-dog-\d+", build_filename("a dog", "unknown"))
    assert re.fullmatch(r"ai-generation-\d+", build_filename(None, "unknown"))
    assert re.fullmatch(r"a-dog-\d+\.mp4", build_filename("a dog", "unknown", {"format": "mp4"}))
    # slice after trimming: a trailing "-" at char 50 survives, as in TS
    assert re.fullmatch("x" * 49 + r"--\d{6}", build_filename("x" * 49 + " y", "unknown"))


def test_infer_resource_type() -> None:
    assert [infer_resource_type(m) for m in ("video", "audio", "image", "text")] == ["VIDEO", "AUDIO", "PHOTO", "PHOTO"]


def test_sdk_payload_prompt_first_and_empties_dropped() -> None:
    payload = to_sdk_payload({"aspectRatio": "16:9", "prompt": "p", "seed": None, "neg": "", "n": 0, "x": False})
    assert list(payload) == ["prompt", "aspectRatio", "n", "x"]


def test_generation_attributes() -> None:
    attrs = build_generation_attributes(
        GenerationProvenance("flux", {"prompt": "a cat", "duration": 10.0}, AppIdentity("com.app", "miniapp"))
    )
    assert attrs == {
        "model": "flux",
        "aiSDKPayload": '{"prompt":"a cat","duration":10}',
        "appId": "com.app",
        "appType": "miniapp",
    }
    assert build_generation_attributes(GenerationProvenance("m", {})) == {"model": "m", "aiSDKPayload": '{"prompt":""}'}


def test_payload_cut_to_20000_utf16_units_without_splitting_surrogates() -> None:
    attrs = build_generation_attributes(GenerationProvenance("m", {"prompt": "😀" * 15000, "seed": 1}))
    serialized = attrs["aiSDKPayload"]
    assert utf16_len(serialized) <= 20000
    prompt = json.loads(serialized)["prompt"]
    assert set(prompt) == {"😀"}  # no lone surrogate
    assert utf16_len(to_sdk_payload({"prompt": "a" * 18001})["prompt"]) == 18000


def test_attribute_cap_trims_the_prompt_further_when_other_params_are_long() -> None:
    # The 18 000-unit prompt cut alone leaves this payload over 20 000: only the attribute cap brings it under.
    params = {"prompt": "😀" * 15000, "negativePrompt": "n" * 5000}
    assert utf16_len(js_json_dumps(to_sdk_payload(params))) > 20000
    serialized = build_generation_attributes(GenerationProvenance("m", params))["aiSDKPayload"]
    assert 19990 <= utf16_len(serialized) <= 20000
    payload = json.loads(serialized)
    assert payload["negativePrompt"] == "n" * 5000
    assert utf16_len(payload["prompt"]) < 18000
    assert set(payload["prompt"]) == {"😀"}  # no lone surrogate


def test_js_json_dumps() -> None:
    assert js_json_dumps({"a": 1.0, "b": [2.5, None, True], "c": "é"}) == '{"a":1,"b":[2.5,null,true],"c":"é"}'
    assert js_json_dumps({"n": float("nan")}) == '{"n":null}'

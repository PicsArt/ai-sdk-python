"""Port of ``__tests__/unit/deeplink.test.ts``, plus JS-semantics and cross-SDK cases."""

import base64
import json
import zlib
from typing import Any

import pytest

from picsart_ai_sdk import decode_deep_link_payload, encode_deep_link_payload
from picsart_ai_sdk.core.active_catalog import _reset_active_catalog_for_tests, install_catalog
from picsart_ai_sdk.core.deeplink.codec import decode, encode
from picsart_ai_sdk.core.deeplink.sanitize import is_allowed_url, sanitize_prompt
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from tests.fixtures.catalog import make_model

CDN = "https://cdn-cms-uploads.picsart.com/cms-uploads"
TEXT = ParamEntry({"kind": "text"}, label="Prompt")


@pytest.fixture(autouse=True)
def catalog() -> Any:
    install_catalog(
        [
            make_model(
                id="gemini-3-pro-image",
                name="Gemini 3 Pro Image",
                mode="image",
                param_config={"prompt": TEXT, "negativePrompt": ParamEntry({"kind": "text"}, label="Negative")},
            ),
            make_model(
                id="veo-3.1",
                name="Veo 3.1",
                mode="video",
                param_config={
                    "prompt": TEXT,
                    "aspectRatio": ParamEntry(
                        {
                            "kind": "enum",
                            "valueType": "string",
                            "options": [{"id": "16:9"}, {"id": "9:16"}],
                            "default": "16:9",
                        },
                        label="AR",
                    ),
                },
            ),
            make_model(
                id="kling-v3-pro",
                name="Kling v3 Pro",
                mode="video",
                param_config={
                    "prompt": TEXT,
                    "duration": ParamEntry({"kind": "range", "min": 3, "max": 15, "default": 5}, label="Duration"),
                },
            ),
            make_model(id="kling-avatar", name="Kling Avatar", mode="video", param_config={"prompt": TEXT}),
        ]
    )
    yield
    _reset_active_catalog_for_tests()


def raw(encoded: str) -> Any:
    """The payload object inside an encoded link."""
    padded = encoded.replace("-", "+").replace("_", "/") + "=" * (-len(encoded) % 4)
    return json.loads(zlib.decompress(base64.b64decode(padded), -15))


def make_link(payload: Any) -> str:
    """A link carrying exactly this JSON text (bypasses the serializer)."""
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.urlsafe_b64encode(c.compress(data.encode()) + c.flush()).decode().rstrip("=")


# ── 1. sanitize_prompt ─────────────────────────────────────────────────
def test_sanitize_prompt() -> None:
    assert sanitize_prompt("Normal prompt") == "Normal prompt"
    assert sanitize_prompt('Hello <script>alert("xss")</script> world') == 'Hello alert("xss") world'
    assert sanitize_prompt("<b>bold</b> text") == "bold text"
    assert sanitize_prompt('<div class="x">content</div>') == "content"
    assert sanitize_prompt("javascript:void(0)") == "void(0)"
    assert sanitize_prompt("onclick=steal()") == "steal()"


def test_sanitize_prompt_js_regex_semantics() -> None:
    assert sanitize_prompt("a <b\nc") == "a "  # `[^>]+(>|$)`: an unclosed tag runs to the end, across lines
    assert sanitize_prompt("x <b\n") == "x "  # JS `$` is end of input, not "before a final newline"
    assert sanitize_prompt("JavaScript ﻿:go") == "go"  # JS `\s` includes NBSP and BOM
    assert sanitize_prompt("javascript\x1c:go") == "javascript\x1c:go"  # …but not the separators Python adds
    assert sanitize_prompt("javaſcript:go") == "javaſcript:go"  # /i folds ASCII only (no ſ → s)
    assert sanitize_prompt("onclick = x") == " x"
    assert sanitize_prompt("oné= x") == "oné= x"  # JS `\w` is ASCII


# ── 2. is_allowed_url ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "url",
    [
        "https://cdn.picsart.com/image.jpg",
        "https://images.picsart.com/photo.png",
        "https://cdn-picsart.com/asset.webp",
        "https://static.pastatic.com/file.jpg",
        f"{CDN}/abc123.png",
        "HTTPS://CDN.PICSART.COM/x.jpg",
        "https://picsart.com",
    ],
)
def test_allowed_urls(url: str) -> None:
    assert is_allowed_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.picsart.com/image.jpg",
        "data:image/png;base64,abc",
        "blob:http://example.com/uuid",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "https://evil.com/image.jpg",
        "not a url",
        "https://picsart.com.evil.com/x",
        "https://evilpicsart.com/x",
        "https://picsart.com\n",  # JS `$` does not match before a trailing newline
        " https://picsart.com/x",
        "https://user@picsart.com/x",
        "https://Kpicsart.com/x",  # Kelvin sign: ASCII-only case folding
        42,
    ],
)
def test_rejected_urls(url: Any) -> None:
    assert is_allowed_url(url) is False


# ── 3. encode + decode ─────────────────────────────────────────────────
def test_basic_prompt_round_trip() -> None:
    encoded = encode_deep_link_payload("gemini-3-pro-image", {"prompt": "A sunset over mountains"})
    assert isinstance(encoded, str) and encoded and "=" not in encoded
    result = decode_deep_link_payload(encoded)
    assert result is not None
    assert (result.model_id, result.context, result.model_known, result.warnings) == (
        "gemini-3-pro-image",
        {"prompt": "A sunset over mountains"},
        True,
        [],
    )


def test_dedicated_fields_survive() -> None:
    r = decode_deep_link_payload(
        encode_deep_link_payload("gemini-3-pro-image", {"prompt": "test", "negativePrompt": "blurry, low quality"})
    )
    assert r is not None and r.context["negativePrompt"] == "blurry, low quality"
    urls = [f"{CDN}/img1.jpg", f"{CDN}/img2.png"]
    r = decode_deep_link_payload(encode_deep_link_payload("gemini-3-pro-image", {"prompt": "edit", "imageUrls": urls}))
    assert r is not None and r.context["imageUrls"] == urls
    r = decode_deep_link_payload(
        encode_deep_link_payload("kling-v3-pro", {"prompt": "edit video", "videoUrl": f"{CDN}/video.mp4"})
    )
    assert r is not None and r.context["videoUrl"] == f"{CDN}/video.mp4"
    r = decode_deep_link_payload(
        encode_deep_link_payload("kling-avatar", {"prompt": "avatar", "audioUrl": f"{CDN}/audio.mp3"})
    )
    assert r is not None and r.context["audioUrl"] == f"{CDN}/audio.mp3"


def test_non_default_params_survive() -> None:
    r = decode_deep_link_payload(encode_deep_link_payload("veo-3.1", {"prompt": "test", "aspectRatio": "9:16"}))
    assert r is not None and r.context["aspectRatio"] == "9:16"
    r = decode_deep_link_payload(encode_deep_link_payload("kling-v3-pro", {"prompt": "test", "duration": 10}))
    assert r is not None and r.context["duration"] == 10


def test_empty_context_is_a_minimal_payload() -> None:
    encoded = encode_deep_link_payload("gemini-3-pro-image", {})
    assert raw(encoded) == {"v": 1, "m": "gemini-3-pro-image"}
    r = decode_deep_link_payload(encoded)
    assert r is not None and r.model_id == "gemini-3-pro-image" and "prompt" not in r.context and r.model_known


def test_unknown_model_passes_params_through() -> None:
    r = decode_deep_link_payload(
        encode_deep_link_payload("nonexistent-model-xyz", {"prompt": "test", "aspectRatio": "4:3"})
    )
    assert r is not None
    assert (r.model_id, r.model_known) == ("nonexistent-model-xyz", False)
    assert any("Unknown model" in w for w in r.warnings)
    assert (r.context["prompt"], r.context["aspectRatio"]) == ("test", "4:3")


def test_only_cdn_urls_survive() -> None:
    r = decode_deep_link_payload(
        encode_deep_link_payload(
            "gemini-3-pro-image",
            {
                "prompt": "test",
                "imageUrls": [
                    f"{CDN}/good.jpg",
                    "data:image/png;base64,evil",
                    "https://evil.com/bad.jpg",
                    f"{CDN}/also-good.png",
                ],
            },
        )
    )
    assert r is not None
    assert r.context["imageUrls"] == [f"{CDN}/good.jpg", f"{CDN}/also-good.png"]
    assert r.warnings == ["Rejected URL: data:image/png;base64,evil", "Rejected URL: https://evil.com/bad.jpg"]


def test_invalid_input_is_none() -> None:
    assert decode_deep_link_payload("!!!invalid!!!") is None
    assert decode_deep_link_payload("") is None


# ── payload shape (serialize.ts) ───────────────────────────────────────
def test_payload_key_order_defaults_and_exclusions() -> None:
    encoded = encode_deep_link_payload(
        "veo-3.1",
        {
            "seed": 7,
            "endFrame": f"{CDN}/e.png",
            "aspectRatio": "16:9",  # the model default → left out of `o`
            "prompt": "p",
            "imageUrls": [],  # empty → no `i`
            "negativePrompt": "",  # falsy → no `np`
            "callbackUrl": "https://x",  # excluded
            "modelId": "x",  # excluded
            "unset": None,  # nullish → skipped
            "startFrame": f"{CDN}/s.png",
            "scale": 2.0,
        },
    )
    payload = raw(encoded)
    assert list(payload) == ["v", "m", "p", "sf", "ef", "o"]
    assert payload["o"] == {"seed": 7, "scale": 2}
    assert (
        '"scale":2}' in zlib.decompress(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)), -15).decode()
    )  # integral float written as an integer, like JSON.stringify


def test_defaults_compare_strictly() -> None:
    # `defaults[key] === value`: True is not the number 1 default
    install_catalog(
        [
            make_model(
                id="m",
                name="M",
                mode="image",
                param_config={"n": ParamEntry({"kind": "range", "min": 0, "max": 9, "default": 1})},
            )
        ]
    )
    assert raw(encode_deep_link_payload("m", {"n": True}))["o"] == {"n": True}
    assert "o" not in raw(encode_deep_link_payload("m", {"n": 1.0}))


def test_unicode_is_kept_raw_in_json() -> None:
    encoded = encode_deep_link_payload("gemini-3-pro-image", {"prompt": "закат 🌅"})
    text = zlib.decompress(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)), -15).decode()
    assert text == '{"v":1,"m":"gemini-3-pro-image","p":"закат 🌅"}'
    r = decode_deep_link_payload(encoded)
    assert r is not None and r.context["prompt"] == "закат 🌅"


# ── decode validation (deserialize.ts) ─────────────────────────────────
@pytest.mark.parametrize(
    "payload",
    [
        {"v": 2, "m": "x"},
        {"v": True, "m": "x"},  # `v === 1`: true is not 1
        {"v": "1", "m": "x"},
        {"v": 1, "m": ""},
        {"v": 1, "m": 5},
        [1, 2],
        "text",
        None,
    ],
)
def test_invalid_payloads_are_none(payload: Any) -> None:
    assert decode_deep_link_payload(make_link(payload)) is None


def test_v_as_float_one_is_valid() -> None:
    r = decode_deep_link_payload(make_link('{"v":1.0,"m":"kling-avatar"}'))
    assert r is not None and r.model_known


def test_rejected_media_warnings_and_known_model_validation() -> None:
    link = make_link(
        {
            "v": 1,
            "m": "kling-v3-pro",
            "p": "<i>hi</i>",
            "np": "",
            "vi": "https://evil.com/v.mp4",
            "au": f"{CDN}/a.mp3",
            "sf": 42,
            "ef": "http://cdn.picsart.com/e.png",
            "o": {"duration": 99, "junk": 1},
        }
    )
    r = decode_deep_link_payload(link)
    assert r is not None
    assert r.warnings == [
        "Rejected video URL: https://evil.com/v.mp4",
        "Rejected start frame URL: 42",
        "Rejected end frame URL: http://cdn.picsart.com/e.png",
    ]
    # transfer_values against the known model: an out-of-range duration is clamped, unknown keys are dropped
    assert r.context["prompt"] == "hi" and r.context["audioUrl"] == f"{CDN}/a.mp3"
    assert "junk" not in r.context and r.context["duration"] == 15


def test_non_string_prompt_raises_like_js() -> None:
    with pytest.raises(TypeError):
        decode_deep_link_payload(make_link({"v": 1, "m": "kling-avatar", "p": 5}))


def test_codec_round_trip_and_forgiving_base64() -> None:
    data = {"a": [1, 2.5, "ü"], "b": None}
    encoded = encode(data)
    assert decode(encoded) == data
    padded = encoded + "=" * (-len(encoded) % 4)
    assert decode(padded) == data  # atob accepts padding
    with pytest.raises(ValueError):
        decode("A")  # length ≡ 1 (mod 4) is invalid base64, as atob() says


# ── cross-SDK: links made by the TypeScript SDK (fflate) ───────────────
def test_links_encoded_by_the_ts_sdk() -> None:
    from pathlib import Path

    golden = json.loads((Path(__file__).parents[1] / "fixtures/golden/deeplink-ts.json").read_text("utf-8"))
    for case in golden["cases"]:
        assert decode(case["encoded"]) == case["payload"]
        assert decode(encode(case["payload"])) == case["payload"]  # bytes may differ (spec §5.9); content must not


def test_lone_surrogates_encode_like_json_stringify() -> None:
    from picsart_ai_sdk.core.js import js_json_dumps

    assert js_json_dumps("a\ud800b") == '"a\\ud800b"'
    assert js_json_dumps("😀") == '"😀"'  # a split pair is one character in JS
    encoded = encode_deep_link_payload("kling-avatar", {"prompt": "a\ud800b"})
    r = decode_deep_link_payload(encoded)
    assert r is not None and r.context["prompt"] == "a\ud800b"


def test_float_warnings_use_js_number_text() -> None:
    from picsart_ai_sdk.core.js import js_string

    assert (js_string(1.5e-7), js_string(1.2345678901234568e20), js_string(2.5)) == (
        "1.5e-7",
        "123456789012345680000",
        "2.5",
    )
    r = decode_deep_link_payload(make_link('{"v":1,"m":"kling-avatar","vi":1.5e-7}'))
    assert r is not None and r.warnings == ["Rejected video URL: 1.5e-7"]


def test_js_to_number_takes_ascii_digits_only() -> None:
    import math

    from picsart_ai_sdk.core.js import js_to_number

    assert math.isnan(js_to_number("١٢")) and math.isnan(js_to_number("１２"))
    assert js_to_number("12") == 12


def test_sanitize_prompt_handler_strip_is_linear_time() -> None:
    # Python's `re` backtracks `on\w+\s*=` quadratically over a long word with no `=` (32k chars took ~4.5 s).
    import time

    start = time.perf_counter()
    assert sanitize_prompt("on" * 20_000) == "on" * 20_000
    assert sanitize_prompt("on" * 20_000 + "=") == ""
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize(
    "text",
    [
        "onclick=x",
        "on=x",
        "onx=y",
        "ON_1 \t= z",
        "aonload=1",
        "xxonclick  =onfoo=bar",
        "onon= on onb=",
        "click on it, oncl ick=1",
        "on x﻿=1",
        "o\nnx=1 onx\n=2",
        "_on9=",
        "foonbar",
        "onab ==c",
        "é onab=1",
    ],
)
def test_sanitize_prompt_handler_strip_matches_the_js_regex(text: str) -> None:
    import re

    from picsart_ai_sdk.core.js import JS_WHITESPACE

    js_regex = re.compile(r"on[A-Za-z0-9_]+[" + re.escape(JS_WHITESPACE) + "]*=", re.IGNORECASE | re.ASCII)
    assert sanitize_prompt(text) == js_regex.sub("", text)


def test_sanitize_prompt_handler_strip_matches_the_js_regex_fuzzed() -> None:
    import random
    import re

    from picsart_ai_sdk.core.js import JS_WHITESPACE

    js_regex = re.compile(r"on[A-Za-z0-9_]+[" + re.escape(JS_WHITESPACE) + "]*=", re.IGNORECASE | re.ASCII)
    rng = random.Random(7)
    alphabet = ["o", "n", "O", "N", "x", "_", "1", "=", " ", "\t", " ", "é", "-", "\n"]
    for _ in range(5000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 24)))
        assert sanitize_prompt(text) == js_regex.sub("", text), text


def test_options_cannot_override_validated_dedicated_fields() -> None:
    # `o` must not smuggle a raw prompt or a non-CDN URL past the dedicated-field checks (known or unknown model).
    evil = "https://evil.com/a.png"
    params = {"prompt": "<script>x</script>", "negativePrompt": "<b>n</b>", "startFrame": evil, "imageUrls": [evil]}
    for model in ("kling-v3-pro", "not-in-catalog"):
        r = decode_deep_link_payload(
            make_link({"v": 1, "m": model, "p": "<i>hi</i>", "sf": evil, "o": {**params, "duration": 4}})
        )
        assert r is not None
        assert r.context.get("prompt") == "hi"
        assert not {"negativePrompt", "startFrame", "imageUrls"} & set(r.context)
        assert r.context["duration"] == 4

"""JS-semantics helpers in ``core/js.py`` (review findings on MR !2)."""

from collections import UserDict
from types import MappingProxyType

import pytest

from picsart_ai_sdk.client.drive import parse_generation
from picsart_ai_sdk.core.js import js_prop, utf16_code_at, utf16_len, utf16_slice


def test_utf16_helpers_keep_lone_surrogates_like_js_strings() -> None:
    # A JS string may hold a lone surrogate (the deep-link codec decodes one); `.length` / `.slice` /
    # `.charCodeAt` count it as one code unit instead of failing.
    assert utf16_len("a\ud800b") == 3
    assert utf16_slice("a\ud800b", 2) == "a\ud800"
    assert utf16_code_at("a\udc00b", 1) == 0xDC00
    # `'a😀'.slice(0, 2)` cuts the pair and keeps the high surrogate, as JS does.
    assert utf16_slice("a\U0001f600", 2) == "a\ud83d"
    assert utf16_slice("a\U0001f600", 3) == "a\U0001f600"
    assert utf16_len("a\U0001f600") == 3


@pytest.mark.parametrize("wrap", [MappingProxyType, UserDict])
def test_js_prop_reads_any_mapping(wrap: type) -> None:
    assert js_prop(wrap({"k": 1}), "k") == 1
    assert js_prop(wrap({}), "k") is None
    assert js_prop("str", "k") is None
    with pytest.raises(TypeError):
        js_prop(None, "k")


def test_parse_generation_accepts_a_read_only_mapping() -> None:
    attrs = {"model": "flux-2-pro", "aiSDKPayload": '{"prompt":"a cat"}'}
    g = parse_generation(MappingProxyType({"attributes": attrs}))
    assert g.model == "flux-2-pro" and g.ai_sdk_payload == {"prompt": "a cat"}

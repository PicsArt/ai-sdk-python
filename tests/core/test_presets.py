"""``p.*`` presets — descriptors as ``src/core/descriptors/presets.ts`` builds them (TS ``undefined`` left out)."""

import pytest

from picsart_ai_sdk.core.descriptors.presets import p
from picsart_ai_sdk.core.descriptors.types import ParamEntry
from picsart_ai_sdk.core.descriptors.utils import transfer_values, validate_all


def test_enum_presets_default_to_the_first_option() -> None:
    assert p.aspect_ratio() == {
        "aspectRatio": ParamEntry(
            {
                "kind": "enum",
                "valueType": "string",
                "options": [{"id": "16:9"}, {"id": "9:16"}, {"id": "1:1"}],
                "default": "16:9",
            }
        )
    }
    assert p.aspect_ratio(["1:1", "4:3"], "4:3")["aspectRatio"].descriptor["default"] == "4:3"
    assert p.count()["count"].descriptor == {
        "kind": "enum",
        "valueType": "number",
        "options": [{"id": n} for n in (1, 2, 4, 6, 8, 10)],
        "default": 1,
    }
    for name, key in (("resolution", "resolution"), ("quality", "quality"), ("size", "size")):
        d = getattr(p, name)(["a", "b"])[key].descriptor
        assert (d["valueType"], d["options"], d["default"]) == ("string", [{"id": "a"}, {"id": "b"}], "a")
    style = p.style([{"id": "s1", "label": "One"}])["style"].descriptor
    assert style["options"] == [{"id": "s1", "label": "One"}] and style["default"] == "s1"
    assert p.rendering_speed([{"id": "fast", "label": "Fast"}], "x")["renderingSpeed"].descriptor["default"] == "x"


def test_duration_accepts_bare_seconds_and_labelled_options() -> None:
    d = p.duration([{"id": -1, "label": "Auto"}, 5])["duration"].descriptor
    assert d == {
        "kind": "enum",
        "valueType": "number",
        "options": [{"id": -1, "label": "Auto"}, {"id": 5}],
        "default": -1,
    }
    assert p.duration_range(2, 15, 5)["duration"].descriptor == {
        "kind": "range",
        "min": 2,
        "max": 15,
        "step": 1,
        "default": 5,
    }


def test_boolean_and_text_presets() -> None:
    assert p.generate_audio() == {"generateAudio": ParamEntry({"kind": "boolean", "default": True})}
    assert p.return_last_frame() == {
        "returnLastFrame": ParamEntry({"kind": "boolean", "default": False}, label="Capture Last Frame")
    }
    assert p.enhance_prompt(False)["enhancePrompt"].descriptor["default"] is False
    assert p.audio_setting()["audioSetting"] == ParamEntry(
        {"kind": "enum", "valueType": "string", "options": [{"id": "auto"}, {"id": "origin"}], "default": "auto"},
        label="Audio",
    )
    assert p.prompt() == {"prompt": ParamEntry({"kind": "text"}, label="Prompt", required=True)}
    assert p.prompt(max_length=10, required=False)["prompt"] == ParamEntry(
        {"kind": "text", "maxLength": 10}, label="Prompt", required=False
    )
    assert p.negative_prompt("no blur")["negativePrompt"] == ParamEntry(
        {"kind": "text", "placeholder": "no blur"}, label="Negative Prompt"
    )
    assert p.text("caption", max_length=5, label="Caption")["caption"] == ParamEntry(
        {"kind": "text", "maxLength": 5}, label="Caption"
    )
    lang = p.language(True)
    assert list(lang) == ["language", "accent"] and list(p.language(False)) == ["language"]


def test_range_presets() -> None:
    assert p.seed()["seed"] == ParamEntry({"kind": "range", "min": 0, "max": 2147483647, "step": 1}, label="Seed")
    assert p.cfg_scale(1, 10, 5)["cfgScale"] == ParamEntry(
        {"kind": "range", "min": 1, "max": 10, "default": 5}, label="CFG Scale"
    )
    assert p.image_weight(0, 1, 0.5, 0.1)["imageWeight"].descriptor["step"] == 0.1
    assert p.range("strength", 0, 1, 0.3, label="Strength")["strength"] == ParamEntry(
        {"kind": "range", "min": 0, "max": 1, "default": 0.3}, label="Strength"
    )


def test_generic_enum_infers_the_value_type() -> None:
    assert p.enum("format", ["png", "jpeg"])["format"].descriptor["valueType"] == "string"
    assert p.enum("steps", [{"id": 4, "label": "Fast"}, 8], 8)["steps"].descriptor == {
        "kind": "enum",
        "valueType": "number",
        "options": [{"id": 4, "label": "Fast"}, {"id": 8}],
        "default": 8,
    }
    assert p.enum("flag", [True, False])["flag"].descriptor["valueType"] == "string"  # typeof true !== 'number'


def test_file_writes_only_the_bounds_given() -> None:
    entry = p.file(
        "videoUrl", "video", label="Video", required=True, category="reference", max_duration_sec=30.2, max_bytes=5
    )["videoUrl"]
    assert entry == ParamEntry(
        {"kind": "file", "accept": "video", "maxDurationSec": 30.2, "maxBytes": 5},
        label="Video",
        required=True,
        category="reference",
    )
    assert p.file("imageUrls", "image", array={})["imageUrls"].descriptor == {
        "kind": "file",
        "accept": "image",
        "array": {},
    }
    with pytest.raises(TypeError, match="unexpected bounds: max_size"):
        p.file("x", "image", max_size=1)


def test_catalog_and_bound_lists() -> None:
    built = p.catalog(
        "templateId", label="Effect", required=True, source={"workflow": "kling/v1/catalog/templates"}, default="kb"
    )
    entry = built["templateId"]
    assert (entry.label, entry.required, entry.descriptor["kind"], entry.descriptor["default"]) == (
        "Effect",
        True,
        "catalog",
        "kb",
    )
    validate_all(built, {"templateId": "anything-goes"})
    with pytest.raises(Exception, match="must be a string"):
        validate_all(built, {"templateId": 42})
    assert transfer_values(built, {"templateId": "hug_pro"})["templateId"] == "kb"

    seeded = p.voice_id([{"id": "seed-1", "name": "Seed"}], "seed-1", catalog={"workflow": "fake/v1/catalog/voices"})
    assert seeded["voiceId"].descriptor["kind"] == "catalog"
    assert seeded["voiceId"].catalog_options == [{"id": "seed-1", "name": "Seed"}]
    validate_all(seeded, {"voiceId": "not-in-seed"})
    with pytest.raises(Exception, match="must be a string"):
        validate_all(seeded, {"voiceId": 42})
    plain = p.voice_id([{"id": "only"}], "only")
    assert plain["voiceId"].descriptor == {
        "kind": "enum",
        "valueType": "string",
        "options": [{"id": "only", "label": "only"}],
        "default": "only",
    }
    with pytest.raises(Exception, match="must be one of"):
        validate_all(plain, {"voiceId": "other"})
    assert transfer_values(p.voice_id([{"id": "a"}, {"id": "b"}], "a"), {"voiceId": "b"})["voiceId"] == "b"
    avatar = p.video_id([{"id": "av", "name": "Ava"}], "av")["videoId"]
    assert avatar.label == "Avatar" and avatar.descriptor["options"] == [{"id": "av", "label": "Ava"}]

"""Port of ``__tests__/unit/helpers/fixture-catalog.ts``."""

from __future__ import annotations

import dataclasses
from typing import Any

from picsart_ai_sdk.core.descriptors.types import ParamEntry
from picsart_ai_sdk.core.types import ModelDefinition


def make_model(**over: Any) -> ModelDefinition:
    base: dict[str, Any] = {
        "provider": "flux",
        "provider_name": "Fixture",
        "provider_color": "#123456",
        "provider_label": "FX",
        "description": "test-only model",
        "features": [],
        "input_type": "t2i",
        "release": "production",
        "workflow": f"fixture/v1/{over['id']}",
        "param_config": {"prompt": ParamEntry(label="Prompt", required=True, descriptor={"kind": "text"})},
    }
    base.update(over)
    return ModelDefinition(**base)


FLUX = make_model(id="flux-2-pro", name="Flux 2 Pro", mode="image")
CLAUDE = make_model(
    id="claude-opus-5",
    name="Claude Opus 5",
    mode="text",
    input_type="t2t",
    workflow="claude/v1/messages",
    provider="anthropic",
)
KLING = make_model(
    id="kling-v3", name="Kling v3", mode="video", input_type="t2v", workflow="kling/v1/video", provider="kling"
)
FIXTURE_CATALOG = [FLUX, CLAUDE, KLING]

RICH = make_model(
    id="rich-fixture",
    name="Rich Fixture",
    mode="image",
    edit_workflow="fixture/v1/rich-edit",
    model_id="rich-vendor-id",
    badge=["new", "popular"],
    constraints=[{"when": {"resolution": "1080p"}, "require": {"duration": 8}}],
    param_config={
        "prompt": ParamEntry(label="Prompt", required=True, descriptor={"kind": "text", "maxLength": 2000}),
        "aspectRatio": ParamEntry(
            label="Aspect ratio",
            descriptor={
                "kind": "enum",
                "valueType": "string",
                "options": [{"id": "1:1"}, {"id": "16:9"}],
                "default": "1:1",
            },
        ),
        "voiceId": ParamEntry(
            label="Voice",
            descriptor={"kind": "catalog", "source": {"workflow": "fixture/v1/catalog/voices"}, "default": ""},
        ),
        "steps": ParamEntry(label="Steps", descriptor={"kind": "range", "min": 1, "max": 50, "default": 20}),
        "hd": ParamEntry(label="HD", descriptor={"kind": "boolean", "default": False}),
        "image": ParamEntry(
            label="Image",
            descriptor={
                "kind": "file",
                "accept": "image",
                "minPixels": 1024,
                "minSidePixels": 32,
                "maxShortSidePixels": 4096,
                "maxDurationSec": 60,
                "maxBytes": 30_000_000,
            },
        ),
        "turns": ParamEntry(
            label="Turns",
            descriptor={
                "kind": "object",
                "array": {"min": 1},
                "additionalProperties": {"kind": "unknown"},
                "fields": {"voiceId": {"kind": "text", "placeholder": "v1"}, "text": {"kind": "text"}},
            },
        ),
    },
)


def serializable_view(model: ModelDefinition) -> dict[str, Any]:
    """The model without what never travels on the wire (output_extension, hydrated catalog options)."""
    view = dataclasses.asdict(model)
    view.pop("output_extension")
    for entry in view["param_config"].values():
        entry.pop("catalog_options")
    return view

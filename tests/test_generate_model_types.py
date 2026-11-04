"""``scripts/generate_model_types.py`` — descriptor → type mapping (spec §6), determinism, ``--check``."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_model_types", ROOT / "scripts/generate_model_types.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_model_types"] = module
    spec.loader.exec_module(module)
    return module


gen = load_generator()


def body(*models: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "response": {"total": len(models), "models": list(models)}}


def model(model_id: str, params: dict[str, Any], mode: str = "image") -> dict[str, Any]:
    return {"id": model_id, "name": model_id, "mode": mode, "params": params}


RICH = model(
    "rich-2.5",
    {
        "prompt": {"kind": "text", "required": True, "maxLength": 10},
        "voiceId": {"kind": "catalog", "required": False, "source": {"workflow": "w"}, "default": ""},
        "aspectRatio": {
            "kind": "enum",
            "required": False,
            "valueType": "string",
            "options": [{"id": "1:1"}, {"id": "16:9"}],
        },
        "count": {
            "kind": "enum",
            "required": False,
            "valueType": "number",
            "options": [{"id": 1}, {"id": 2}, {"id": 1}],
        },
        "stability": {"kind": "enum", "required": False, "valueType": "number", "options": [{"id": 0}, {"id": 0.5}]},
        "duration": {"kind": "range", "required": False, "min": 3, "max": 15, "step": 1, "default": 5},
        "strength": {"kind": "range", "required": False, "min": 0, "max": 1, "step": 0.1},
        "seed": {"kind": "range", "required": False, "min": 0, "max": 9},
        "audio": {"kind": "boolean", "required": False, "default": True},
        "imageUrls": {"kind": "file", "required": True, "accept": "image", "array": {"min": 1}},
        "videoUrl": {"kind": "file", "required": False, "accept": "video"},
        "shots": {
            "kind": "object",
            "required": False,
            "array": {"max": 6},
            "fields": {"index": {"kind": "range", "required": True, "min": 0, "max": 5, "step": 1}},
        },
        "labels": {"kind": "object", "required": False, "additionalProperties": {"kind": "text", "required": False}},
        "mixed": {
            "kind": "object",
            "required": False,
            "fields": {"a": {"kind": "text"}},
            "additionalProperties": {"kind": "unknown"},
        },
        "extra": {"kind": "unknown", "required": False},
    },
)


def test_descriptor_mapping() -> None:
    out = gen.render(body(RICH))
    block = out[out.index("Rich25Input = TypedDict(") :]
    for line in (
        '"prompt": str,',
        '"voiceId": NotRequired[str],',
        '"aspectRatio": NotRequired[Literal["1:1", "16:9"]],',
        '"count": NotRequired[Literal[1, 2]],',  # duplicate ids collapse
        '"stability": NotRequired[float],',  # Literal cannot hold 0.5
        '"duration": NotRequired[int],',
        '"strength": NotRequired[float],',
        '"seed": NotRequired[float],',  # no step → not provably integral
        '"audio": NotRequired[bool],',
        '"imageUrls": list[str],',
        '"videoUrl": NotRequired[str],',
        '"shots": NotRequired[list[Rich25Shots]],',
        '"labels": NotRequired[dict[str, str]],',
        '"mixed": NotRequired[dict[str, Any]],',
        '"extra": NotRequired[object],',
    ):
        assert line in block, line
    nested = out[out.index("Rich25Shots = TypedDict(") :]
    assert '"index": int,' in nested
    assert out.index("Rich25Shots = TypedDict(") < out.index("Rich25Input = TypedDict(")  # defined before use


def test_ids_modes_and_maps_sorted_and_deterministic() -> None:
    models = [
        model("zeta", {"prompt": {"kind": "text", "required": True}}, "video"),
        model("chat-1", {"prompt": {"kind": "text", "required": True}}, "text"),
        model("judge", {"state": {"kind": "text", "required": True}}, "json"),
    ]
    out = gen.render(body(*models))
    assert out == gen.render(body(*reversed(models)))
    assert 'TypedModelId = Literal[\n    "chat-1",\n    "judge",\n    "zeta",\n]' in out
    assert 'TextModelId = Literal[\n    "chat-1",\n]' in out
    assert 'JsonModelId = Literal[\n    "judge",\n]' in out
    general, text = map_block(out, "ModelInputById"), map_block(out, "TextModelInputById")
    assert '"chat-1": Chat1Input,' in general and '"zeta": ZetaInput,' in general
    assert '"chat-1": Chat1Input,' in text and '"zeta"' not in text and '"judge"' not in text


def map_block(out: str, name: str) -> str:
    """The body of one ``<name> = TypedDict(...)`` assignment, anchored at the line start."""
    start = out.index(f"\n{name} = TypedDict(")
    return out[start : out.index("\n)\n", start)]


def test_map_block_is_anchored_at_the_assignment() -> None:
    out = 'ModelInputById = TypedDict(\n    {\n        "chat-1": Chat1Input,\n    },\n)\n'
    out = "\n" + out + "TextModelInputById = TypedDict(\n    {\n    },\n)\n"
    assert '"chat-1"' not in map_block(out, "TextModelInputById")
    assert '"chat-1"' in map_block(out, "ModelInputById")


def test_name_clashes_and_leading_digits() -> None:
    out = gen.render(
        body(
            model("gpt-5.1", {}),
            model("gpt-51", {}),
            model("3d-model", {}),
        )
    )
    assert 'M3dModelInput = TypedDict("M3dModelInput", {})' in out
    assert "Gpt51Input = TypedDict" in out and "Gpt51Input2 = TypedDict" in out


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"x": {"kind": "slider"}}, "unknown descriptor kind 'slider'"),
        ({"x": {"kind": "enum", "options": []}}, "enum without options"),
        ({"x": {"kind": "enum", "options": [{"id": "a"}, {"id": 1}]}}, "mixed or unsupported"),
        ({"x": "text"}, "descriptor is not an object"),
    ],
)
def test_unknown_shapes_fail_loudly(params: dict[str, Any], message: str) -> None:
    with pytest.raises(gen.SchemaError, match=message):
        gen.render(body(model("m", params)))


def test_bad_bodies_fail() -> None:
    with pytest.raises(gen.SchemaError, match="expected"):
        gen.render({"status": "success", "response": {"models": []}})
    with pytest.raises(gen.SchemaError, match="include=schema"):
        gen.render(body({"id": "m", "mode": "image"}))


def test_cli_write_check_and_exit_codes(tmp_path: Path) -> None:
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(body(RICH)), "utf-8")
    output = tmp_path / "model_types.py"
    assert gen.main(["--from-file", str(source), "--output", str(output), "--check"]) == 1  # missing → stale
    assert gen.main(["--from-file", str(source), "--output", str(output)]) == 0
    assert gen.main(["--from-file", str(source), "--output", str(output), "--check"]) == 0
    source.write_text(json.dumps(body(model("m", {"x": {"kind": "slider"}}))), "utf-8")
    assert gen.main(["--from-file", str(source), "--output", str(output)]) == 2


def test_generated_module_imports_and_typechecks(tmp_path: Path) -> None:
    output = tmp_path / "generated_types.py"
    output.write_text(gen.render(body(RICH)), "utf-8")
    spec = importlib.util.spec_from_file_location("generated_types", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.Rich25Input.__required_keys__) == {"prompt", "imageUrls"}
    assert "Rich25Input" in module.__all__


def test_the_committed_module_is_importable_from_the_package() -> None:
    from picsart_ai_sdk import JsonModelId, ModelInputById, TextModelId, TextModelInputById, TypedModelId
    from picsart_ai_sdk.model_types import Flux2ProInput

    assert "prompt" in Flux2ProInput.__required_keys__
    assert ModelInputById.__annotations__ and TextModelInputById.__annotations__
    assert TypedModelId and TextModelId and JsonModelId

# picsart-ai-sdk

Generate images, video, audio, and text with 100+ AI models — the Python counterpart of
[`@picsart/ai-sdk`](https://github.com/PicsArt/ai-sdk) (TypeScript), with the same public surface:
snake_case names, async like the TypeScript client, Python 3.10+.

## Documentation

- [Documentation](https://picsart.com/api-platform/docs) — overview and guides
- [Quickstart](https://picsart.com/api-platform/docs/quickstart) — install, authenticate, first generation
- [Authentication](https://picsart.com/api-platform/docs/authentication) — create an API key
- [Model catalog](https://picsart.com/api-platform/models) — browse every supported model

Repository: [github.com/PicsArt/ai-sdk-python](https://github.com/PicsArt/ai-sdk-python)

## Quick start

```bash
pip install picsart-ai-sdk
```

```python
import asyncio
import os

from picsart_ai_sdk import Model, catalog, create_client


async def main() -> None:
    async with create_client(api_key=os.environ["PICSART_API_KEY"], api_url="https://api.picsart.com") as ai:
        # Generate — model ids are plain strings, resolved server-side; no catalog load needed.
        result = await ai.generate("flux-2-pro", {"prompt": "a cat on mars"})
        print(result.url)

        text = await ai.generate_text("gemini-2.5-flash", {"prompt": "Name three colors"})
        print(text.text)

        # The local catalog accessors (Model / catalog / ai.validate) read the models catalog: load it once.
        await ai.init_model_catalog()
        catalog.all()  # every model
        catalog.find(output="video")  # video models only
        catalog.search("kling")  # by name / id / provider
        Model("flux-2-pro").meta()  # mode, provider, badges, …
        Model("flux-2-pro").params().to_schema()  # param schema
        print(Model("flux-2-pro").validate({"prompt": "a cat"}))  # ValidationResult(valid=True, errors=None)


asyncio.run(main())
```

Every generation rides the models-hub `v1/models` workflow: the server resolves the model, builds the vendor
payload and normalizes the result. The catalog accessors raise `ApiError` 409 `catalog_not_loaded` until
`await ai.init_model_catalog()`; `generate()` / `generate_text()` / `run()` / `submit()` never wait on it.

## Authentication

Pass `api_key` and the SDK sends `Authorization: Bearer <api_key>` on every request. Apps that handle auth
themselves can pass their own async `fetch(url, init) -> httpx.Response` instead (it wins over `api_key`):

```python
ai = create_client(fetch=my_authenticated_fetch, api_url="https://api.picsart.com")
```

Or replace the requests altogether with a custom `transport` (`execute`, and optionally `submit` / `poll` /
`status` / `options`) — the same contract as the TypeScript `SdkTransport`.

## Async jobs, progress and cancellation

```python
from picsart_ai_sdk import AbortController, GenerateOptions, GenerationEventType

generation_id = await ai.submit("kling-v3", {"prompt": "a drone shot over the sea"})

async with ai.subscribe("kling-v3", generation_id) as events:
    async for event in events:
        if event.type == GenerationEventType.Completed:
            print(event.result.url)

controller = AbortController()
job = asyncio.create_task(ai.generate("kling-v3", {"prompt": "…"}, GenerateOptions(signal=controller.signal)))
controller.abort()  # `await job` raises ApiError 499 `aborted`; `job.cancel()` works too
```

Failures are always `ApiError` (`status`, `code`, message), as in the TypeScript SDK.

## More

- **Typed inputs** — `picsart_ai_sdk.model_types` has one `TypedDict` per model (`Flux2ProInput`, …) plus
  `TypedModelId`, `TextModelId`, `JsonModelId`; it is regenerated from the models catalog on every catalog
  release.
- **Raw APIs** — `await ai.apis.run("vendor/v1/task", payload)` runs any model API by name.
- **Catalogs** — `await ai.catalogs.voices(model)` / `avatars` / `templates`: paged, cached, and hydrated
  into `Model(id).params()`.
- **Drive** — `create_client(..., drive=DriveConfig(folder="AI"))` saves generations to Picsart Drive and
  exposes `ai.drive` (folders, listing, `save`, reactions, `get_generation`).
- **Deep links** — `encode_deep_link_payload(model, context)` / `decode_deep_link_payload(encoded)`, compatible
  with the links the TypeScript SDK makes.
- **Credits** — `await ai.get_credits(model, params)`; `catalog.pricing` for credit ranges.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check . && ruff format --check . && mypy && pytest
```

`scripts/generate_model_types.py` regenerates `model_types.py` (`PICSART_TOKEN`, or `--from-file` a saved
`/v1/models-catalog` response). `tests/golden/ts-sdk.json` holds cases recorded from the TypeScript SDK that
this SDK must reproduce.

## License

MIT

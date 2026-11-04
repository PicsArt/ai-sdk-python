"""``create_client`` and the generation surface — port of ``src/client/index.ts``.

The SDK is dynamic-only: every generation rides the models-hub ``v1/models``
workflow; the model catalog (``init_model_catalog``) only feeds local accessors
and Drive naming and never gates a request.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx

from ..abort import AbortController, AbortError, AbortSignal, link_signals
from ..core.active_catalog import peek_active_models
from ..core.catalog_store import ModelCatalogFilters, load_model_catalog, when_model_catalog_loaded
from ..core.descriptors.model_accessor import Model
from ..core.descriptors.types import ValidationResult
from ..core.errors import ApiError
from ..core.js import is_number, nullish
from ..core.model_registry import find_model
from ..core.response import to_completed_status
from ..core.types import ModelDefinition
from ..core.workflow import (
    OptionsResult,
    SdkTransport,
    TransportPollOptions,
    TransportResult,
    WorkflowJobHandle,
    WorkflowProgress,
    WorkflowStatusResult,
    WorkflowSubmitRequest,
    has_method,
)
from .apis import ApisClient, create_apis
from .catalogs import CatalogsClient, create_catalogs
from .drive import DriveClient, DriveFolder, create_drive_client
from .drive_naming import (
    AppIdentity,
    GenerationProvenance,
    ModelLike,
    build_filename,
    build_generation_attributes,
    expected_output_format,
)
from .fetch import AuthenticatedFetch
from .prepare import (
    MODELS_WORKFLOW,
    build_models_command,
    parse_dynamic_result,
    parse_dynamic_run_result,
    parse_dynamic_text_result,
)
from .transport import build_transport, create_workflows_client, maybe_fetch
from .types import (
    GenerateResult,
    GenerateTextResult,
    GenerationCompletedEvent,
    GenerationEvent,
    GenerationFailedEvent,
    GenerationOptions,
    GenerationProgressEvent,
    InputsTransformation,
    RunResponse,
)
from .workflows import WorkflowsClient

# Poll cadence per mode (TS MODE_POLL_DEFAULTS). Dynamic mode knows no mode up front,
# so every job starts from the video default — the longest — as in TS.
VIDEO_POLL_DEFAULTS = (2000, 1800)


@dataclass(frozen=True)
class DriveConfig:
    """Root Drive folder generations are saved into."""

    folder: str


@dataclass(frozen=True)
class CatalogsConfig:
    preload: bool | None = None


class PayloadDriveFolderOptions(TypedDict, total=False):
    path: str
    id: str


class PayloadDriveOptions(TypedDict, total=False):
    """``options.drive`` of a generation — the wire keys of the backend-side save."""

    name: str
    attributes: dict[str, str]
    folder: PayloadDriveFolderOptions
    generation: GenerationProvenance


@dataclass(frozen=True)
class GenerateOptions:
    signal: AbortSignal | None = None
    interval_ms: int | None = None
    max_attempts: int | None = None
    folder: DriveFolder | None = None
    drive: PayloadDriveOptions | None = None
    inputs_transformation: InputsTransformation | None = None
    app: AppIdentity | None = None


@dataclass(frozen=True)
class _EmptyModel:
    param_config: Mapping[str, Any]
    output_extension: str | None = None


def _peek_model(ref: str) -> ModelDefinition | None:
    """The installed catalog's model for ``ref`` without waiting (TS ``peekActiveModels() !== null && findModel``).

    Generation never waits for or validates against the catalog; it only names Drive files after it.
    """
    return find_model(ref) if peek_active_models() is not None else None


def _as_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Public methods take any ``Mapping`` (the generated ``TypedDict`` inputs); a plain dict passes through as is."""
    return params if isinstance(params, dict) else dict(params)


def _unsupported_transport(capability: str) -> ApiError:
    return ApiError(
        f"Transport does not support {capability} (execute-only transport)", status=400, code="unsupported_transport"
    )


def _to_api_error(err: BaseException, signal: AbortSignal | None) -> BaseException:
    """``toApiError`` of the client: ApiError as is; an abort of ours → 499; else 502 ``generation_failed``."""
    if isinstance(err, ApiError):
        return err
    if isinstance(err, AbortError):
        if signal is not None and signal.aborted:
            return ApiError("Operation aborted", status=499, code="aborted")
        return err
    return ApiError(str(err), status=502, code="generation_failed")


def _raise_api(err: BaseException, signal: AbortSignal | None) -> None:
    mapped = _to_api_error(err, signal)
    if mapped is err:
        raise err
    raise mapped from err


class AiClient:
    """The client ``create_client`` returns.

    ``ai.apis`` runs any model API by name; ``ai.catalogs`` pages voice / avatar / template catalogs;
    ``ai.drive`` (with ``drive=DriveConfig(...)``) manages the generations saved to Picsart Drive.
    """

    def __init__(
        self,
        *,
        transport: SdkTransport,
        workflows: WorkflowsClient | None,
        drive_config: DriveConfig | None,
        inputs_transformation: InputsTransformation | None,
        http: httpx.AsyncClient | None,
        api_url: str | None = None,
        authed_fetch: AuthenticatedFetch | None = None,
        catalogs: CatalogsConfig | None = None,
    ) -> None:
        self._api_url = api_url
        self._authed_fetch = authed_fetch
        self._transport = transport
        self._workflows = workflows
        self._drive_config = drive_config
        self._inputs_transformation = inputs_transformation
        self._http = http
        self._supports_async = has_method(transport, "submit") and has_method(transport, "poll")
        self.apis: ApisClient = create_apis(workflows)
        """Direct, low-level access to the Picsart model APIs."""
        self.catalogs: CatalogsClient = create_catalogs(
            transport, preload=bool(catalogs.preload) if catalogs is not None else False
        )
        """Voice / avatar / template catalogs: fetch, ttl-cache, hydrate model params."""
        self.drive: DriveClient | None = (
            create_drive_client(authed_fetch, api_url, drive_config.folder)
            if drive_config is not None and authed_fetch is not None and api_url
            else None
        )
        """Drive operations — only with ``drive=DriveConfig(...)``."""

    async def aclose(self) -> None:
        """Close the HTTP connection pool the client owns (none when a custom ``fetch`` / transport is used)."""
        if self._http is not None:
            await self._http.aclose()

    async def __aenter__(self) -> AiClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ── lifecycle helpers ───────────────────────────────────────────────
    def _assert_async_lifecycle(self) -> None:
        if not has_method(self._transport, "submit"):
            raise _unsupported_transport("submit")
        if not has_method(self._transport, "poll"):
            raise _unsupported_transport("polling")

    async def _submit_job(self, workflow: str, payload: dict[str, Any], signal: AbortSignal | None) -> str:
        if signal is not None and signal.aborted:
            raise ApiError("Operation aborted", status=499, code="aborted")
        if not has_method(self._transport, "submit"):
            raise _unsupported_transport("submit")
        task_id: str = await self._transport.submit(WorkflowSubmitRequest(workflow, payload, signal))  # type: ignore[attr-defined]
        if not task_id:
            raise ApiError("No task id in response", status=502, code="invalid_response")
        return task_id

    async def _poll_job(
        self,
        handle: WorkflowJobHandle,
        poll: TransportPollOptions,
        on_progress: Callable[[WorkflowProgress], None] | None = None,
    ) -> TransportResult:
        if not has_method(self._transport, "poll"):
            raise _unsupported_transport("polling")
        options = TransportPollOptions(poll.interval_ms, poll.max_attempts, poll.signal, on_progress)
        result: TransportResult = await self._transport.poll(handle, options)  # type: ignore[attr-defined]
        return result

    @staticmethod
    def _poll_options_for(overrides: GenerateOptions | GenerationOptions | None) -> TransportPollOptions:
        interval, attempts = VIDEO_POLL_DEFAULTS
        signal = None
        if overrides is not None:
            interval = overrides.interval_ms if overrides.interval_ms is not None else interval
            attempts = overrides.max_attempts if overrides.max_attempts is not None else attempts
            signal = overrides.signal
        return TransportPollOptions(interval_ms=interval, max_attempts=attempts, signal=signal)

    @staticmethod
    def _as_completed(handle: WorkflowJobHandle, res: TransportResult) -> WorkflowStatusResult:
        return to_completed_status(handle, res.result, nullish(res.raw, res.result), res.usage)

    async def _execute_dynamic(self, command: dict[str, Any], options: GenerateOptions | None) -> WorkflowStatusResult:
        signal = options.signal if options else None
        try:
            if not self._supports_async:
                res = await self._transport.execute(WorkflowSubmitRequest(MODELS_WORKFLOW, command, signal))
                return to_completed_status(
                    WorkflowJobHandle(MODELS_WORKFLOW, ""), res.result, nullish(res.raw, res.result), res.usage
                )
            task_id = await self._submit_job(MODELS_WORKFLOW, command, signal)
            res = await self._poll_job(WorkflowJobHandle(MODELS_WORKFLOW, task_id), self._poll_options_for(options))
            return self._as_completed(WorkflowJobHandle(MODELS_WORKFLOW, task_id), res)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_api(err, signal)
            raise

    # ── command building ────────────────────────────────────────────────
    def _drive_options(
        self, model_id: str, mode: str, model: ModelLike, params: dict[str, Any], options: GenerateOptions | None
    ) -> dict[str, Any] | None:
        explicit = options.drive if options else None
        if self._drive_config is None and explicit is None:
            return None
        app = options.app if options else None
        generation = explicit.get("generation") if explicit else None
        provenance = (
            GenerationProvenance(generation.model_id, generation.params, generation.app or app)
            if generation is not None
            else GenerationProvenance(model_id, params, app)
        )
        attributes = {**((explicit or {}).get("attributes") or {}), **build_generation_attributes(provenance)}
        folder_path = nullish(
            options.folder.name if options and options.folder else None,
            self._drive_config.folder if self._drive_config else None,
        )
        prompt = params.get("prompt")
        name = (explicit or {}).get("name")
        drive: dict[str, Any] = {
            "name": name
            if name is not None
            else build_filename(
                prompt if isinstance(prompt, str) else None,
                mode,
                {"format": expected_output_format(model, params) or ""},
            ),
            "attributes": attributes,
        }
        folder = (explicit or {}).get("folder")
        if folder is not None:
            drive["folder"] = folder
        elif folder_path:
            drive["folder"] = {"path": folder_path}
        return drive

    def _dynamic_drive_options(
        self, model_ref: str, params: dict[str, Any], options: GenerateOptions | None
    ) -> dict[str, Any] | None:
        definition = _peek_model(model_ref)
        if definition is not None:
            return self._drive_options(definition.id, definition.mode, definition, params, options)
        return self._drive_options(model_ref, "unknown", _EmptyModel({}), params, options)

    def _downscale(self, options: GenerateOptions | None) -> bool:
        return bool(
            nullish(
                options.inputs_transformation.downscale_oversized_images
                if options and options.inputs_transformation
                else None,
                self._inputs_transformation.downscale_oversized_images if self._inputs_transformation else None,
                False,
            )
        )

    def _media_command(self, model: str, params: dict[str, Any], options: GenerateOptions | None) -> dict[str, Any]:
        return build_models_command(
            model,
            params,
            with_options=True,
            drive=self._dynamic_drive_options(model, params, options),
            downscale_oversized_images=self._downscale(options),
        )

    async def _fetch_options(self, model: str, params: dict[str, Any]) -> OptionsResult | None:
        if not has_method(self._transport, "options"):
            return None
        res = await self._transport.options(MODELS_WORKFLOW, build_models_command(model, params))  # type: ignore[attr-defined]
        if is_number(res):
            return {"credits": res}
        if isinstance(res, dict) and is_number(res.get("credits")):
            return res  # type: ignore[return-value]
        return None

    # ── catalog ─────────────────────────────────────────────────────────
    async def init_model_catalog(
        self,
        filters_options: ModelCatalogFilters | None = None,
        refresh_interval_seconds: float = -1,
    ) -> None:
        """Load the model catalog from models-hub (once per process; optional background refresh)."""
        if not self._api_url or self._authed_fetch is None:
            raise ApiError(
                "initModelCatalog() requires `apiUrl` plus `fetch` or `apiKey` — the catalog is a REST surface, "
                "not served by a custom transport.",
                status=400,
                code="validation_error",
            )
        await load_model_catalog(self._authed_fetch, self._api_url, filters_options, refresh_interval_seconds)

    async def validate(self, model: str, params: Mapping[str, Any]) -> ValidationResult:
        """Validate ``params`` against the loaded catalog (waits for an in-flight ``init_model_catalog``)."""
        loading = when_model_catalog_loaded()
        if loading is None:
            raise ApiError(
                "Model catalog not loaded — call ai.initModelCatalog() before ai.validate().",
                status=409,
                code="catalog_not_loaded",
            )
        await loading
        return Model(model).validate(_as_params(params))

    # ── public surface ──────────────────────────────────────────────────
    async def run(
        self, model: str, params: Mapping[str, Any], options: GenerateOptions | None = None
    ) -> RunResponse[Any]:
        completed = await self._execute_dynamic(build_models_command(model, _as_params(params)), options)
        return parse_dynamic_run_result(completed, model)

    async def generate(
        self, model: str, params: Mapping[str, Any], options: GenerateOptions | None = None
    ) -> GenerateResult:
        completed = await self._execute_dynamic(self._media_command(model, _as_params(params), options), options)
        return parse_dynamic_result(completed, model)

    async def generate_text(
        self, model: str, params: Mapping[str, Any], options: GenerateOptions | None = None
    ) -> GenerateTextResult:
        completed = await self._execute_dynamic(build_models_command(model, _as_params(params)), options)
        return parse_dynamic_text_result(completed, model)

    async def get_credits(self, model: str, params: Mapping[str, Any]) -> float | None:
        res = await self._fetch_options(model, _as_params(params))
        return res["credits"] if res is not None else None

    async def options(self, model: str, params: Mapping[str, Any]) -> OptionsResult | None:
        return await self._fetch_options(model, _as_params(params))

    async def submit(self, model: str, params: Mapping[str, Any], options: GenerateOptions | None = None) -> str:
        self._assert_async_lifecycle()
        command = self._media_command(model, _as_params(params), options)
        signal = options.signal if options else None
        try:
            return await self._submit_job(MODELS_WORKFLOW, command, signal)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_api(err, signal)
            raise

    async def result(self, model: str, generation_id: str, options: GenerationOptions | None = None) -> GenerateResult:
        self._assert_async_lifecycle()
        handle = WorkflowJobHandle(MODELS_WORKFLOW, generation_id)
        signal = options.signal if options else None
        try:
            completed = self._as_completed(handle, await self._poll_job(handle, self._poll_options_for(options)))
        except asyncio.CancelledError:
            raise
        except Exception as err:
            _raise_api(err, signal)
            raise
        return parse_dynamic_result(completed, model)

    def subscribe(
        self, model: str, generation_id: str, options: GenerationOptions | None = None
    ) -> GenerationSubscription:
        """Progress events, then one completed or failed event.

        Use it as ``async with ai.subscribe(...) as events: async for e in events: ...`` — leaving the
        ``async with`` block (``break``, ``return``, an exception) stops polling at once, as ``break``
        does in TS. A bare ``async for`` works too; it stops polling when the iteration ends or when
        the subscription is closed (``await sub.aclose()``) or garbage-collected.
        """
        self._assert_async_lifecycle()
        return GenerationSubscription(
            self._subscribe(
                WorkflowJobHandle(MODELS_WORKFLOW, generation_id),
                self._poll_options_for(options),
                lambda completed: parse_dynamic_result(completed, model),
                options.signal if options else None,
            )
        )

    async def _subscribe(
        self,
        handle: WorkflowJobHandle,
        poll: TransportPollOptions,
        parse: Callable[[WorkflowStatusResult], GenerateResult],
        signal: AbortSignal | None,
    ) -> AsyncGenerator[GenerationEvent, None]:
        controller = AbortController()
        unlink = link_signals(signal, controller)
        queue: list[WorkflowProgress] = []
        wake = asyncio.Event()

        def on_progress(progress: WorkflowProgress) -> None:
            queue.append(progress)
            wake.set()

        options = TransportPollOptions(poll.interval_ms, poll.max_attempts, controller.signal)
        done: asyncio.Task[TransportResult] = asyncio.ensure_future(self._poll_job(handle, options, on_progress))
        try:
            while True:
                while queue:
                    yield GenerationProgressEvent(queue.pop(0))
                wake.clear()
                waiter = asyncio.ensure_future(wake.wait())
                try:
                    await asyncio.wait({done, waiter}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    waiter.cancel()
                if not done.done():
                    continue
                while queue:
                    yield GenerationProgressEvent(queue.pop(0))
                err = done.exception()
                if err is not None:
                    mapped = _to_api_error(err, signal)
                    if not isinstance(mapped, ApiError):
                        raise mapped  # an abort that is not ours propagates, as in TS
                    yield GenerationFailedEvent(mapped)
                    return
                completed = self._as_completed(handle, done.result())
                try:
                    yield GenerationCompletedEvent(parse(completed))
                except ApiError as parse_err:
                    yield GenerationFailedEvent(parse_err)
                return
        finally:
            controller.abort()
            unlink()
            if not done.done():
                done.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await done


class GenerationSubscription:
    """``subscribe()``'s events: an async iterator that is also an async context manager."""

    def __init__(self, events: AsyncGenerator[GenerationEvent, None]) -> None:
        self._events = events

    def __aiter__(self) -> GenerationSubscription:
        return self

    async def __anext__(self) -> GenerationEvent:
        return await self._events.__anext__()

    async def aclose(self) -> None:
        """Stop polling and release the subscription."""
        await self._events.aclose()

    async def __aenter__(self) -> GenerationSubscription:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# Node's fetch (undici) gives up on a response whose headers or body stall for 300 s;
# the workflows client then retries it as a connection error. Same limits here.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=None, read=300.0, write=None, pool=None)


def create_client(
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    fetch: AuthenticatedFetch | None = None,
    transport: SdkTransport | None = None,
    drive: DriveConfig | None = None,
    inputs_transformation: InputsTransformation | None = None,
    catalogs: CatalogsConfig | None = None,
) -> AiClient:
    """Create a client. Pass ``api_url`` with ``api_key`` (or your own authenticated ``fetch``),
    or a custom ``transport``."""
    has_auth = fetch is not None or bool(api_key)
    # Validate first, so a rejected config never opens a connection pool.
    if transport is None and not (api_url and has_auth):
        raise Exception(  # noqa: TRY002 — TS throws a plain Error here
            "createClient config requires either `fetch` or `apiKey` (or a custom `transport`)."
            if api_url
            else "createClient config requires `apiUrl` (or a custom `transport`)."
        )
    if drive is not None and not (has_auth and api_url):
        raise Exception(  # noqa: TRY002
            "createClient `drive` requires `apiUrl` plus `fetch` or `apiKey` — Drive is a REST surface, "
            "not served by a custom transport."
        )
    # The default fetch needs a pool only when it will be used: a URL to call and no custom fetch.
    http = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) if fetch is None and api_key and api_url else None
    authed = maybe_fetch(fetch=fetch, api_key=api_key, http=http)
    workflows = create_workflows_client(api_url, authed) if api_url and authed else None
    if transport is None:
        assert workflows is not None
        transport = build_transport(workflows)
    return AiClient(
        transport=transport,
        workflows=workflows,
        drive_config=drive,
        inputs_transformation=inputs_transformation,
        http=http,
        api_url=api_url,
        authed_fetch=authed,
        catalogs=catalogs,
    )


__all__ = [
    "AiClient",
    "CatalogsConfig",
    "GenerationSubscription",
    "DriveConfig",
    "DriveFolder",
    "GenerateOptions",
    "PayloadDriveFolderOptions",
    "PayloadDriveOptions",
    "create_client",
]

"""Picsart AI SDK for Python — generate images, video, audio, and text with 100+ AI models.

The Python counterpart of the TypeScript SDK ``@picsart/ai-sdk``, with the same
public surface (snake_case names, async like the TypeScript client).
"""

from importlib import metadata as _metadata

from .abort import AbortController, AbortError, AbortSignal
from .client.apis import ApiResponse, ApisClient
from .client.catalogs import CatalogPage, CatalogsClient
from .client.client import (
    AiClient,
    CatalogsConfig,
    DriveConfig,
    GenerateOptions,
    GenerationSubscription,
    PayloadDriveFolderOptions,
    PayloadDriveOptions,
    create_client,
)
from .client.drive import (
    DriveClient,
    DriveFileDetails,
    DriveFolder,
    DriveMediaItem,
    DriveSaveResult,
    GenerationFile,
    MediaTypeFilter,
    SaveParams,
    UserReaction,
    parse_generation,
)
from .client.drive_naming import (
    AppIdentity,
    AppType,
    FilenameHints,
    GenerationProvenance,
    build_filename,
    build_generation_attributes,
    expected_output_format,
    infer_resource_type,
    resolve_extension,
)
from .client.fetch import AuthenticatedFetch, FetchInit
from .client.types import (
    GenerateResult,
    GenerateResultItem,
    GenerateTextResult,
    GenerationCompletedEvent,
    GenerationEvent,
    GenerationEventType,
    GenerationFailedEvent,
    GenerationOptions,
    GenerationProgressEvent,
    InputsTransformation,
    RunResponse,
)
from .client.workflows import ExecutionMode as ApiRunMode
from .core.catalog_store import ModelCatalogFilters
from .core.catalog_wire import deserialize_catalog, serialize_catalog
from .core.catalogs import (
    CatalogItem,
    CatalogPreview,
    CatalogQuery,
    CatalogResult,
    CatalogSource,
    get_voice_by_id,
    to_avatar_option,
    to_voice_option,
)
from .core.deeplink import DeepLinkResult, decode_deep_link_payload, encode_deep_link_payload
from .core.descriptors.model_accessor import Model, ModelDescriptor, ModelParamsAccessor, catalog
from .core.descriptors.pricing import PricingOptions
from .core.descriptors.types import (
    CreditRange,
    CreditRangeContext,
    CreditTier,
    EnumOption,
    FlatParamEntry,
    ModelFilter,
    ModelMeta,
    ModelParams,
    ParamDescriptor,
    ParamEntry,
    ProviderInfo,
    ValidationResult,
)
from .core.errors import ApiError, ApiErrorCode
from .core.model_registry import find_model, get_model
from .core.response import GenerateResultItemMetadata, SeedanceDraftTask
from .core.types import AvatarOption, GenerationMode, ModelDefinition, ParamOption, ReleaseTag, VoiceOption
from .core.visibility import DEFAULT_VISIBLE_RELEASES, is_visible_for_releases, release_of
from .core.workflow import (
    CreditUsage,
    OptionsResult,
    SdkTransport,
    ToolUsage,
    TransportPollOptions,
    TransportResult,
    WorkflowJobHandle,
    WorkflowProgress,
    WorkflowSubmitRequest,
)
from .model_types import JsonModelId, ModelInputById, TextModelId, TextModelInputById, TypedModelId

GenerationProgress = WorkflowProgress

try:
    __version__ = _metadata.version("picsart-ai-sdk")
except _metadata.PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = [
    "AbortController",
    "AbortError",
    "AbortSignal",
    "AiClient",
    "ApiError",
    "ApiErrorCode",
    "ApiResponse",
    "ApiRunMode",
    "ApisClient",
    "AppIdentity",
    "AppType",
    "AuthenticatedFetch",
    "AvatarOption",
    "CatalogItem",
    "CatalogPage",
    "CatalogPreview",
    "CatalogQuery",
    "CatalogResult",
    "CatalogSource",
    "CatalogsClient",
    "CatalogsConfig",
    "CreditRange",
    "CreditRangeContext",
    "CreditTier",
    "CreditUsage",
    "DEFAULT_VISIBLE_RELEASES",
    "DeepLinkResult",
    "DriveClient",
    "DriveConfig",
    "DriveFileDetails",
    "DriveFolder",
    "DriveMediaItem",
    "DriveSaveResult",
    "EnumOption",
    "FetchInit",
    "FilenameHints",
    "FlatParamEntry",
    "GenerateOptions",
    "GenerateResult",
    "GenerateResultItem",
    "GenerateResultItemMetadata",
    "GenerateTextResult",
    "GenerationCompletedEvent",
    "GenerationEvent",
    "GenerationEventType",
    "GenerationFailedEvent",
    "GenerationFile",
    "GenerationMode",
    "GenerationOptions",
    "GenerationProgress",
    "GenerationProgressEvent",
    "GenerationProvenance",
    "GenerationSubscription",
    "InputsTransformation",
    "JsonModelId",
    "MediaTypeFilter",
    "Model",
    "ModelCatalogFilters",
    "ModelDefinition",
    "ModelDescriptor",
    "ModelFilter",
    "ModelInputById",
    "ModelMeta",
    "ModelParams",
    "ModelParamsAccessor",
    "OptionsResult",
    "ParamDescriptor",
    "ParamEntry",
    "ParamOption",
    "PayloadDriveFolderOptions",
    "PayloadDriveOptions",
    "PricingOptions",
    "ProviderInfo",
    "ReleaseTag",
    "RunResponse",
    "SaveParams",
    "SdkTransport",
    "SeedanceDraftTask",
    "TextModelId",
    "TextModelInputById",
    "ToolUsage",
    "TransportPollOptions",
    "TransportResult",
    "TypedModelId",
    "UserReaction",
    "ValidationResult",
    "VoiceOption",
    "WorkflowJobHandle",
    "WorkflowSubmitRequest",
    "__version__",
    "build_filename",
    "build_generation_attributes",
    "catalog",
    "create_client",
    "decode_deep_link_payload",
    "deserialize_catalog",
    "encode_deep_link_payload",
    "expected_output_format",
    "find_model",
    "get_model",
    "get_voice_by_id",
    "infer_resource_type",
    "is_visible_for_releases",
    "parse_generation",
    "release_of",
    "resolve_extension",
    "serialize_catalog",
    "to_avatar_option",
    "to_voice_option",
]

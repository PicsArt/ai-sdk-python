"""Results and options of the generation surface — port of ``src/client/types.ts``.

SDK-built results are frozen dataclasses with snake_case fields; what the
server sent (``metadata``, ``usage``, ``progress``) stays a dict with its keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from ..abort import AbortSignal
from ..core.errors import ApiError
from ..core.response import GenerateResultItemMetadata
from ..core.workflow import CreditUsage, WorkflowProgress

R = TypeVar("R")


class GenerationEventType:
    Progress = "generation.progress"
    Completed = "generation.completed"
    Failed = "generation.failed"


@dataclass(frozen=True)
class GenerateResultItem:
    url: str
    metadata: GenerateResultItemMetadata | None = None


@dataclass(frozen=True)
class GenerateResult:
    url: str
    items: list[GenerateResultItem]
    results: list[GenerateResultItem]
    generation_id: str | None = None
    usage: CreditUsage | None = None


@dataclass(frozen=True)
class GenerateTextResult:
    text: str
    model: str
    raw: Any
    usage: CreditUsage | None = None


@dataclass(frozen=True)
class RunResponse(Generic[R]):
    result: R
    status: Literal["COMPLETED"] = "COMPLETED"
    usage: CreditUsage | None = None
    id: str | None = None


@dataclass(frozen=True)
class GenerationProgressEvent:
    progress: WorkflowProgress | None = None
    type: Literal["generation.progress"] = "generation.progress"


@dataclass(frozen=True)
class GenerationCompletedEvent:
    result: GenerateResult
    type: Literal["generation.completed"] = "generation.completed"


@dataclass(frozen=True)
class GenerationFailedEvent:
    error: ApiError
    type: Literal["generation.failed"] = "generation.failed"


GenerationEvent = GenerationProgressEvent | GenerationCompletedEvent | GenerationFailedEvent


@dataclass(frozen=True)
class InputsTransformation:
    downscale_oversized_images: bool | None = None


ServiceTier = Literal["auto", "priority"]
"""How a generation is served, sent as ``options.service_tier``.

- ``"auto"`` — the platform default. The user's discount is used when one applies, along with the limits it
  carries: a number of generations at a time (a 429 when they are all in use) and a wait before a generation
  starts after heavy use.
- ``"priority"`` — the regular price, with no concurrency limit and no wait. The discount is not used.
"""


@dataclass(frozen=True)
class PriceOptions:
    """Options of the price calls — ``get_credits`` / ``options``.

    ``service_tier`` overrides the client's. Pass the tier the generation will be sent with, so the quote is
    the price it is charged: ``"priority"`` quotes the regular price, not the discounted one.
    """

    service_tier: ServiceTier | None = None


@dataclass(frozen=True)
class GenerationOptions:
    interval_ms: int | None = None
    max_attempts: int | None = None
    signal: AbortSignal | None = None

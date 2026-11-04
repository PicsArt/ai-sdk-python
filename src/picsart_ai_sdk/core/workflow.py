"""Transport-level types — port of ``src/core/workflow.ts``.

Wire shapes (usage, progress, options) are ``TypedDict``s with the TS keys;
SDK-built values are dataclasses.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from typing_extensions import NotRequired

from ..abort import AbortSignal

WorkflowStatus = Literal["ACCEPTED", "IN_PROGRESS", "COMPLETED", "FAILED", "CANCELED", "UNKNOWN"]


class WorkflowProgress(TypedDict, total=False):
    percent: float
    estimatedSecondsLeft: float


class ToolUsage(TypedDict):
    operationId: NotRequired[str]
    toolId: str
    price: float
    amount: float
    credits: float
    failed: NotRequired[bool]


class CreditUsage(TypedDict):
    toolId: NotRequired[str]
    details: NotRequired[list[ToolUsage]]
    credits: float
    balance: NotRequired[float]


class OptionsResult(TypedDict):
    toolId: NotRequired[str]
    usageAmount: NotRequired[float]
    credits: float
    originalCredits: NotRequired[float]
    estimated: NotRequired[bool]
    unit: NotRequired[str]
    price: NotRequired[float]
    hasPermission: NotRequired[bool]
    missingPermissions: NotRequired[list[str]]


@dataclass(frozen=True)
class WorkflowSubmitRequest:
    workflow: str
    payload: dict[str, Any]
    signal: AbortSignal | None = None


@dataclass(frozen=True)
class WorkflowJobHandle:
    workflow: str
    id: str


@dataclass(frozen=True)
class TransportResult:
    result: Any = None
    usage: CreditUsage | None = None
    raw: Any = None


@dataclass(frozen=True)
class TransportPollOptions:
    interval_ms: int | None = None
    max_attempts: int | None = None
    signal: AbortSignal | None = None
    on_progress: Callable[[WorkflowProgress], None] | None = None


@dataclass(frozen=True)
class WorkflowStatusResult:
    handle: WorkflowJobHandle
    status: WorkflowStatus
    result: Any = None
    usage: CreditUsage | None = None
    raw: Any = None
    progress: WorkflowProgress | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SdkTransport(Protocol):
    """How the SDK reaches the platform. Only ``execute`` is required.

    Optional methods, detected like TS ``typeof transport.x === 'function'``:
    ``submit(request) -> str``, ``poll(handle, options) -> TransportResult``,
    ``status(handle, signal) -> TransportResult``,
    ``options(workflow, payload) -> float | OptionsResult | None``.
    """

    def execute(self, request: WorkflowSubmitRequest) -> Awaitable[TransportResult]: ...


def has_method(obj: object, name: str) -> bool:
    return callable(getattr(obj, name, None))

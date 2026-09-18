"""Result models returned by ``ctx.distributed_map``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aws_durable_execution_sdk_python.exceptions import (
    DistributedMapError,
    ExecutionError,
)
from aws_durable_execution_sdk_python.config import (
    DistributedMapCompletionReason,
    DistributedMapItemStatus,
    DistributedMapStatus,
)


@dataclass(frozen=True)
class DistributedMapSummary:
    """Outcome of a map run without per-item results.

    Resolved by ``ctx.distributed_map`` for every terminal state. A non-``SUCCEEDED``
    run resolves with this summary rather than raising. Use
    :meth:`throw_if_error` to opt into raising.
    """

    status: DistributedMapStatus
    completion_reason: DistributedMapCompletionReason
    success_count: int
    failure_count: int
    unprocessed_count: int
    distributed_map_run_arn: str | None = None
    completion_details: str | None = None
    total_count: int | None = None

    @property
    def distributed_map_id(self) -> str | None:
        """Map-run id derived from the ARN, or ``None`` when the ARN is absent."""
        if not self.distributed_map_run_arn:
            return None
        # The run ARN appends /distributed-map-run/<id> to the parent execution's ARN.
        marker = "/distributed-map-run/"
        _, separator, run_id = self.distributed_map_run_arn.rpartition(marker)
        if not separator or not run_id:
            msg = (
                f"distributed_map_run_arn is missing the {marker} segment: "
                f"{self.distributed_map_run_arn}"
            )
            raise ExecutionError(msg)
        return run_id

    @property
    def has_failure(self) -> bool:
        """``True`` when any item permanently failed."""
        return self.failure_count > 0

    def throw_if_error(self) -> None:
        """Raise :class:`DistributedMapError` on any non-success outcome."""
        if self.status is not DistributedMapStatus.SUCCEEDED:
            detail = f", {self.completion_details}" if self.completion_details else ""
            msg = (
                f"Map run ended {self.status.value} "
                f"(reason: {self.completion_reason.value}{detail})"
            )
            raise DistributedMapError(msg)
        if self.failure_count > 0:
            msg = (
                f"Map run succeeded but {self.failure_count} item(s) permanently failed"
            )
            raise DistributedMapError(msg)


@dataclass(frozen=True)
class DistributedMapItemError:
    """Error for a single failed map run item."""

    error_type: str
    error_message: str


@dataclass(frozen=True)
class DistributedMapResultItem:
    """Outcome of a single map run item."""

    item_id: str
    status: DistributedMapItemStatus
    output: Any | None = None
    error: DistributedMapItemError | None = None


@dataclass(frozen=True)
class DistributedMapResult(DistributedMapSummary):
    """Outcome of a map run with per-item results.

    Resolved by ``ctx.distributed_map`` when inline result collection is enabled.
    Extends :class:`DistributedMapSummary` with the retained per-item results.
    """

    all: list[DistributedMapResultItem] = field(default_factory=list)

    def succeeded(self) -> list[DistributedMapResultItem]:
        """Return the items that succeeded."""
        return [
            item
            for item in self.all
            if item.status is DistributedMapItemStatus.SUCCEEDED
        ]

    def failed(self) -> list[DistributedMapResultItem]:
        """Return the items that permanently failed."""
        return [
            item for item in self.all if item.status is DistributedMapItemStatus.FAILED
        ]

    def get_results(self) -> list[Any]:
        """Return the outputs of the succeeded items."""
        return [
            item.output
            for item in self.all
            if item.status is DistributedMapItemStatus.SUCCEEDED
            and item.output is not None
        ]

    def get_errors(self) -> list[DistributedMapItemError]:
        """Return the errors of the failed items."""
        return [
            item.error
            for item in self.all
            if item.status is DistributedMapItemStatus.FAILED and item.error is not None
        ]

    def throw_if_error(self) -> None:
        """Raise the first failed item's error, otherwise defer to the summary rule."""
        if self.status is DistributedMapStatus.SUCCEEDED and self.failure_count > 0:
            failed = self.failed()
            if failed:
                first = failed[0]
                if first.error is not None:
                    msg = f"{first.error.error_type}: {first.error.error_message}"
                    raise DistributedMapError(msg)
                msg = f"item {first.item_id} failed"
                raise DistributedMapError(msg)
        super().throw_if_error()

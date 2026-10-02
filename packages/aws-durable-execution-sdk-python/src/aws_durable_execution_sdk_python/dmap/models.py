"""Result models returned by ``ctx.distributed_map``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from aws_durable_execution_sdk_python.exceptions import (
    DistributedMapError,
    ExecutionError,
    ValidationError,
)
from aws_durable_execution_sdk_python.config import (
    DistributedMapCompletionReason,
    DistributedMapItemStatus,
    DistributedMapStatus,
)
from aws_durable_execution_sdk_python.lambda_service import OperationStatus
from aws_durable_execution_sdk_python.types import DurableExecutionArn

if TYPE_CHECKING:
    from aws_durable_execution_sdk_python.lambda_service import (
        DistributedMapDetails,
        Operation,
    )

_STATUS_FOR_OPERATION_STATUS = {
    OperationStatus.SUCCEEDED: DistributedMapStatus.SUCCEEDED,
    OperationStatus.FAILED: DistributedMapStatus.FAILED,
    OperationStatus.STOPPED: DistributedMapStatus.STOPPED,
    OperationStatus.TIMED_OUT: DistributedMapStatus.TIMED_OUT,
}


def _to_distributed_map_status(operation: Operation | None) -> DistributedMapStatus:
    """Derive the map run status from the operation's terminal status."""
    op_status = operation.status if operation else None
    resolved = (
        _STATUS_FOR_OPERATION_STATUS.get(op_status) if op_status is not None else None
    )
    if resolved is None:
        msg = (
            "Cannot derive distributed map status from operation status "
            f"{op_status.value if op_status else 'UNKNOWN'}"
        )
        raise ExecutionError(msg)
    return resolved


def _resolve_terminal(
    operation: Operation | None,
) -> tuple[DistributedMapDetails, DistributedMapStatus, DistributedMapCompletionReason]:
    """Read the details, status and completion reason a terminal operation must carry."""
    details = operation.distributed_map_details if operation else None
    if details is None:
        msg = "DISTRIBUTED_MAP operation succeeded but carried no DistributedMapDetails"
        raise ExecutionError(msg)
    status = _to_distributed_map_status(operation)
    if details.completion_reason is None:
        msg = (
            f"DISTRIBUTED_MAP operation ended {status.value} but carried no "
            f"CompletionReason"
        )
        raise ExecutionError(msg)
    return details, status, details.completion_reason


_RUN_ARN_SEPARATOR = "/distributed-map-run/"


@dataclass(frozen=True)
class DistributedMapRunArn:
    """Parsed components of a distributed map run ARN.

    Attributes:
        durable_execution_arn: The parent durable execution's ARN.
        run_id: The map run id, unique per account.
    """

    durable_execution_arn: str
    run_id: str

    @classmethod
    def from_arn(cls, arn: str) -> DistributedMapRunArn | None:
        """Parse a map run ARN, returning ``None`` when it does not match the format."""
        parent, _, run_id = arn.partition(_RUN_ARN_SEPARATOR)
        if not run_id or "/" in run_id or DurableExecutionArn.from_arn(parent) is None:
            return None
        return cls(durable_execution_arn=parent, run_id=run_id)


@dataclass(frozen=True)
class DistributedMapSummary:
    """Outcome of a map run without per-item results.

    Resolved by ``ctx.distributed_map`` for every terminal state. A non-``SUCCEEDED``
    run resolves rather than raising. Use :meth:`throw_if_error` to opt into raising.
    """

    status: DistributedMapStatus
    completion_reason: DistributedMapCompletionReason
    success_count: int
    failure_count: int
    unprocessed_count: int
    distributed_map_run_arn: str | None = None
    completion_details: str | None = None
    total_count: int | None = None

    @classmethod
    def from_operation(cls, operation: Operation | None) -> DistributedMapSummary:
        """Rebuild the summary from a terminal DISTRIBUTED_MAP operation."""
        details, status, completion_reason = _resolve_terminal(operation)
        return cls(
            status=status,
            completion_reason=completion_reason,
            success_count=details.success_count,
            failure_count=details.failure_count,
            unprocessed_count=details.unprocessed_count,
            distributed_map_run_arn=details.distributed_map_run_arn,
            completion_details=details.completion_details,
            total_count=details.total_count,
        )

    @property
    def distributed_map_id(self) -> str | None:
        """Run id from :attr:`distributed_map_run_arn`, ``None`` if absent or unparseable."""
        if self.distributed_map_run_arn is None:
            return None
        parsed = DistributedMapRunArn.from_arn(self.distributed_map_run_arn)
        return parsed.run_id if parsed else None

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

    Resolved by ``ctx.distributed_map`` when passed a
    :class:`DistributedMapResultConfig`. Extends :class:`DistributedMapSummary`
    with the retained per-item results.
    """

    all: list[DistributedMapResultItem] = field(default_factory=list)

    @classmethod
    def from_operation(cls, operation: Operation | None) -> NoReturn:
        """Always raises. Use :meth:`from_operation_and_items` instead."""
        msg = (
            "DistributedMapResult carries per-item results, which from_operation "
            "cannot supply. Call DistributedMapResult.from_operation_and_items "
            "with the deserialized items, or DistributedMapSummary.from_operation "
            "for the counts alone."
        )
        raise ValidationError(msg)

    @classmethod
    def from_operation_and_items(
        cls, operation: Operation | None, items: list[DistributedMapResultItem]
    ) -> DistributedMapResult:
        """Rebuild the result from a terminal operation and its deserialized items."""
        details, status, completion_reason = _resolve_terminal(operation)
        return cls(
            status=status,
            completion_reason=completion_reason,
            success_count=details.success_count,
            failure_count=details.failure_count,
            unprocessed_count=details.unprocessed_count,
            distributed_map_run_arn=details.distributed_map_run_arn,
            completion_details=details.completion_details,
            total_count=details.total_count,
            all=items,
        )

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

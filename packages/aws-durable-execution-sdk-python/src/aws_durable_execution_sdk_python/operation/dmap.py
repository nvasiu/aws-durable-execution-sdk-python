"""Implement the Durable map run operation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from aws_durable_execution_sdk_python.concurrency.models import (
    DistributedMapItemError,
    DistributedMapResult,
    DistributedMapResultItem,
    DistributedMapSummary,
)
from aws_durable_execution_sdk_python.config import (
    DistributedMapProcessor,
    DistributedMapSource,
)
from aws_durable_execution_sdk_python.exceptions import (
    DistributedMapError,
    ExecutionError,
    ValidationError,
)
from aws_durable_execution_sdk_python.lambda_service import (
    DistributedMapOptions,
    DistributedMapReaderConfigWire,
    DistributedMapResultCollectionMode,
    DistributedMapResultCollectionWire,
    DistributedMapSourceType,
    DistributedMapSourceWire,
    OperationUpdate,
)
from aws_durable_execution_sdk_python.operation.base import (
    CheckResult,
    OperationExecutor,
)
from aws_durable_execution_sdk_python.serdes import (
    DEFAULT_JSON_SERDES,
    deserialize,
    serialize,
)
from aws_durable_execution_sdk_python.suspend import suspend_with_optional_resume_delay

if TYPE_CHECKING:
    from collections.abc import Sequence

    from aws_durable_execution_sdk_python.config import (
        DistributedMapConfig,
    )
    from aws_durable_execution_sdk_python.identifier import OperationIdentifier
    from aws_durable_execution_sdk_python.lambda_service import DistributedMapDetails
    from aws_durable_execution_sdk_python.state import (
        CheckpointedResult,
        ExecutionState,
    )

logger = logging.getLogger(__name__)

# Size limits for the inline item list (1 MB) and the reader's saved state (32 KB).
_INLINE_SIZE_LIMIT = 1024 * 1024
_READER_STATE_LIMIT = 32 * 1024


def _inline_items_to_wire(
    items: tuple[Any, ...],
    serdes: Any,
    operation_id: str,
    durable_execution_arn: str,
) -> tuple[Any, ...]:
    """Serialize each inline item to its wire JSON value, enforcing the 1 MB cap."""
    wire_items: list[Any] = []
    for item in items:
        serialized = serialize(
            serdes=serdes,
            value=item,
            operation_id=operation_id,
            durable_execution_arn=durable_execution_arn,
        )
        try:
            wire_items.append(json.loads(serialized))
        except json.JSONDecodeError as e:
            msg = "inline source serdes must produce a JSON value for each item"
            raise ValidationError(msg) from e
    total = len(json.dumps(wire_items, separators=(",", ":")).encode("utf-8"))
    if total > _INLINE_SIZE_LIMIT:
        msg = (
            f"inline source exceeds the {_INLINE_SIZE_LIMIT // 1024 // 1024} MB limit "
            f"(serialized size: {total} bytes)"
        )
        raise ValidationError(msg)
    return tuple(wire_items)


def _source_to_wire(
    source: DistributedMapSource | Sequence[Any],
    operation_id: str,
    durable_execution_arn: str,
) -> DistributedMapSourceWire:
    """Translate a source (typed or plain-list shorthand) into its wire form."""
    if not isinstance(source, DistributedMapSource):
        # A plain list is treated as an inline source with the default serializer.
        wire_items = _inline_items_to_wire(
            tuple(source), DEFAULT_JSON_SERDES, operation_id, durable_execution_arn
        )
        return DistributedMapSourceWire(
            source_type=DistributedMapSourceType.INLINE, inline_items=wire_items
        )

    if source.source_type is DistributedMapSourceType.INLINE:
        wire_items = _inline_items_to_wire(
            source.inline_items or (),
            source.inline_serdes or DEFAULT_JSON_SERDES,
            operation_id,
            durable_execution_arn,
        )
        return DistributedMapSourceWire(
            source_type=DistributedMapSourceType.INLINE,
            inline_items=wire_items,
            max_items=source.max_items,
        )
    if source.source_type is DistributedMapSourceType.S3 and source.s3 is not None:
        return DistributedMapSourceWire(
            source_type=DistributedMapSourceType.S3,
            max_items=source.max_items,
            s3_config=source.s3.to_wire(),
        )
    if (
        source.source_type is DistributedMapSourceType.READER_FUNCTION
        and source.reader is not None
    ):
        reader_config = DistributedMapReaderConfigWire(
            function_name=source.reader.function_name
        )
        if source.reader.initial_state is not None:
            state = serialize(
                serdes=source.reader.state_serdes or DEFAULT_JSON_SERDES,
                value=source.reader.initial_state,
                operation_id=operation_id,
                durable_execution_arn=durable_execution_arn,
            )
            if len(state.encode("utf-8")) > _READER_STATE_LIMIT:
                msg = (
                    f"reader initial_state exceeds the "
                    f"{_READER_STATE_LIMIT // 1024} KB limit"
                )
                raise ValidationError(msg)
            reader_config = DistributedMapReaderConfigWire(
                function_name=source.reader.function_name, initial_state=state
            )
        return DistributedMapSourceWire(
            source_type=DistributedMapSourceType.READER_FUNCTION,
            max_items=source.max_items,
            reader_config=reader_config,
        )
    msg = f"Unsupported map run source type: {source.source_type}"
    raise ExecutionError(msg)


def _build_distributed_map_options(
    source: DistributedMapSource | Sequence[Any],
    processor: DistributedMapProcessor,
    max_concurrency: int,
    config: DistributedMapConfig,
    operation_id: str,
    durable_execution_arn: str,
) -> DistributedMapOptions:
    """Assemble the wire options payload from the operands and config."""
    result_collection = (
        DistributedMapResultCollectionWire(
            mode=DistributedMapResultCollectionMode.INLINE
        )
        if config.collect_results
        else None
    )
    return DistributedMapOptions(
        max_concurrency=max_concurrency,
        source=_source_to_wire(source, operation_id, durable_execution_arn),
        processor=processor.to_wire(),
        destination=(
            config.destination.to_wire() if config.destination is not None else None
        ),
        completion_config=(
            config.completion_config.to_wire()
            if config.completion_config is not None
            else None
        ),
        result_collection=result_collection,
        timeout_seconds=config.timeout.to_seconds()
        if config.timeout is not None
        else None,
    )


def _summary_fields(details: DistributedMapDetails) -> dict[str, Any]:
    """Shared summary fields extracted from the terminal details block."""
    return {
        "status": details.status,
        "completion_reason": details.completion_reason,
        "success_count": details.success_count,
        "failure_count": details.failure_count,
        "unprocessed_count": details.unprocessed_count,
        "distributed_map_run_arn": details.distributed_map_run_arn,
        "completion_details": details.completion_details,
        "total_count": details.total_count,
    }


class DistributedMapOperationExecutor(OperationExecutor[DistributedMapSummary]):
    """Executor for map run operations.

    Creates the START checkpoint if none exists, then suspends until the
    backend completes the run and re-invokes the parent. On resume, a
    ``DistributedMapSummary`` (or ``DistributedMapResult`` when result collection is enabled)
    is built from the checkpointed ``DistributedMapDetails``.
    """

    def __init__(
        self,
        source: DistributedMapSource | Sequence[Any],
        processor: DistributedMapProcessor,
        max_concurrency: int,
        state: ExecutionState,
        operation_identifier: OperationIdentifier,
        config: DistributedMapConfig,
    ):
        """Initialize the map run operation executor.

        Args:
            source: The items to process (typed source or plain-list shorthand)
            processor: The processor configuration
            max_concurrency: Maximum concurrent processor invocations
            state: The execution state
            operation_identifier: The operation identifier
            config: Configuration for the map run operation
        """
        self.source = source
        self.processor = processor
        self.max_concurrency = max_concurrency
        self.state = state
        self.operation_identifier = operation_identifier
        self.config = config

    def _resolve_summary(
        self, details: DistributedMapDetails | None
    ) -> DistributedMapSummary:
        """Reconstruct the resolved summary/result from the terminal details."""
        if details is None:
            msg = "DISTRIBUTED_MAP operation succeeded but carried no DistributedMapDetails"
            raise ExecutionError(msg)
        fields = _summary_fields(details)
        if not self.config.collect_results:
            return DistributedMapSummary(**fields)
        return DistributedMapResult(**fields, all=self._deserialize_items(details))

    def _deserialize_items(
        self, details: DistributedMapDetails
    ) -> list[DistributedMapResultItem]:
        """Deserialize the per-item wire results into customer result items."""
        items: list[DistributedMapResultItem] = []
        for wire in details.results or ():
            output: Any | None = None
            if wire.output is not None:
                # Output is already a JSON value; re-dump to text for the serdes.
                output = deserialize(
                    serdes=self.config.result_serdes or DEFAULT_JSON_SERDES,
                    data=json.dumps(wire.output),
                    operation_id=self.operation_identifier.operation_id,
                    durable_execution_arn=self.state.durable_execution_arn,
                )
            error = (
                DistributedMapItemError(
                    error_type=wire.error.type or "",
                    error_message=wire.error.message or "",
                )
                if wire.error is not None
                else None
            )
            items.append(
                DistributedMapResultItem(
                    item_id=wire.item_id,
                    status=wire.status,
                    output=output,
                    error=error,
                )
            )
        return items

    def check_result_status(self) -> CheckResult[DistributedMapSummary]:
        """Check operation status and create the START checkpoint if needed.

        Called twice by process() when creating synchronous checkpoints: once before
        and once after, to detect if the operation completed immediately.

        Returns:
            CheckResult indicating the next action to take

        Raises:
            SuspendExecution: For STARTED operations waiting for completion
        """
        checkpointed_result: CheckpointedResult = self.state.get_checkpoint_result(
            self.operation_identifier.operation_id
        )

        # Terminal success - build the summary/result from DistributedMapDetails
        if checkpointed_result.is_succeeded():
            operation = checkpointed_result.operation
            summary = self._resolve_summary(
                operation.distributed_map_details if operation else None
            )
            return CheckResult.create_completed(summary)

        # Operation-level terminal failure
        if (
            checkpointed_result.is_failed()
            or checkpointed_result.is_timed_out()
            or checkpointed_result.is_stopped()
        ):
            msg = (
                f"Distributed map operation "
                f"'{self.operation_identifier.name or self.operation_identifier.operation_id}' "
                f"ended with status "
                f"{checkpointed_result.status.value if checkpointed_result.status else 'UNKNOWN'}"
            )
            checkpointed_result.raise_operation_error(DistributedMapError, msg=msg)

        # Started - ready to suspend
        if checkpointed_result.is_started():
            logger.debug(
                "⏳ Map run %s still in progress, will suspend",
                self.operation_identifier.name
                or self.operation_identifier.operation_id,
            )
            return CheckResult.create_is_ready_to_execute(checkpointed_result)

        # Create START checkpoint if not exists
        if not checkpointed_result.is_existent():
            start_operation: OperationUpdate = (
                OperationUpdate.create_distributed_map_start(
                    identifier=self.operation_identifier,
                    distributed_map_options=_build_distributed_map_options(
                        source=self.source,
                        processor=self.processor,
                        max_concurrency=self.max_concurrency,
                        config=self.config,
                        operation_id=self.operation_identifier.operation_id,
                        durable_execution_arn=self.state.durable_execution_arn,
                    ),
                )
            )
            # Checkpoint map run START with blocking (is_sync=True).
            # Must ensure the map run is recorded before suspending execution.
            self.state.create_checkpoint(operation_update=start_operation, is_sync=True)

            logger.debug(
                "🚀 Map run %s started, will check for immediate completion",
                self.operation_identifier.name
                or self.operation_identifier.operation_id,
            )

            # Signal to process() that checkpoint was created - to recheck status
            # for immediate completion before proceeding.
            return CheckResult.create_started()

        # Ready to suspend (checkpoint exists but not in a terminal or started state)
        return CheckResult.create_is_ready_to_execute(checkpointed_result)

    def execute(
        self, _checkpointed_result: CheckpointedResult
    ) -> DistributedMapSummary:
        """Execute map run operation by suspending to wait for async completion.

        The map run operation doesn't execute synchronously - it suspends and
        the backend runs the map run asynchronously.

        Args:
            checkpointed_result: The checkpoint data (unused, but required by interface)

        Returns:
            Never returns - always suspends

        Raises:
            Always suspends via suspend_with_optional_resume_delay
            ExecutionError: If suspend doesn't raise (should never happen)
        """
        msg: str = f"Map run {self.operation_identifier.operation_id} started, suspending for completion"
        suspend_with_optional_resume_delay(msg)
        # This line should never be reached since suspend_with_optional_resume_delay always raises
        error_msg: str = "suspend_with_optional_resume_delay should have raised an exception, but did not."
        raise ExecutionError(error_msg) from None

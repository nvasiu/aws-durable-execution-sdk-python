"""Implement the Durable map run operation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from aws_durable_execution_sdk_python.config import (
    DistributedMapProcessor,
    DistributedMapResultConfig,
    DistributedMapSource,
    DistributedMapSourceFormat,
    DistributedMapStatus,
    ProcessorResponseMode,
)
from aws_durable_execution_sdk_python.dmap.models import (
    DistributedMapItemError,
    DistributedMapResult,
    DistributedMapResultItem,
    DistributedMapSummary,
)
from aws_durable_execution_sdk_python.exceptions import (
    ExecutionError,
    ValidationError,
)
from aws_durable_execution_sdk_python.lambda_service import (
    DistributedMapCompletionConfig,
    DistributedMapCsvFormatOptions,
    DistributedMapCsvHeaderLocation,
    DistributedMapDestinationConfig,
    DistributedMapDestinationInclude,
    DistributedMapDestinationType,
    DistributedMapFunctionResponseType,
    DistributedMapInlineSourceConfig,
    DistributedMapOnFailureConfig,
    DistributedMapOnSuccessConfig,
    DistributedMapOptions,
    DistributedMapProcessorConfig,
    DistributedMapReaderFunctionSourceConfig,
    DistributedMapResultCollectionConfig,
    DistributedMapResultCollectionMode,
    DistributedMapS3DestinationConfig,
    DistributedMapS3SourceConfig,
    DistributedMapS3SourceTransform,
    DistributedMapSourceConfig,
    DistributedMapSourceType,
    OperationStatus,
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
    from aws_durable_execution_sdk_python.lambda_service import (
        DistributedMapDetails,
        Operation,
    )
    from aws_durable_execution_sdk_python.state import (
        CheckpointedResult,
        ExecutionState,
    )

logger = logging.getLogger(__name__)

# Size limits for the inline item list (1 MB) and the reader's saved state (32 KB).
_INLINE_SIZE_LIMIT = 1024 * 1024
_READER_STATE_LIMIT = 32 * 1024
_UNLIMITED_RETRY_WIRE = -1

_RESPONSE_TYPE_FOR_MODE = {
    ProcessorResponseMode.ITEM_FAILURES: DistributedMapFunctionResponseType.REPORT_BATCH_ITEM_FAILURES,
    ProcessorResponseMode.ITEM_RESULTS: DistributedMapFunctionResponseType.REPORT_BATCH_ITEM_RESULTS,
}


def _build_inline_items(
    items: tuple[Any, ...],
    serdes: Any,
    operation_id: str,
    durable_execution_arn: str,
) -> tuple[str, ...]:
    """Serialize each inline item, enforcing the 1 MB cap on the whole list."""
    serialized_items: list[str] = []
    for item in items:
        serialized_items.append(
            serialize(
                serdes=serdes,
                value=item,
                operation_id=operation_id,
                durable_execution_arn=durable_execution_arn,
            )
        )
    total = len(json.dumps(serialized_items, separators=(",", ":")).encode("utf-8"))
    if total > _INLINE_SIZE_LIMIT:
        msg = (
            f"inline source exceeds the {_INLINE_SIZE_LIMIT // 1024 // 1024} MB limit "
            f"(serialized size: {total} bytes)"
        )
        raise ValidationError(msg)
    return tuple(serialized_items)


def _build_completion_config(
    completion_config: Any,
) -> DistributedMapCompletionConfig | None:
    """Translate the completion config, or None when it carries no threshold."""
    if (
        completion_config.tolerated_failure_count is None
        and completion_config.tolerated_failure_percentage is None
        and completion_config.minimum_sample_size is None
    ):
        return None
    return DistributedMapCompletionConfig(
        tolerated_failure_count=completion_config.tolerated_failure_count,
        tolerated_failure_percentage=completion_config.tolerated_failure_percentage,
        minimum_sample_size=completion_config.minimum_sample_size,
    )


def _build_processor_config(
    processor: DistributedMapProcessor,
) -> DistributedMapProcessorConfig:
    """Translate the processor, mapping the response mode and unlimited retries."""
    response_type = _RESPONSE_TYPE_FOR_MODE.get(processor.response_mode)
    max_retry_attempts: int | None = None
    max_retry_duration_seconds: int | None = None
    attempts = processor.max_retry_attempts
    if attempts == DistributedMapProcessor.UNLIMITED:
        max_retry_attempts = _UNLIMITED_RETRY_WIRE
    elif isinstance(attempts, int):
        max_retry_attempts = attempts
    if processor.max_retry_duration is not None:
        max_retry_duration_seconds = processor.max_retry_duration.to_seconds()
    return DistributedMapProcessorConfig(
        function_name=processor.function_name,
        function_response_types=(response_type,) if response_type else None,
        batch_size=processor.batch_size,
        max_retry_attempts=max_retry_attempts,
        max_retry_duration_seconds=max_retry_duration_seconds,
        durable_execution_name_prefix=processor.durable_execution_name_prefix,
    )


def _transform_for(s3: Any) -> DistributedMapS3SourceTransform | None:
    """A prefix source flattens when a format is set, and lists keys when it is not."""
    if s3.prefix is None:
        return None
    if s3.fmt is None:
        return DistributedMapS3SourceTransform.NONE
    return DistributedMapS3SourceTransform.LOAD_AND_FLATTEN


def _build_s3_source_config(s3: Any) -> DistributedMapS3SourceConfig:
    """Translate the S3 source, deriving the transform and CSV header location."""
    csv_format_options: DistributedMapCsvFormatOptions | None = None
    if s3.fmt is DistributedMapSourceFormat.CSV:
        csv_format_options = DistributedMapCsvFormatOptions(
            header_location=(
                DistributedMapCsvHeaderLocation.GIVEN
                if s3.headers is not None
                else DistributedMapCsvHeaderLocation.FIRST_ROW
            ),
            headers=s3.headers,
            delimiter=s3.delimiter,
        )
    return DistributedMapS3SourceConfig(
        bucket=s3.bucket,
        key=s3.key,
        key_prefix=s3.prefix,
        transform=_transform_for(s3),
        expected_bucket_owner=s3.expected_bucket_owner,
        fmt=s3.fmt,
        csv_format_options=csv_format_options,
    )


def _destination_entry_fields(
    destination: Any, *, second_include: DistributedMapDestinationInclude
) -> dict[str, Any]:
    """Build the members the OnSuccess and OnFailure shapes share."""
    include: list[DistributedMapDestinationInclude] = []
    if destination.include_input:
        include.append(DistributedMapDestinationInclude.INPUT)
    if second_include is DistributedMapDestinationInclude.OUTPUT:
        if destination.include_output:
            include.append(second_include)
    elif destination.include_error:
        include.append(second_include)
    return {
        "type": DistributedMapDestinationType.S3,
        "include": tuple(include),
        "s3_destination_config": DistributedMapS3DestinationConfig(
            bucket=destination.bucket,
            key_prefix=destination.prefix,
            expected_bucket_owner=destination.expected_bucket_owner,
        ),
    }


def _build_destination_config(
    destination: Any,
) -> DistributedMapDestinationConfig | None:
    """Translate the destination config, or None when neither side is set."""
    on_success = (
        DistributedMapOnSuccessConfig(
            **_destination_entry_fields(
                destination.on_success,
                second_include=DistributedMapDestinationInclude.OUTPUT,
            )
        )
        if destination.on_success is not None
        else None
    )
    on_failure = (
        DistributedMapOnFailureConfig(
            **_destination_entry_fields(
                destination.on_failure,
                second_include=DistributedMapDestinationInclude.ERROR,
            )
        )
        if destination.on_failure is not None
        else None
    )
    if on_success is None and on_failure is None:
        return None
    return DistributedMapDestinationConfig(on_success=on_success, on_failure=on_failure)


def _build_source_config(
    source: DistributedMapSource | Sequence[Any],
    operation_id: str,
    durable_execution_arn: str,
) -> DistributedMapSourceConfig:
    """Translate a source (typed or plain-list shorthand) into its source config."""
    if not isinstance(source, DistributedMapSource):
        # A plain list is treated as an inline source with the default serializer.
        serialized_items = _build_inline_items(
            tuple(source), DEFAULT_JSON_SERDES, operation_id, durable_execution_arn
        )
        return DistributedMapSourceConfig(
            source_type=DistributedMapSourceType.INLINE,
            inline_source_config=DistributedMapInlineSourceConfig(
                items=serialized_items
            ),
        )

    if source.inline_items is not None:
        serialized_items = _build_inline_items(
            source.inline_items,
            source.inline_serdes or DEFAULT_JSON_SERDES,
            operation_id,
            durable_execution_arn,
        )
        return DistributedMapSourceConfig(
            source_type=DistributedMapSourceType.INLINE,
            inline_source_config=DistributedMapInlineSourceConfig(
                items=serialized_items
            ),
            max_items=source.max_items,
        )
    if source.s3 is not None:
        return DistributedMapSourceConfig(
            source_type=DistributedMapSourceType.S3,
            max_items=source.max_items,
            s3_config=_build_s3_source_config(source.s3),
        )
    if source.reader is not None:
        reader_config = DistributedMapReaderFunctionSourceConfig(
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
            reader_config = DistributedMapReaderFunctionSourceConfig(
                function_name=source.reader.function_name, initial_state=state
            )
        return DistributedMapSourceConfig(
            source_type=DistributedMapSourceType.READER_FUNCTION,
            max_items=source.max_items,
            reader_config=reader_config,
        )
    msg = "Distributed map source has no configured items"
    raise ExecutionError(msg)


def _build_distributed_map_options(
    source: DistributedMapSource | Sequence[Any],
    processor: DistributedMapProcessor,
    max_concurrency: int,
    config: DistributedMapConfig,
    operation_id: str,
    durable_execution_arn: str,
) -> DistributedMapOptions:
    """Assemble the DistributedMapOptions payload from the operands and config."""
    result_collection = (
        DistributedMapResultCollectionConfig(
            mode=DistributedMapResultCollectionMode.INLINE
        )
        if isinstance(config, DistributedMapResultConfig)
        else None
    )
    return DistributedMapOptions(
        max_concurrency=max_concurrency,
        source=_build_source_config(source, operation_id, durable_execution_arn),
        processor=_build_processor_config(processor),
        destination=(
            _build_destination_config(config.destination)
            if config.destination is not None
            else None
        ),
        completion_config=(
            _build_completion_config(config.completion_config)
            if config.completion_config is not None
            else None
        ),
        result_collection=result_collection,
        timeout_seconds=config.timeout.to_seconds()
        if config.timeout is not None
        else None,
    )


def _distributed_map_status_from_operation(
    operation: Operation | None,
) -> DistributedMapStatus:
    """Derive the map run status from the operation's terminal status."""
    op_status = operation.status if operation else None
    resolved = (
        {
            OperationStatus.SUCCEEDED: DistributedMapStatus.SUCCEEDED,
            OperationStatus.FAILED: DistributedMapStatus.FAILED,
            OperationStatus.STOPPED: DistributedMapStatus.STOPPED,
            OperationStatus.TIMED_OUT: DistributedMapStatus.TIMED_OUT,
        }.get(op_status)
        if op_status is not None
        else None
    )
    if resolved is None:
        msg = (
            "Cannot derive distributed map status from operation status "
            f"{op_status.value if op_status else 'UNKNOWN'}"
        )
        raise ExecutionError(msg)
    return resolved


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

    def _resolve_summary(self, operation: Operation | None) -> DistributedMapSummary:
        """Reconstruct the resolved summary/result from the terminal operation."""
        details = operation.distributed_map_details if operation else None
        if details is None:
            msg = "DISTRIBUTED_MAP operation succeeded but carried no DistributedMapDetails"
            raise ExecutionError(msg)
        status = _distributed_map_status_from_operation(operation)
        completion_reason = details.completion_reason
        if completion_reason is None:
            msg = (
                f"DISTRIBUTED_MAP operation ended {status.value} but carried no "
                f"CompletionReason"
            )
            raise ExecutionError(msg)
        if not isinstance(self.config, DistributedMapResultConfig):
            return DistributedMapSummary(
                status=status,
                completion_reason=completion_reason,
                success_count=details.success_count,
                failure_count=details.failure_count,
                unprocessed_count=details.unprocessed_count,
                distributed_map_run_arn=details.distributed_map_run_arn,
                completion_details=details.completion_details,
                total_count=details.total_count,
            )
        return DistributedMapResult(
            status=status,
            completion_reason=completion_reason,
            success_count=details.success_count,
            failure_count=details.failure_count,
            unprocessed_count=details.unprocessed_count,
            distributed_map_run_arn=details.distributed_map_run_arn,
            completion_details=details.completion_details,
            total_count=details.total_count,
            all=self._deserialize_items(details, self.config.result_serdes),
        )

    def _deserialize_items(
        self, details: DistributedMapDetails, result_serdes: Any
    ) -> list[DistributedMapResultItem]:
        """Deserialize the service result items into customer result items."""
        items: list[DistributedMapResultItem] = []
        for entry in details.results or ():
            output: Any | None = None
            if entry.output is not None:
                output = deserialize(
                    serdes=result_serdes or DEFAULT_JSON_SERDES,
                    data=entry.output,
                    operation_id=self.operation_identifier.operation_id,
                    durable_execution_arn=self.state.durable_execution_arn,
                )
            error = (
                DistributedMapItemError(
                    error_type=entry.error.type or "",
                    error_message=entry.error.message or "",
                )
                if entry.error is not None
                else None
            )
            items.append(
                DistributedMapResultItem(
                    item_id=entry.item_id,
                    status=entry.status,
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

        # Terminal success - build the summary/result from the operation
        if checkpointed_result.is_succeeded():
            operation = checkpointed_result.operation
            summary = self._resolve_summary(operation)
            return CheckResult.create_completed(summary)

        # Operation-level terminal failure. Every terminal state resolves with the
        # summary, and throw_if_error opts into raising.
        if (
            checkpointed_result.is_failed()
            or checkpointed_result.is_timed_out()
            or checkpointed_result.is_stopped()
        ):
            operation = checkpointed_result.operation
            if operation is not None and operation.distributed_map_details is not None:
                return CheckResult.create_completed(self._resolve_summary(operation))
            status_value = (
                checkpointed_result.status.value
                if checkpointed_result.status
                else "UNKNOWN"
            )
            msg = (
                f"DISTRIBUTED_MAP operation ended {status_value} but carried no "
                f"DistributedMapDetails"
            )
            raise ExecutionError(msg)

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

"""Integration tests for the durable distributed map authoring wrappers.

The durable variants are ``@durable_execution`` handlers; they run through the
full invocation harness with a mocked checkpoint backend.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from unittest.mock import Mock, patch

from aws_durable_execution_sdk_python.dmap.handlers import (
    durable_distributed_map_batch_handler,
    durable_distributed_map_item_handler,
)
from aws_durable_execution_sdk_python.exceptions import DurableOperationError
from aws_durable_execution_sdk_python.execution import (
    InvocationStatus,
    durable_execution,  # noqa: F401  (ensures decorator import path is valid)
)
from aws_durable_execution_sdk_python.lambda_service import (
    CheckpointOutput,
    CheckpointUpdatedExecutionState,
    Operation,
    OperationStatus,
    OperationType,
)
from aws_durable_execution_sdk_python.serdes import SerDes, SerDesContext

_ARN = "arn:aws:lambda:us-east-1:123456789012:function:proc:1/durable-execution/execution-1"


def _lambda_context():
    ctx = Mock()
    ctx.aws_request_id = "test-request-id"
    ctx.client_context = None
    ctx.identity = None
    ctx._epoch_deadline_time_in_ms = 0  # noqa: SLF001
    ctx.invoked_function_arn = "test-arn"
    ctx.tenant_id = None
    return ctx


def _event(records: list[dict]):
    return {
        "DurableExecutionArn": _ARN,
        "CheckpointToken": "test-token",
        "InitialExecutionState": {
            "Operations": [
                {
                    "Id": "execution-1",
                    "Type": "EXECUTION",
                    "Status": "STARTED",
                    "ExecutionDetails": {
                        "InputPayload": json.dumps({"records": records})
                    },
                }
            ],
            "NextMarker": "",
        },
        "LocalRunner": True,
    }


def _run(handler, records: list[dict]):
    operations = [
        Operation(
            operation_id="execution-1",
            operation_type=OperationType.EXECUTION,
            status=OperationStatus.STARTED,
        )
    ]

    def mock_checkpoint(
        durable_execution_arn, checkpoint_token, updates, client_token="token"
    ):  # noqa: S107
        for update in updates:
            operations.append(
                Operation(
                    operation_id=update.operation_id,
                    operation_type=update.operation_type,
                    status=OperationStatus.STARTED,
                    parent_id=update.parent_id,
                )
            )
        return CheckpointOutput(
            checkpoint_token="new_token",  # noqa: S106
            new_execution_state=CheckpointUpdatedExecutionState(
                operations=operations.copy()
            ),
        )

    with patch(
        "aws_durable_execution_sdk_python.execution.LambdaClient"
    ) as mock_client_class:
        mock_client = Mock()
        mock_client_class.initialize_client.return_value = mock_client
        mock_client.checkpoint = mock_checkpoint
        return handler(_event(records), _lambda_context())


def test_durable_item_handler_reports_results():
    handler = durable_distributed_map_item_handler(lambda _ctx, item: item * 2)
    result = _run(handler, [{"itemId": "0", "body": "2"}, {"itemId": "1", "body": "3"}])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])
    assert data["batchItemResults"] == [
        {"itemIdentifier": "0", "output": "4"},
        {"itemIdentifier": "1", "output": "6"},
    ]
    assert data["batchItemFailures"] == []


def test_durable_item_handler_captures_failure():
    def process(_ctx, item):
        if item == "bad":
            msg = "boom"
            raise ValueError(msg)
        return item

    handler = durable_distributed_map_item_handler(process)
    result = _run(
        handler, [{"itemId": "0", "body": '"ok"'}, {"itemId": "1", "body": '"bad"'}]
    )
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])
    assert data["batchItemResults"] == [{"itemIdentifier": "0", "output": '"ok"'}]
    assert data["batchItemFailures"][0]["itemIdentifier"] == "1"
    assert data["batchItemFailures"][0]["error"]["errorType"] == "ValueError"


def test_durable_item_handler_forwards_every_error_member():
    """The branch error's data and stack trace reach the service, not just type and message."""

    def process(_ctx, _item):
        raise DurableOperationError(
            message="boom",
            error_type="Inner",
            data="payload",
            stack_trace=["frame one", "frame two"],
        )

    handler = durable_distributed_map_item_handler(process)
    result = _run(handler, [{"itemId": "0", "body": "1"}])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])

    error = data["batchItemFailures"][0]["error"]
    # A branch error is checkpointed and rebuilt, which normalizes errorType to
    # the raised class's qualified name.
    assert error["errorType"] == (
        f"{DurableOperationError.__module__}.{DurableOperationError.__qualname__}"
    )
    assert error["errorMessage"] == "boom"
    assert error["errorData"] == "payload"
    assert error["stackTrace"] == ["frame one", "frame two"]


def test_durable_batch_handler_records_no_output():
    """A batch processor reports no output, so the return value is not forwarded."""
    seen: list = []

    def process(_ctx, items):
        seen.extend(items)
        return {"count": len(items)}

    handler = durable_distributed_map_batch_handler(process)
    result = _run(handler, [{"itemId": "0", "body": "1"}, {"itemId": "1", "body": "2"}])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    assert json.loads(result["Result"]) is None
    assert seen == [1, 2]


def test_durable_item_handler_fails_only_the_item_whose_output_cannot_be_serialized():
    """A result serdes failure is that item's failure, not the batch's."""

    class _RejectsBad(SerDes[Any]):
        def serialize(self, value: Any, _serdes_context: SerDesContext) -> str:
            if value == "bad":
                msg = "cannot serialize"
                raise TypeError(msg)
            return json.dumps(value)

        def deserialize(self, data: str, _serdes_context: SerDesContext) -> Any:
            return json.loads(data)

    handler = durable_distributed_map_item_handler(
        lambda _ctx, item: item, result_serdes=_RejectsBad()
    )
    result = _run(
        handler, [{"itemId": "0", "body": '"ok"'}, {"itemId": "1", "body": '"bad"'}]
    )
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])
    assert data["batchItemResults"] == [{"itemIdentifier": "0", "output": '"ok"'}]
    assert data["batchItemFailures"] == [
        {
            "itemIdentifier": "1",
            "error": {"errorType": "TypeError", "errorMessage": "cannot serialize"},
        }
    ]


def test_durable_item_handler_omits_the_output_of_a_handler_returning_none():
    handler = durable_distributed_map_item_handler(lambda _ctx, _item: None)
    result = _run(handler, [{"itemId": "0", "body": "1"}])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])
    assert data == {
        "batchItemFailures": [],
        "batchItemResults": [{"itemIdentifier": "0"}],
    }


def test_durable_item_handler_runs_items_one_at_a_time_by_default():
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def process(_ctx, item):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Hold the branch so that a wider bound would overlap this item with the
        # next one, which is what makes the peak below meaningful.
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return item

    handler = durable_distributed_map_item_handler(process)
    result = _run(handler, [{"itemId": str(i), "body": "1"} for i in range(4)])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    assert peak == 1


def test_durable_item_handler_honors_requested_concurrency():
    # Each item blocks until all four have arrived, so the batch only finishes
    # if the map really runs four items at once. At a smaller bound the first
    # item waits out the timeout and every item fails.
    barrier = threading.Barrier(4, timeout=5)

    def process(_ctx, item):
        barrier.wait()
        return item

    handler = durable_distributed_map_item_handler(process, concurrency=4)
    result = _run(handler, [{"itemId": str(i), "body": str(i)} for i in range(4)])
    assert result["Status"] == InvocationStatus.SUCCEEDED.value
    data = json.loads(result["Result"])
    assert data["batchItemFailures"] == []
    assert data["batchItemResults"] == [
        {"itemIdentifier": "0", "output": "0"},
        {"itemIdentifier": "1", "output": "1"},
        {"itemIdentifier": "2", "output": "2"},
        {"itemIdentifier": "3", "output": "3"},
    ]

"""Unit tests for the non-durable distributed map authoring wrappers."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from aws_durable_execution_sdk_python.config import (
    DistributedMapConfig,
    DistributedMapProcessor,
    DistributedMapStatus,
)
from aws_durable_execution_sdk_python.dmap.handlers import (
    ReaderPage,
    distributed_map_batch_handler,
    distributed_map_item_handler,
    durable_distributed_map_item_handler,
    distributed_map_reader,
)
from aws_durable_execution_sdk_python.exceptions import ExecutionError, ValidationError
from aws_durable_execution_sdk_python.lambda_service import (
    DistributedMapCompletionReason,
    DistributedMapDetails,
    Operation,
    OperationStatus,
    OperationType,
)
from aws_durable_execution_sdk_python.operation.dmap import (
    DistributedMapOperationExecutor,
    _distributed_map_status_from_operation,
)


def _terminal_operation(
    status: OperationStatus, details: DistributedMapDetails
) -> Operation:
    return Operation(
        operation_id="dmap-id",
        operation_type=OperationType.DISTRIBUTED_MAP,
        status=status,
        distributed_map_details=details,
    )


def test_status_derived_from_operation_when_details_omit_status():
    """Status comes from the operation, not from details."""
    details = DistributedMapDetails(
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=3,
        total_count=3,
    )
    op = _terminal_operation(OperationStatus.SUCCEEDED, details)
    assert _distributed_map_status_from_operation(op) is DistributedMapStatus.SUCCEEDED


def test_status_from_operation_rejects_non_terminal():
    """A non-terminal operation status cannot map to a DistributedMapStatus."""
    op = _terminal_operation(
        OperationStatus.STARTED,
        DistributedMapDetails(success_count=0),
    )
    with pytest.raises(ExecutionError, match="Cannot derive distributed map status"):
        _distributed_map_status_from_operation(op)


def test_resolved_summary_uses_operation_status_not_details():
    """_resolve_summary carries the derived status when details omit it."""
    details = DistributedMapDetails(
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=2,
        failure_count=0,
        unprocessed_count=0,
        total_count=2,
    )
    executor = DistributedMapOperationExecutor(
        source=["a", "b"],
        processor=DistributedMapProcessor.batch("proc"),
        max_concurrency=1,
        state=MagicMock(),
        operation_identifier=MagicMock(),
        config=DistributedMapConfig(),
    )
    summary = executor._resolve_summary(
        _terminal_operation(OperationStatus.SUCCEEDED, details)
    )
    assert summary.status is DistributedMapStatus.SUCCEEDED
    assert summary.completion_reason is DistributedMapCompletionReason.ALL_COMPLETED
    assert summary.success_count == 2


def test_item_handler_invalid_report_rejected():
    with pytest.raises(ValidationError, match="report must be"):
        distributed_map_item_handler(lambda x: x, report="bogus")


def test_durable_item_handler_invalid_report_rejected():
    with pytest.raises(ValidationError, match="report must be"):
        durable_distributed_map_item_handler(lambda _ctx, item: item, report="bogus")


def test_item_handler_rejects_concurrency_below_one():
    with pytest.raises(ValidationError, match="concurrency must be at least 1"):
        distributed_map_item_handler(lambda x: x, concurrency=0)
    with pytest.raises(ValidationError, match="concurrency must be at least 1"):
        distributed_map_item_handler(lambda x: x, concurrency=-1)


def test_item_handler_runs_items_one_at_a_time_by_default():
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def process(x):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Hold the worker so that a wider pool would overlap this item with the
        # next one, which is what makes the peak below meaningful.
        time.sleep(0.02)
        with lock:
            in_flight -= 1
        return x

    handler = distributed_map_item_handler(process)
    handler({"records": [{"itemId": str(i), "body": "1"} for i in range(4)]})
    assert peak == 1


def test_item_handler_honors_requested_concurrency():
    # Each item blocks until all four have arrived, so the batch only finishes
    # if the pool really runs four items at once. At a smaller pool size the
    # first item waits out the timeout and every item fails.
    barrier = threading.Barrier(4, timeout=5)

    def process(x):
        barrier.wait()
        return x

    handler = distributed_map_item_handler(process, concurrency=4)
    resp = handler({"records": [{"itemId": str(i), "body": str(i)} for i in range(4)]})
    assert resp["batchItemFailures"] == []
    # The four items are released together, so completion order is arbitrary
    # while the reported results stay in item order.
    assert resp["batchItemResults"] == [
        {"itemIdentifier": "0", "output": "0"},
        {"itemIdentifier": "1", "output": "1"},
        {"itemIdentifier": "2", "output": "2"},
        {"itemIdentifier": "3", "output": "3"},
    ]


def test_item_handler_reports_results_in_order():
    handler = distributed_map_item_handler(lambda x: x * 2)
    resp = handler(
        {"records": [{"itemId": "0", "body": "2"}, {"itemId": "1", "body": "3"}]}
    )
    assert resp["batchItemResults"] == [
        {"itemIdentifier": "0", "output": "4"},
        {"itemIdentifier": "1", "output": "6"},
    ]
    assert resp["batchItemFailures"] == []


def test_item_handler_captures_failures():
    def process(x):
        if x == "bad":
            msg = "boom"
            raise ValueError(msg)
        return x

    handler = distributed_map_item_handler(process)
    resp = handler(
        {"records": [{"itemId": "0", "body": '"ok"'}, {"itemId": "1", "body": '"bad"'}]}
    )
    assert resp["batchItemResults"] == [{"itemIdentifier": "0", "output": '"ok"'}]
    assert resp["batchItemFailures"] == [
        {
            "itemIdentifier": "1",
            "error": {"errorType": "ValueError", "errorMessage": "boom"},
        }
    ]


def test_item_handler_failures_form_reports_only_failures():
    handler = distributed_map_item_handler(lambda x: x, report="failures")
    resp = handler({"records": [{"itemId": "0", "body": "1"}]})
    assert resp == {"batchItemFailures": []}


def test_batch_handler_success_and_propagates_error():
    seen: list = []
    handler = distributed_map_batch_handler(lambda items: seen.extend(items) or "done")
    assert (
        handler(
            {"records": [{"itemId": "0", "body": "1"}, {"itemId": "1", "body": "2"}]}
        )
        == "done"
    )
    assert seen == [1, 2]

    def boom(_items):
        msg = "batch failed"
        raise RuntimeError(msg)

    failing = distributed_map_batch_handler(boom)
    with pytest.raises(RuntimeError, match="batch failed"):
        failing({"records": [{"itemId": "0", "body": "1"}]})


def test_reader_returns_items_and_next_state_then_exhausts():
    def read(state):
        if state is None:
            return ReaderPage(items=[1, 2], next_state={"page": 1})
        return ReaderPage(items=[3])

    handler = distributed_map_reader(read)
    first = handler({"state": None, "maxItems": 10})
    assert first["items"] == [1, 2]
    assert first["nextState"] == '{"page": 1}'

    second = handler({"state": '{"page": 1}', "maxItems": 10})
    assert second["items"] == [3]
    assert "nextState" not in second


def test_reader_rejects_page_over_max_items():
    handler = distributed_map_reader(lambda _s: ReaderPage(items=[1, 2, 3]))
    with pytest.raises(ValidationError, match="exceeding maxItems"):
        handler({"state": None, "maxItems": 2})


def test_reader_rejects_oversized_next_state():
    handler = distributed_map_reader(
        lambda _s: ReaderPage(items=[1], next_state="x" * 40_000)
    )
    with pytest.raises(ValidationError, match="32 KB limit"):
        handler({"state": None, "maxItems": 10})


def test_item_handler_rejects_non_processor_envelope():
    handler = distributed_map_item_handler(lambda x: x)
    with pytest.raises(ValidationError, match="processor envelope"):
        handler({})


def test_reader_rejects_non_reader_envelope():
    handler = distributed_map_reader(lambda _state: ReaderPage(items=[]))
    with pytest.raises(ValidationError, match="reader envelope"):
        handler({"state": None})

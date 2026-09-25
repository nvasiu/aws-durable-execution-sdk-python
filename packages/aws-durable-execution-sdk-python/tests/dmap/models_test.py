"""Unit tests for the distributed map result types."""

from __future__ import annotations

import pytest

from aws_durable_execution_sdk_python.config import (
    DistributedMapCompletionReason,
    DistributedMapItemStatus,
    DistributedMapStatus,
)
from aws_durable_execution_sdk_python.dmap.models import (
    DistributedMapItemError,
    DistributedMapResult,
    DistributedMapResultItem,
    DistributedMapSummary,
)
from aws_durable_execution_sdk_python.exceptions import DistributedMapError


def _result(status, completion_reason, failure_count, items):
    return DistributedMapResult(
        status=status,
        completion_reason=completion_reason,
        success_count=len(items) - failure_count,
        failure_count=failure_count,
        unprocessed_count=0,
        all=items,
    )


def test_summary_throw_if_error():
    ok = DistributedMapSummary(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=2,
        failure_count=0,
        unprocessed_count=0,
    )
    ok.throw_if_error()  # no raise

    failed = DistributedMapSummary(
        status=DistributedMapStatus.FAILED,
        completion_reason=DistributedMapCompletionReason.FAILURE_TOLERANCE_EXCEEDED,
        success_count=0,
        failure_count=1,
        unprocessed_count=0,
    )
    with pytest.raises(DistributedMapError):
        failed.throw_if_error()

    succeeded_with_failures = DistributedMapSummary(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=1,
        failure_count=1,
        unprocessed_count=0,
    )
    with pytest.raises(DistributedMapError):
        succeeded_with_failures.throw_if_error()


def test_map_run_result_succeeded_failed_filters():
    items = [
        DistributedMapResultItem(
            item_id="0", status=DistributedMapItemStatus.SUCCEEDED, output=1
        ),
        DistributedMapResultItem(
            item_id="1",
            status=DistributedMapItemStatus.FAILED,
            error=DistributedMapItemError(error_type="E", error_message="boom"),
        ),
    ]
    result = DistributedMapResult(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=1,
        failure_count=1,
        unprocessed_count=0,
        all=items,
    )
    assert [i.item_id for i in result.succeeded()] == ["0"]
    assert [i.item_id for i in result.failed()] == ["1"]
    assert result.get_results() == [1]
    assert result.get_errors()[0].error_message == "boom"


def test_summary_has_failure_false_without_failures():
    summary = DistributedMapSummary(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=0,
        failure_count=0,
        unprocessed_count=0,
    )
    assert summary.has_failure is False


def test_result_throw_if_error_raises_first_item_error():
    result = _result(
        DistributedMapStatus.SUCCEEDED,
        DistributedMapCompletionReason.ALL_COMPLETED,
        1,
        [
            DistributedMapResultItem(
                item_id="0", status=DistributedMapItemStatus.SUCCEEDED, output=1
            ),
            DistributedMapResultItem(
                item_id="1",
                status=DistributedMapItemStatus.FAILED,
                error=DistributedMapItemError(error_type="E", error_message="boom"),
            ),
        ],
    )
    with pytest.raises(DistributedMapError, match="E: boom"):
        result.throw_if_error()


def test_result_throw_if_error_names_item_without_detail():
    result = _result(
        DistributedMapStatus.SUCCEEDED,
        DistributedMapCompletionReason.ALL_COMPLETED,
        1,
        [
            DistributedMapResultItem(
                item_id="7", status=DistributedMapItemStatus.FAILED, error=None
            )
        ],
    )
    with pytest.raises(DistributedMapError, match="item 7 failed"):
        result.throw_if_error()


def test_result_throw_if_error_falls_back_to_summary_when_no_items():
    result = DistributedMapResult(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=0,
        failure_count=2,
        unprocessed_count=0,
        all=[],
    )
    with pytest.raises(DistributedMapError, match="2 item"):
        result.throw_if_error()


def test_result_throw_if_error_run_level_failure():
    result = DistributedMapResult(
        status=DistributedMapStatus.FAILED,
        completion_reason=DistributedMapCompletionReason.FAILURE_TOLERANCE_EXCEEDED,
        success_count=0,
        failure_count=1,
        unprocessed_count=0,
        all=[],
    )
    with pytest.raises(DistributedMapError, match="Map run ended FAILED"):
        result.throw_if_error()


def test_result_throw_if_error_clean_success_does_not_raise():
    result = DistributedMapResult(
        status=DistributedMapStatus.SUCCEEDED,
        completion_reason=DistributedMapCompletionReason.ALL_COMPLETED,
        success_count=1,
        failure_count=0,
        unprocessed_count=0,
        all=[
            DistributedMapResultItem(
                item_id="0", status=DistributedMapItemStatus.SUCCEEDED, output=1
            )
        ],
    )
    result.throw_if_error()

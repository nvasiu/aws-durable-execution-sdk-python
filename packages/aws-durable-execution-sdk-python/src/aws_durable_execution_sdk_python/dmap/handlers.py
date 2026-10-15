"""Authoring helpers for distributed map processor and reader Lambda functions.

Wrappers that own the request/response format so customers can write a plain
function to process items/batches or read pages. Imported explicitly from this
module, not the main package.
"""

from __future__ import annotations

import functools
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from aws_durable_execution_sdk_python.config import (
    BatchItemStatus,
    CompletionConfig,
    MapConfig,
)
from aws_durable_execution_sdk_python.exceptions import ValidationError
from aws_durable_execution_sdk_python.execution import durable_execution
from aws_durable_execution_sdk_python.lambda_service import ErrorObject
from aws_durable_execution_sdk_python.serdes import (
    DEFAULT_JSON_SERDES,
    SerDes,
    SerDesContext,
)

_READER_STATE_LIMIT = 32 * 1024


@dataclass(frozen=True)
class ReaderPage:
    """A page returned by a reader function: its items and the next state."""

    items: list[Any] = field(default_factory=list)
    next_state: Any | None = None


@dataclass(frozen=True)
class ProcessorRecord:
    """One item of the batch handed to a processor function."""

    item_id: str
    body: Any = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ProcessorRecord:
        return cls(item_id=data.get("itemId", ""), body=data.get("body"))

    def to_dict(self) -> MutableMapping[str, Any]:
        return {"itemId": self.item_id, "body": self.body}


@dataclass(frozen=True)
class ProcessorEvent:
    """The event a processor function is invoked with, one batch of items."""

    records: list[ProcessorRecord] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ProcessorEvent:
        records = data.get("records")
        if not isinstance(records, list):
            msg = "expected a distributed map processor envelope with a 'records' list"
            raise ValidationError(msg)
        return cls(records=[ProcessorRecord.from_dict(r) for r in records])

    def to_dict(self) -> MutableMapping[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


@dataclass(frozen=True)
class ItemResult:
    """One item's output, reported by an item_outcomes processor."""

    item_identifier: str
    output: str | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemResult:
        return cls(
            item_identifier=data.get("itemIdentifier", ""), output=data.get("output")
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"itemIdentifier": self.item_identifier}
        if self.output is not None:
            result["output"] = self.output
        return result


@dataclass(frozen=True)
class ItemFailure:
    """One item's failure, reported by an item_outcomes processor."""

    item_identifier: str
    error: ErrorObject | None = None

    @classmethod
    def from_exception(cls, item_identifier: str, exc: Exception) -> ItemFailure:
        return cls(
            item_identifier=item_identifier, error=ErrorObject.from_exception(exc)
        )

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemFailure:
        error = data.get("error")
        return cls(
            item_identifier=data.get("itemIdentifier", ""),
            error=ErrorObject(
                message=error.get("errorMessage"),
                type=error.get("errorType"),
                data=error.get("errorData"),
                stack_trace=error.get("stackTrace"),
            )
            if error
            else None,
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"itemIdentifier": self.item_identifier}
        if self.error is None:
            return result
        # The processor response envelope is camelCase, so ErrorObject.to_dict
        # cannot be reused here. Every member is optional and the service
        # stores each one unchanged, so an absent member is omitted.
        error: MutableMapping[str, Any] = {}
        if self.error.type is not None:
            error["errorType"] = self.error.type
        if self.error.message is not None:
            error["errorMessage"] = self.error.message
        if self.error.data is not None:
            error["errorData"] = self.error.data
        if self.error.stack_trace is not None:
            error["stackTrace"] = self.error.stack_trace
        if error:
            result["error"] = error
        return result


@dataclass(frozen=True)
class ItemHandlerResponse:
    """What an item handler returns: a failure or a result for each item."""

    failures: list[ItemFailure] = field(default_factory=list)
    results: list[ItemResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemHandlerResponse:
        return cls(
            failures=[
                ItemFailure.from_dict(f) for f in data.get("batchItemFailures", [])
            ],
            results=[ItemResult.from_dict(r) for r in data.get("batchItemResults", [])],
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "batchItemFailures": [f.to_dict() for f in self.failures],
            "batchItemResults": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True)
class ReaderEvent:
    """The event a reader function is invoked with."""

    max_items_per_page: int
    state: str | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ReaderEvent:
        max_items_per_page = data.get("maxItemsPerPage")
        if not isinstance(max_items_per_page, int):
            msg = (
                "expected a distributed map reader envelope with an integer "
                "'maxItemsPerPage'"
            )
            raise ValidationError(msg)
        return cls(max_items_per_page=max_items_per_page, state=data.get("state"))

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"maxItemsPerPage": self.max_items_per_page}
        if self.state is not None:
            result["state"] = self.state
        return result


@dataclass(frozen=True)
class ReaderResponse:
    """What a reader function returns. A None ``next_state`` exhausts the source."""

    items: list[str] = field(default_factory=list)
    next_state: str | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ReaderResponse:
        return cls(items=list(data.get("items", [])), next_state=data.get("nextState"))

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"items": self.items}
        if self.next_state is not None:
            result["nextState"] = self.next_state
        return result


def _to_item(serdes: SerDes[Any], body: str, ctx: SerDesContext) -> Any:
    """Recover a typed item from a record body."""
    return serdes.deserialize(body, ctx)


def _to_output(serdes: SerDes[Any], value: Any, ctx: SerDesContext) -> str | None:
    """Serialize an item's output, or None when the handler returned nothing."""
    if value is None:
        return None
    return serdes.serialize(value, ctx)


def _validate_concurrency(concurrency: int) -> None:
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got: {concurrency}"
        raise ValidationError(msg)


def distributed_map_item_handler(
    func: Callable[[Any], Any] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
    result_serdes: SerDes[Any] | None = None,
    concurrency: int = 1,
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as an item_outcomes processor.

    ``func`` takes one item and returns its output or raises. The decorated name
    becomes the Lambda handler and takes ``(event, context)``, so name it
    ``handler``. An item whose handler returns ``None`` succeeds with no output.

    Pair this with ``DistributedMapProcessor.report_item_outcomes``. Paired with
    ``report_batch_outcome`` instead, the service ignores the response this
    returns, so every item in the batch is marked succeeded, including the items
    whose handler raised.

    Items in a batch are processed one at a time. Raise ``concurrency`` to run
    that many items at once, which requires ``func`` to be safe to call from
    several threads.
    """
    if func is None:
        return functools.partial(
            distributed_map_item_handler,
            item_serdes=item_serdes,
            result_serdes=result_serdes,
            concurrency=concurrency,
        )
    _validate_concurrency(concurrency)
    serdes = item_serdes or DEFAULT_JSON_SERDES
    out_serdes = result_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
        ctx = SerDesContext()
        records = ProcessorEvent.from_dict(event).records

        def run(record: ProcessorRecord) -> str | None:
            # Serializing here, not after the loop, keeps a serialization
            # failure on the one item whose output could not be serialized.
            return _to_output(out_serdes, func(_to_item(serdes, record.body, ctx)), ctx)

        outputs: list[str | None] = [None] * len(records)
        errors: list[Exception | None] = [None] * len(records)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(run, r): i for i, r in enumerate(records)}
            for future, i in futures.items():
                try:
                    outputs[i] = future.result()
                except Exception as exc:  # noqa: BLE001
                    errors[i] = exc

        results: list[ItemResult] = []
        failures: list[ItemFailure] = []
        for i, record in enumerate(records):
            err = errors[i]
            if err is not None:
                failures.append(ItemFailure.from_exception(record.item_id, err))
            else:
                results.append(
                    ItemResult(item_identifier=record.item_id, output=outputs[i])
                )

        response = ItemHandlerResponse(failures=failures, results=results)
        return dict(response.to_dict())

    return handler


def distributed_map_batch_handler(
    func: Callable[[list[Any]], None] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as a batch processor.

    ``func`` takes the whole batch of items. Returning succeeds every item;
    raising fails every item. The decorated name becomes the Lambda handler and
    takes ``(event, context)``, so name it ``handler``.

    A batch processor reports no per-item outcome, so it records no output. Use
    ``distributed_map_item_handler`` to report an output per item.
    """
    if func is None:
        return functools.partial(distributed_map_batch_handler, item_serdes=item_serdes)
    serdes = item_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> None:
        ctx = SerDesContext()
        func(
            [
                _to_item(serdes, record.body, ctx)
                for record in ProcessorEvent.from_dict(event).records
            ]
        )

    return handler


def distributed_map_reader(
    func: Callable[[Any, int], ReaderPage] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
    state_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as a reader source.

    ``func`` takes the current state and the maximum items this page may
    return, and returns a ReaderPage. A ``next_state`` of ``None`` signals the
    source is exhausted. The decorated name becomes the Lambda handler and
    takes ``(event, context)``, so name it ``handler``.

    ``item_serdes`` serializes each item of the page. It must be the same serdes
    the processor is configured with, since the processor deserializes every
    item body with its own ``item_serdes``.
    """
    if func is None:
        return functools.partial(
            distributed_map_reader,
            item_serdes=item_serdes,
            state_serdes=state_serdes,
        )
    item_sd = item_serdes or DEFAULT_JSON_SERDES
    state_sd = state_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
        ctx = SerDesContext()
        reader_event = ReaderEvent.from_dict(event)
        state = (
            state_sd.deserialize(reader_event.state, ctx)
            if reader_event.state is not None
            else None
        )

        page = func(state, reader_event.max_items_per_page)
        if len(page.items) > reader_event.max_items_per_page:
            msg = (
                f"reader returned {len(page.items)} items, exceeding "
                f"maxItemsPerPage {reader_event.max_items_per_page}"
            )
            raise ValidationError(msg)

        items = [item_sd.serialize(item, ctx) for item in page.items]

        next_state: str | None = None
        if page.next_state is not None:
            next_state = state_sd.serialize(page.next_state, ctx)
            if len(next_state.encode("utf-8")) > _READER_STATE_LIMIT:
                msg = f"reader next_state exceeds the {_READER_STATE_LIMIT // 1024} KB limit"
                raise ValidationError(msg)

        return dict(ReaderResponse(items=items, next_state=next_state).to_dict())

    return handler


def durable_distributed_map_item_handler(
    func: Callable[..., Any] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
    result_serdes: SerDes[Any] | None = None,
    concurrency: int = 1,
) -> Callable[..., Any]:
    """Durable variant of the item handler. ``func`` receives (context, item).

    The Lambda must be deployed as a durable function. The decorated name becomes
    the Lambda handler and takes ``(event, context)``, so name it ``handler``.
    An item whose handler returns ``None`` succeeds with no output.

    Pair this with ``DistributedMapProcessor.report_item_outcomes``. Paired with
    ``report_batch_outcome`` instead, the service ignores the response this
    returns, so every item in the batch is marked succeeded, including the items
    whose handler raised.

    Items in a batch are processed one at a time. Raise ``concurrency`` to run
    that many items at once.
    """
    if func is None:
        return functools.partial(
            durable_distributed_map_item_handler,
            item_serdes=item_serdes,
            result_serdes=result_serdes,
            concurrency=concurrency,
        )
    _validate_concurrency(concurrency)
    serdes = item_serdes or DEFAULT_JSON_SERDES
    out_serdes = result_serdes or DEFAULT_JSON_SERDES

    @durable_execution
    def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
        ctx = SerDesContext()
        records = ProcessorEvent.from_dict(event).records

        def per_item(inner_ctx: Any, body: Any, _index: int, _inputs: Any) -> Any:
            return func(inner_ctx, _to_item(serdes, body, ctx))

        batch = context.map(
            [record.body for record in records],
            per_item,
            config=MapConfig(
                max_concurrency=concurrency,
                completion_config=CompletionConfig.all_completed(),
            ),
        )

        results: list[ItemResult] = []
        failures: list[ItemFailure] = []
        for bi in batch.all:
            item_id = records[bi.index].item_id
            if bi.status is BatchItemStatus.SUCCEEDED:
                # Catching here, not around the loop, keeps a serialization
                # failure on the one item whose output could not be serialized.
                try:
                    output = _to_output(out_serdes, bi.result, ctx)
                except Exception as exc:  # noqa: BLE001
                    failures.append(ItemFailure.from_exception(item_id, exc))
                else:
                    results.append(ItemResult(item_identifier=item_id, output=output))
            else:
                failures.append(ItemFailure(item_identifier=item_id, error=bi.error))

        response = ItemHandlerResponse(failures=failures, results=results)
        return dict(response.to_dict())

    return handler


def durable_distributed_map_batch_handler(
    func: Callable[..., None] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Durable variant of the batch handler. ``func`` receives (context, items).

    The Lambda must be deployed as a durable function. The decorated name becomes
    the Lambda handler and takes ``(event, context)``, so name it ``handler``.

    A batch processor reports no per-item outcome, so it records no output. Use
    ``durable_distributed_map_item_handler`` to report an output per item.
    """
    if func is None:
        return functools.partial(
            durable_distributed_map_batch_handler, item_serdes=item_serdes
        )
    serdes = item_serdes or DEFAULT_JSON_SERDES

    @durable_execution
    def handler(event: dict[str, Any], context: Any) -> None:
        ctx = SerDesContext()
        func(
            context,
            [
                _to_item(serdes, record.body, ctx)
                for record in ProcessorEvent.from_dict(event).records
            ],
        )

    return handler

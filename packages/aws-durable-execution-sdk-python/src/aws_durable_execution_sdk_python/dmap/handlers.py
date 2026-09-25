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
from typing import Any, Callable, Literal

from aws_durable_execution_sdk_python.concurrency.models import BatchItemStatus
from aws_durable_execution_sdk_python.config import CompletionConfig, MapConfig
from aws_durable_execution_sdk_python.exceptions import ValidationError
from aws_durable_execution_sdk_python.execution import durable_execution
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

    records: tuple[ProcessorRecord, ...] = ()

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ProcessorEvent:
        records = data.get("records")
        if not isinstance(records, list):
            msg = "expected a distributed map processor envelope with a 'records' list"
            raise ValidationError(msg)
        return cls(records=tuple(ProcessorRecord.from_dict(r) for r in records))

    def to_dict(self) -> MutableMapping[str, Any]:
        return {"records": [record.to_dict() for record in self.records]}


@dataclass(frozen=True)
class ItemResult:
    """One item's output, reported by an item_results processor."""

    item_identifier: str
    output: Any = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemResult:
        return cls(
            item_identifier=data.get("itemIdentifier", ""), output=data.get("output")
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        return {"itemIdentifier": self.item_identifier, "output": self.output}


@dataclass(frozen=True)
class ItemFailure:
    """One item's failure, reported by an item_failures or item_results processor."""

    item_identifier: str
    error_type: str = ""
    error_message: str = ""

    @classmethod
    def from_exception(cls, item_identifier: str, exc: BaseException) -> ItemFailure:
        return cls(
            item_identifier=item_identifier,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemFailure:
        error = data.get("error") or {}
        return cls(
            item_identifier=data.get("itemIdentifier", ""),
            error_type=error.get("errorType", ""),
            error_message=error.get("errorMessage", ""),
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        return {
            "itemIdentifier": self.item_identifier,
            "error": {
                "errorType": self.error_type,
                "errorMessage": self.error_message,
            },
        }


@dataclass(frozen=True)
class ItemHandlerResponse:
    """What an item handler returns. A None ``results`` reports failures only."""

    failures: tuple[ItemFailure, ...] = ()
    results: tuple[ItemResult, ...] | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ItemHandlerResponse:
        results = data.get("batchItemResults")
        return cls(
            failures=tuple(
                ItemFailure.from_dict(f) for f in data.get("batchItemFailures", [])
            ),
            results=tuple(ItemResult.from_dict(r) for r in results)
            if results is not None
            else None,
        )

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {
            "batchItemFailures": [f.to_dict() for f in self.failures]
        }
        if self.results is not None:
            result["batchItemResults"] = [r.to_dict() for r in self.results]
        return result


@dataclass(frozen=True)
class ReaderEvent:
    """The event a reader function is invoked with."""

    max_items: int
    state: str | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ReaderEvent:
        max_items = data.get("maxItems")
        if not isinstance(max_items, int):
            msg = (
                "expected a distributed map reader envelope with an integer 'maxItems'"
            )
            raise ValidationError(msg)
        return cls(max_items=max_items, state=data.get("state"))

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"maxItems": self.max_items}
        if self.state is not None:
            result["state"] = self.state
        return result


@dataclass(frozen=True)
class ReaderResponse:
    """What a reader function returns. A None ``next_state`` exhausts the source."""

    items: tuple[Any, ...] = ()
    next_state: str | None = None

    @classmethod
    def from_dict(cls, data: MutableMapping[str, Any]) -> ReaderResponse:
        return cls(items=tuple(data.get("items", ())), next_state=data.get("nextState"))

    def to_dict(self) -> MutableMapping[str, Any]:
        result: MutableMapping[str, Any] = {"items": list(self.items)}
        if self.next_state is not None:
            result["nextState"] = self.next_state
        return result


def _to_item(serdes: SerDes[Any], body: str, ctx: SerDesContext) -> Any:
    """Recover a typed item from a record body."""
    return serdes.deserialize(body, ctx)


def _to_output(serdes: SerDes[Any], value: Any, ctx: SerDesContext) -> str:
    """Serialize an item's output for the Output field."""
    return serdes.serialize(value, ctx)


def _validate_report(report: str) -> None:
    if report not in ("results", "failures"):
        msg = f"report must be 'results' or 'failures', got: {report!r}"
        raise ValidationError(msg)


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
    report: Literal["results", "failures"] = "results",
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as an item_results processor.

    Pass ``report="failures"`` for an item_failures processor. ``func``
    takes one item and returns its output or raises. The decorated name becomes
    the Lambda handler and takes ``(event, context)``, so name it ``handler``.

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
            report=report,
        )
    _validate_report(report)
    _validate_concurrency(concurrency)
    serdes = item_serdes or DEFAULT_JSON_SERDES
    out_serdes = result_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
        ctx = SerDesContext()
        records = ProcessorEvent.from_dict(event).records

        def run(record: ProcessorRecord) -> Any:
            return func(_to_item(serdes, record.body, ctx))

        outputs: list[Any] = [None] * len(records)
        errors: list[BaseException | None] = [None] * len(records)
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
            elif report == "results":
                results.append(
                    ItemResult(
                        item_identifier=record.item_id,
                        output=_to_output(out_serdes, outputs[i], ctx),
                    )
                )

        response = ItemHandlerResponse(
            failures=tuple(failures),
            results=tuple(results) if report == "results" else None,
        )
        return dict(response.to_dict())

    return handler


def distributed_map_batch_handler(
    func: Callable[[list[Any]], Any] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as a batch processor.

    ``func`` takes the whole batch of items. Returning succeeds every item;
    raising fails every item. The decorated name becomes the Lambda handler and
    takes ``(event, context)``, so name it ``handler``.
    """
    if func is None:
        return functools.partial(distributed_map_batch_handler, item_serdes=item_serdes)
    serdes = item_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> Any:
        ctx = SerDesContext()
        return func(
            [
                _to_item(serdes, record.body, ctx)
                for record in ProcessorEvent.from_dict(event).records
            ]
        )

    return handler


def distributed_map_reader(
    func: Callable[[Any], ReaderPage] | None = None,
    *,
    state_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Wrap a Lambda to be used as a reader source.

    ``func`` takes the current state and returns a ReaderPage. A ``next_state``
    of ``None`` signals the source is exhausted. The decorated name becomes the
    Lambda handler and takes ``(event, context)``, so name it ``handler``.
    """
    if func is None:
        return functools.partial(distributed_map_reader, state_serdes=state_serdes)
    serdes = state_serdes or DEFAULT_JSON_SERDES

    def handler(event: dict[str, Any], _context: Any = None) -> dict[str, Any]:
        ctx = SerDesContext()
        reader_event = ReaderEvent.from_dict(event)
        state = (
            serdes.deserialize(reader_event.state, ctx)
            if reader_event.state is not None
            else None
        )

        page = func(state)
        if len(page.items) > reader_event.max_items:
            msg = (
                f"reader returned {len(page.items)} items, exceeding maxItems "
                f"{reader_event.max_items}"
            )
            raise ValidationError(msg)

        next_state: str | None = None
        if page.next_state is not None:
            next_state = serdes.serialize(page.next_state, ctx)
            if len(next_state.encode("utf-8")) > _READER_STATE_LIMIT:
                msg = f"reader next_state exceeds the {_READER_STATE_LIMIT // 1024} KB limit"
                raise ValidationError(msg)

        return dict(
            ReaderResponse(items=tuple(page.items), next_state=next_state).to_dict()
        )

    return handler


def durable_distributed_map_item_handler(
    func: Callable[..., Any] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
    result_serdes: SerDes[Any] | None = None,
    report: Literal["results", "failures"] = "results",
) -> Callable[..., Any]:
    """Durable variant of the item handler. ``func`` receives (context, item).

    The Lambda must be deployed as a durable function. The decorated name becomes
    the Lambda handler and takes ``(event, context)``, so name it ``handler``.
    """
    if func is None:
        return functools.partial(
            durable_distributed_map_item_handler,
            item_serdes=item_serdes,
            result_serdes=result_serdes,
            report=report,
        )
    _validate_report(report)
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
            config=MapConfig(completion_config=CompletionConfig.all_completed()),
        )

        results: list[ItemResult] = []
        failures: list[ItemFailure] = []
        for bi in batch.all:
            item_id = records[bi.index].item_id
            if bi.status is BatchItemStatus.SUCCEEDED:
                if report == "results":
                    results.append(
                        ItemResult(
                            item_identifier=item_id,
                            output=_to_output(out_serdes, bi.result, ctx),
                        )
                    )
            else:
                err = bi.error
                failures.append(
                    ItemFailure(
                        item_identifier=item_id,
                        error_type=(err.type or "") if err else "",
                        error_message=(err.message or "") if err else "",
                    )
                )

        response = ItemHandlerResponse(
            failures=tuple(failures),
            results=tuple(results) if report == "results" else None,
        )
        return dict(response.to_dict())

    return handler


def durable_distributed_map_batch_handler(
    func: Callable[..., Any] | None = None,
    *,
    item_serdes: SerDes[Any] | None = None,
) -> Callable[..., Any]:
    """Durable variant of the batch handler. ``func`` receives (context, items).

    The Lambda must be deployed as a durable function. The decorated name becomes
    the Lambda handler and takes ``(event, context)``, so name it ``handler``.
    """
    if func is None:
        return functools.partial(
            durable_distributed_map_batch_handler, item_serdes=item_serdes
        )
    serdes = item_serdes or DEFAULT_JSON_SERDES

    @durable_execution
    def handler(event: dict[str, Any], context: Any) -> Any:
        ctx = SerDesContext()
        return func(
            context,
            [
                _to_item(serdes, record.body, ctx)
                for record in ProcessorEvent.from_dict(event).records
            ],
        )

    return handler

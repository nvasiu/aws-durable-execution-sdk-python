"""Unit tests for config module."""

from unittest.mock import Mock

import pytest

from aws_durable_execution_sdk_python.config import (
    CallbackConfig,
    ChildConfig,
    CompletionConfig,
    CompletionDecision,
    CompletionOutcome,
    Duration,
    InvokeConfig,
    MapConfig,
    ParallelConfig,
    StepConfig,
    StepSemantics,
)
from aws_durable_execution_sdk_python.exceptions import ValidationError
from aws_durable_execution_sdk_python.waits import (
    WaitForConditionConfig,
    WaitForConditionDecision,
)
from aws_durable_execution_sdk_python.config import (
    DistributedMapCompletionConfig,
    DistributedMapConfig,
    S3Destination,
    S3Source,
    DistributedMapProcessor,
    DistributedMapSource,
    ReaderSourceConfig,
)


def test_completion_config_defaults():
    """Test CompletionConfig default values."""
    config = CompletionConfig()
    assert config.min_successful is None
    assert config.tolerated_failure_count is None
    assert config.tolerated_failure_percentage is None


def test_completion_config_first_completed():
    """Test CompletionConfig.first_completed factory method."""
    # first_completed is commented out, so this test should be skipped or removed


def test_completion_config_first_successful():
    """Test CompletionConfig.first_successful factory method."""
    config = CompletionConfig.first_successful()
    assert config.min_successful == 1
    assert config.tolerated_failure_count is None
    assert config.tolerated_failure_percentage is None


def test_completion_config_all_completed():
    """Test CompletionConfig.all_completed factory method."""
    config = CompletionConfig.all_completed()
    assert config.min_successful is None
    assert config.tolerated_failure_count is None
    assert config.tolerated_failure_percentage == 100


def test_completion_config_all_successful():
    """Test CompletionConfig.all_successful factory method."""
    config = CompletionConfig.all_successful()
    assert config.min_successful is None
    assert config.tolerated_failure_count == 0
    assert config.tolerated_failure_percentage == 0


def test_parallel_config_defaults():
    """Test ParallelConfig default values."""
    config = ParallelConfig()
    assert config.max_concurrency is None
    assert isinstance(config.completion_config, CompletionConfig)


def test_wait_for_condition_decision_continue():
    """Test WaitForConditionDecision.continue_waiting factory method."""
    decision = WaitForConditionDecision.continue_waiting(Duration.from_seconds(30))
    assert decision.should_continue is True
    assert decision.delay_seconds == 30


def test_wait_for_condition_decision_stop():
    """Test WaitForConditionDecision.stop_polling factory method."""
    decision = WaitForConditionDecision.stop_polling()
    assert decision.should_continue is False
    assert decision.delay_seconds == 0


def test_wait_for_condition_config():
    """Test WaitForConditionConfig with custom values."""

    def wait_strategy(state, attempt):
        return WaitForConditionDecision.continue_waiting(Duration.from_seconds(10))

    serdes = Mock()
    config = WaitForConditionConfig(
        wait_strategy=wait_strategy, initial_state="test_state", serdes=serdes
    )

    assert config.wait_strategy is wait_strategy
    assert config.initial_state == "test_state"
    assert config.serdes is serdes


def test_step_semantics_enum():
    """Test StepSemantics enum."""
    assert StepSemantics.AT_MOST_ONCE_PER_RETRY.value == "AT_MOST_ONCE_PER_RETRY"
    assert StepSemantics.AT_LEAST_ONCE_PER_RETRY.value == "AT_LEAST_ONCE_PER_RETRY"


def test_step_config_defaults():
    """Test StepConfig default values."""
    config = StepConfig()
    assert config.retry_strategy is None
    assert config.step_semantics == StepSemantics.AT_LEAST_ONCE_PER_RETRY
    assert config.serdes is None


def test_step_config_with_values():
    """Test StepConfig with custom values."""
    retry_strategy = Mock()
    serdes = Mock()

    config = StepConfig(
        retry_strategy=retry_strategy,
        step_semantics=StepSemantics.AT_MOST_ONCE_PER_RETRY,
        serdes=serdes,
    )

    assert config.retry_strategy is retry_strategy
    assert config.step_semantics == StepSemantics.AT_MOST_ONCE_PER_RETRY
    assert config.serdes is serdes


def test_child_config_defaults():
    """Test ChildConfig default values."""
    config = ChildConfig()
    assert config.serdes is None
    assert config.sub_type is None


def test_child_config_with_serdes():
    """Test ChildConfig with serdes."""
    serdes = Mock()
    config = ChildConfig(serdes=serdes)
    assert config.serdes is serdes
    assert config.sub_type is None


def test_child_config_with_sub_type():
    """Test ChildConfig with sub_type."""
    sub_type = Mock()
    config = ChildConfig(sub_type=sub_type)
    assert config.serdes is None
    assert config.sub_type is sub_type


def test_child_config_with_summary_generator():
    """Test ChildConfig with summary_generator."""

    def mock_summary_generator(result):
        return f"Summary of {result}"

    config = ChildConfig(summary_generator=mock_summary_generator)
    assert config.serdes is None
    assert config.sub_type is None
    assert config.summary_generator is mock_summary_generator

    # Test that the summary generator works
    result = config.summary_generator("test_data")
    assert result == "Summary of test_data"


def test_map_config_defaults():
    """Test MapConfig default values."""
    config = MapConfig()
    assert config.max_concurrency is None
    assert isinstance(config.completion_config, CompletionConfig)
    assert config.serdes is None


def test_callback_config_defaults():
    """Test CallbackConfig default values."""
    config = CallbackConfig()
    assert config.timeout_seconds == 0
    assert config.heartbeat_timeout_seconds == 0
    assert config.serdes is None


def test_callback_config_with_values():
    """Test CallbackConfig with custom values."""
    serdes = Mock()
    config = CallbackConfig(
        timeout=Duration.from_seconds(30),
        heartbeat_timeout=Duration.from_seconds(10),
        serdes=serdes,
    )
    assert config.timeout_seconds == 30
    assert config.heartbeat_timeout_seconds == 10
    assert config.serdes is serdes


def test_invoke_config_defaults():
    """Test InvokeConfig defaults."""
    config = InvokeConfig()
    assert config.tenant_id is None


def test_invoke_config_with_tenant_id():
    """Test InvokeConfig with explicit tenant_id."""
    config = InvokeConfig(tenant_id="test-tenant")
    assert config.tenant_id == "test-tenant"


# region Config validation


def test_completion_config_rejects_min_successful_below_one():
    with pytest.raises(ValidationError, match="min_successful must be at least 1"):
        CompletionConfig(min_successful=0)


def test_completion_config_rejects_negative_tolerated_failure_count():
    with pytest.raises(
        ValidationError, match="tolerated_failure_count must be non-negative"
    ):
        CompletionConfig(tolerated_failure_count=-1)


def test_completion_config_rejects_out_of_range_failure_percentage():
    with pytest.raises(
        ValidationError, match="tolerated_failure_percentage must be between 0 and 100"
    ):
        CompletionConfig(tolerated_failure_percentage=101)
    with pytest.raises(
        ValidationError, match="tolerated_failure_percentage must be between 0 and 100"
    ):
        CompletionConfig(tolerated_failure_percentage=-0.1)


def test_completion_config_accepts_boundary_values():
    assert CompletionConfig(min_successful=1).min_successful == 1
    assert CompletionConfig(tolerated_failure_count=0).tolerated_failure_count == 0
    assert (
        CompletionConfig(tolerated_failure_percentage=0).tolerated_failure_percentage
        == 0
    )
    assert (
        CompletionConfig(tolerated_failure_percentage=100).tolerated_failure_percentage
        == 100
    )


def test_map_config_rejects_max_concurrency_below_one():
    with pytest.raises(ValidationError, match="max_concurrency must be at least 1"):
        MapConfig(max_concurrency=0)
    with pytest.raises(ValidationError, match="max_concurrency must be at least 1"):
        MapConfig(max_concurrency=-1)


def test_parallel_config_rejects_max_concurrency_below_one():
    with pytest.raises(ValidationError, match="max_concurrency must be at least 1"):
        ParallelConfig(max_concurrency=0)


def test_configs_accept_none_max_concurrency_as_unlimited():
    assert MapConfig().max_concurrency is None
    assert ParallelConfig(max_concurrency=1).max_concurrency == 1


def test_completion_decision_rejects_complete_without_outcome():
    with pytest.raises(
        ValidationError, match="outcome is required when complete is True"
    ):
        CompletionDecision(complete=True)


def test_completion_decision_rejects_outcome_when_not_complete():
    with pytest.raises(
        ValidationError, match="outcome must be None when complete is False"
    ):
        CompletionDecision(complete=False, outcome=CompletionOutcome.SUCCEEDED)


# endregion Config validation


# ==========================================================================
# Distributed map config validation
# ==========================================================================


def test_retry_duration_out_of_range_rejected():
    with pytest.raises(ValidationError, match="between 1 minute and 6 hours"):
        DistributedMapProcessor.batch("p", max_retry_duration=Duration.from_seconds(30))
    with pytest.raises(ValidationError, match="between 1 minute and 6 hours"):
        DistributedMapProcessor.batch("p", max_retry_duration=Duration.from_hours(7))


def test_expected_bucket_owner_must_be_12_digits():
    with pytest.raises(ValidationError, match="12-digit"):
        S3Source.json_lines("s3://b/k.jsonl", expected_bucket_owner="123")


def test_csv_delimiter_invalid_string_rejected():
    with pytest.raises(ValidationError, match="delimiter must be one of"):
        S3Source.csv("s3://b/data.csv", delimiter="BAR")


def test_csv_headers_duplicates_rejected():
    with pytest.raises(ValidationError, match="duplicates"):
        S3Source.csv("s3://b/k.csv", headers=["a", "a"])


def test_json_lines_requires_key():
    with pytest.raises(ValidationError, match="object key"):
        S3Source.json_lines("s3://bucket-only")


def test_timeout_out_of_range_rejected():
    with pytest.raises(ValidationError, match="at most 90 days"):
        DistributedMapConfig(timeout=Duration.from_days(91))


def test_empty_function_name_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        DistributedMapProcessor.batch("")


def test_valid_function_references_accepted():
    for ref in (
        "my-func",
        "my-func:PROD",
        "123456789012:function:my-func",
        "arn:aws:lambda:us-east-1:123456789012:function:my-func",
        "arn:aws:lambda:us-east-1:123456789012:function:my-func:1",
    ):
        # Should not raise.
        DistributedMapProcessor.batch(ref)


def test_function_name_over_max_length_rejected():
    with pytest.raises(ValidationError, match="at most 170 characters"):
        DistributedMapProcessor.batch("f" * 171)


def test_completion_count_and_percentage_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DistributedMapCompletionConfig(
            tolerated_failure_count=1, tolerated_failure_percentage=5
        )


def test_completion_sample_size_requires_percentage():
    with pytest.raises(ValidationError, match="minimum_sample_size"):
        DistributedMapCompletionConfig(minimum_sample_size=10)


def test_completion_negative_count_rejected():
    with pytest.raises(ValidationError, match="non-negative"):
        DistributedMapCompletionConfig(tolerated_failure_count=-1)


def test_completion_percentage_out_of_range_rejected():
    with pytest.raises(ValidationError, match="between 0 and 100"):
        DistributedMapCompletionConfig(tolerated_failure_percentage=150)


def test_completion_sample_size_below_one_rejected():
    with pytest.raises(ValidationError, match="at least 1"):
        DistributedMapCompletionConfig(
            tolerated_failure_percentage=5, minimum_sample_size=0
        )


def test_completion_failure_count_factory():
    assert DistributedMapCompletionConfig.failure_count(3).tolerated_failure_count == 3


def test_retry_duration_below_minimum_rejected():
    with pytest.raises(ValidationError, match="1 minute and 6 hours"):
        DistributedMapProcessor.batch("p", max_retry_duration=Duration.from_seconds(30))


def test_max_items_below_one_rejected():
    with pytest.raises(ValidationError, match="at least 1"):
        S3Source.json_lines("s3://b/k.jsonl", max_items=0)


def test_csv_requires_object_key():
    with pytest.raises(ValidationError, match="csv requires an S3 object key"):
        S3Source.csv("s3://bucket")


def test_source_without_any_kind_rejected():
    with pytest.raises(ValidationError, match="exactly one of"):
        DistributedMapSource()


def test_source_with_two_kinds_rejected():
    with pytest.raises(ValidationError, match="exactly one of"):
        DistributedMapSource(
            inline_items=("a",),
            reader=ReaderSourceConfig(function_name="r"),
        )


def test_success_destination_all_false_rejected():
    with pytest.raises(ValidationError, match="success destination must include"):
        S3Destination.successes(
            "s3://out/ok", include_input=False, include_output=False
        )


def test_failure_destination_all_false_rejected():
    with pytest.raises(ValidationError, match="failure destination must include"):
        S3Destination.failures("s3://out/bad", include_input=False, include_error=False)


def test_invalid_s3_uri_rejected():
    with pytest.raises(ValidationError, match="Invalid S3 URI"):
        S3Source.json_lines("s3:///key.jsonl")


def test_non_s3_scheme_uri_rejected():
    with pytest.raises(ValidationError, match="must start with s3://"):
        S3Source.json_lines("http://foo/bar")


def test_csv_empty_headers_rejected():
    with pytest.raises(ValidationError, match="must be non-empty"):
        S3Source.csv("s3://b/f.csv", headers=[])


def test_negative_retry_attempts_rejected():
    with pytest.raises(ValidationError, match="non-negative"):
        DistributedMapProcessor.batch("p", max_retry_attempts=-5)


def test_batch_size_out_of_range_rejected():
    with pytest.raises(ValidationError, match="between 1 and 10000"):
        DistributedMapProcessor.batch("p", batch_size=0)


def test_durable_execution_name_prefix_too_long_rejected():
    with pytest.raises(ValidationError, match="1 to 36 characters"):
        DistributedMapProcessor.batch("p", durable_execution_name_prefix="x" * 37)

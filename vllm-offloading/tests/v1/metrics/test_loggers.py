# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics import loggers as metrics_loggers
from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger, LoggingStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats


def _make_logging_config():
    return types.SimpleNamespace(
        kv_transfer_config=None,
        observability_config=types.SimpleNamespace(
            cudagraph_metrics=False,
            enable_mfu_metrics=False,
        ),
        compilation_config=types.SimpleNamespace(
            cudagraph_mode=None,
            cudagraph_capture_sizes=[],
        ),
        cache_config=types.SimpleNamespace(num_gpu_blocks=0),
    )


def test_logging_stat_logger_logs_num_cached_experts(
    monkeypatch,
):
    stat_logger = LoggingStatLogger(_make_logging_config())
    stat_logger.last_scheduler_stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=1,
        kv_cache_usage=0.25,
        num_cached_experts=7,
        num_free_huge_blocks=11,
        num_free_kv_blocks=13,
        num_free_expert_blocks=17,
        estimated_available_kv_blocks=19,
    )
    stat_logger.last_prompt_throughput = 1.0

    logged_messages: list[tuple[str, tuple[object, ...]]] = []

    def fake_info(msg: str, *args: object, **kwargs: object) -> None:
        logged_messages.append((msg, args))

    monkeypatch.setattr(metrics_loggers.logger, "info", fake_info)

    stat_logger.log()

    assert len(logged_messages) == 1
    message, args = logged_messages[0]
    assert "Expert cache: %d resident experts" in message
    assert "Pool free huge/KV/expert: %d/%d/%d" in message
    assert "Est. available KV blocks: %d" in message
    assert 7 in args
    assert 11 in args
    assert 13 in args
    assert 17 in args
    assert 19 in args


def test_aggregated_logging_stat_logger_sums_num_cached_experts():
    stat_logger = AggregatedLoggingStatLogger(_make_logging_config(), [0, 1])
    stat_logger.last_scheduler_stats_dict[0] = SchedulerStats(
        num_cached_experts=3,
        kv_cache_usage=0.2,
        num_free_huge_blocks=2,
        num_free_kv_blocks=5,
        num_free_expert_blocks=7,
        estimated_available_kv_blocks=11,
    )
    stat_logger.last_scheduler_stats_dict[1] = SchedulerStats(
        num_cached_experts=5,
        kv_cache_usage=0.4,
        num_free_huge_blocks=3,
        num_free_kv_blocks=13,
        num_free_expert_blocks=17,
        estimated_available_kv_blocks=19,
    )

    stat_logger.aggregate_scheduler_stats()

    assert stat_logger.last_scheduler_stats.num_cached_experts == 8
    assert stat_logger.last_scheduler_stats.num_free_huge_blocks == 5
    assert stat_logger.last_scheduler_stats.num_free_kv_blocks == 18
    assert stat_logger.last_scheduler_stats.num_free_expert_blocks == 24
    assert stat_logger.last_scheduler_stats.estimated_available_kv_blocks == 30


def test_scheduler_make_stats_includes_num_cached_experts():
    scheduler = types.SimpleNamespace(
        log_stats=True,
        running=[object()],
        waiting=[object(), object()],
        block_manager=types.SimpleNamespace(
            block_pool=types.SimpleNamespace(
                get_num_free_blocks=lambda: 23,
                get_num_free_kv_blocks=lambda: 29,
                get_num_free_split_expert_blocks=lambda: 31,
                estimate_available_kv_capacity=lambda: 37,
            ),
            kv_cache_manager=types.SimpleNamespace(
                usage=0.25,
                make_prefix_cache_stats=lambda: PrefixCacheStats(),
            ),
            expert_manager=types.SimpleNamespace(
                get_num_cached_experts=lambda: 11,
            ),
        ),
        connector_prefix_cache_stats=None,
        kv_metrics_collector=None,
        _get_encoder_cache_usage=lambda: 0.5,
    )

    stats = Scheduler.make_stats(scheduler)

    assert stats is not None
    assert stats.num_cached_experts == 11
    assert stats.num_free_huge_blocks == 23
    assert stats.num_free_kv_blocks == 29
    assert stats.num_free_expert_blocks == 31
    assert stats.estimated_available_kv_blocks == 37

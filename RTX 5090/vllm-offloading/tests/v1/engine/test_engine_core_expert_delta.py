# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput


def _resolved_future(value):
    fut = Future()
    fut.set_result(value)
    return fut


def test_step_completes_expert_delta_from_scheduler_output():
    completed_delta_ids: list[int] = []
    requested_shrink_blocks: list[int] = []
    delta = SimpleNamespace(delta_id=11)
    scheduler_output = SimpleNamespace(
        expert_cache_delta=delta,
        required_expert_shrink_blocks=7,
        total_num_scheduled_tokens=1,
    )
    model_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def _complete_cache_delta(delta_id: int) -> None:
        completed_delta_ids.append(delta_id)

    def _update_from_output(scheduler_output_arg, model_output_arg):
        assert scheduler_output_arg is scheduler_output
        assert model_output_arg is model_output
        assert completed_delta_ids == [11]
        return {}

    engine_core = object.__new__(EngineCore)
    engine_core.block_manager = SimpleNamespace(
        expert_manager=SimpleNamespace(
            adjust_expert_cache_capacity=(
                lambda required_expert_shrink_blocks=0:
                requested_shrink_blocks.append(required_expert_shrink_blocks)),
            complete_cache_delta=_complete_cache_delta,
        ))
    engine_core.scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: scheduler_output,
        get_grammar_bitmask=lambda _scheduler_output: None,
        update_from_output=_update_from_output,
    )
    engine_core.model_executor = SimpleNamespace(
        execute_model=lambda _scheduler_output, non_block=True: _resolved_future(
            model_output))
    engine_core._record_memory_trace = lambda: None
    engine_core._process_aborts_queue = lambda: None
    engine_core.log_error_detail = lambda _scheduler_output: nullcontext()
    engine_core.log_iteration_details = lambda _scheduler_output: nullcontext()

    output, model_executed = EngineCore.step(engine_core)

    assert output == {}
    assert model_executed is True
    assert completed_delta_ids == [11]
    assert requested_shrink_blocks == [7]


def test_step_with_batch_queue_completes_dequeued_expert_delta():
    completed_delta_ids: list[int] = []
    delta = SimpleNamespace(delta_id=23)
    scheduler_output = SimpleNamespace(
        expert_cache_delta=delta,
        required_expert_shrink_blocks=0,
        total_num_scheduled_tokens=1,
    )
    model_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})

    def _complete_cache_delta(delta_id: int) -> None:
        completed_delta_ids.append(delta_id)

    def _update_from_output(scheduler_output_arg, model_output_arg):
        assert scheduler_output_arg is scheduler_output
        assert model_output_arg is model_output
        assert completed_delta_ids == [23]
        return {}

    future = _resolved_future(model_output)
    exec_model_fut = _resolved_future(model_output)

    engine_core = object.__new__(EngineCore)
    engine_core.batch_queue = deque([(future, scheduler_output, exec_model_fut)])
    engine_core.batch_queue_size = 2
    engine_core.block_manager = SimpleNamespace(
        expert_manager=SimpleNamespace(
            complete_cache_delta=_complete_cache_delta,
        ))
    engine_core.scheduler = SimpleNamespace(
        has_requests=lambda: False,
        update_from_output=_update_from_output,
    )
    engine_core._process_aborts_queue = lambda: None
    engine_core.log_error_detail = lambda _scheduler_output: nullcontext()
    engine_core.log_iteration_details = lambda _scheduler_output: nullcontext()
    engine_core.use_spec_decode = False

    output, model_executed = EngineCore.step_with_batch_queue(engine_core)

    assert output == {}
    assert model_executed is False
    assert completed_delta_ids == [23]

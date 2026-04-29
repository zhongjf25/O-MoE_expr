# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import threading
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.model_executor.backend_expert_manager import (  # noqa: E402
    BackendExpertManager,
    OffloadedLayerWeights,
    StreamContext,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402


class _DummyPrefetchDaemon:
    def __init__(self, loaded_experts: list[int]):
        self.loaded_experts = loaded_experts
        self.scheduled: list[tuple[int, object]] = []

    def notify_request_layer_loading(self,
                                     layer_idx: int,
                                     buffer_idx: int | None = None,
                                     num_experts: int | None = None,
                                     device: torch.device | None = None):
        return self.notify_layer_loading(
            layer_idx,
            buffer_idx=buffer_idx,
            num_experts=num_experts,
            device=device,
        )

    def notify_layer_loading(self,
                             layer_idx: int,
                             buffer_idx: int | None = None,
                             num_experts: int | None = None,
                             device: torch.device | None = None):
        if buffer_idx is not None and num_experts is not None and device is not None:
            ready_mask = torch.zeros((num_experts, ),
                                     dtype=torch.bool,
                                     device=device)
            if self.loaded_experts:
                ready_ids = torch.tensor(self.loaded_experts,
                                         dtype=torch.long,
                                         device=device)
                ready_mask.index_fill_(0, ready_ids, True)
            return ready_mask, None
        return self.loaded_experts

    def schedule_prefetch(self,
                          layer_idx: int,
                          topk_ids_pred,
                          next_load_ids=None):
        self.scheduled.append((layer_idx, topk_ids_pred, next_load_ids))

    def clear_schedule(self):
        return None

    def shutdown(self):
        return None


class _DummyBlockTable:
    def __init__(self, gpu_table: torch.Tensor):
        self.block_table = type(
            "DummyBlockTableState",
            (),
            {
                "gpu": gpu_table,
                "np": gpu_table.cpu().numpy().copy(),
                "copy_to_gpu": staticmethod(lambda *args, **kwargs: None),
            },
        )()


def _build_manager(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    prefetched_ids: list[int],
    cached_ids: list[int],
    *,
    pin_memory: bool,
) -> tuple[BackendExpertManager, str]:
    StreamContext.init()
    layer_prefix = "model.layers.0.mlp.experts"

    manager = object.__new__(BackendExpertManager)
    manager.first_k_dense_replace = 0
    manager.num_hidden_layers = 1
    manager.layer_prefixes = [layer_prefix]
    manager.layer_prefix_to_idx = {layer_prefix: 0}
    manager.pending_topk_updates = {}
    manager.pending_topk_lock = threading.Lock()
    manager._load_targets = {}
    manager._evict_targets = {}
    manager._pending_reserved_blocks = {}
    manager.active_expert_to_block = {}
    manager._active_cache_delta_id = None
    manager._completed_cache_delta_ids = set()
    manager._selection_masks = {}
    manager.prefetch_daemon = _DummyPrefetchDaemon(prefetched_ids)
    manager.comp_flag = 1
    manager.w13_weight_1 = torch.empty(
        (num_experts, 2 * intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_weight_1 = torch.empty(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w13_weight_2 = torch.empty_like(manager.w13_weight_1)
    manager.w2_weight_2 = torch.empty_like(manager.w2_weight_1)

    w13_cpu = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
    )
    w2_cpu = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
    )
    if pin_memory:
        w13_cpu = w13_cpu.pin_memory()
        w2_cpu = w2_cpu.pin_memory()

    manager.expert_params = {
        layer_prefix: OffloadedLayerWeights(
            w13_cpu=w13_cpu,
            w2_cpu=w2_cpu,
            intermediate_size=intermediate_size,
        )
    }
    if prefetched_ids:
        prefetched_cpu = torch.as_tensor(prefetched_ids, dtype=torch.long)
        prefetched_gpu = prefetched_cpu.to(device="cuda")
        manager.w13_weight_1[prefetched_gpu].copy_(w13_cpu[prefetched_cpu].cuda())
        manager.w2_weight_1[prefetched_gpu].copy_(w2_cpu[prefetched_cpu].cuda())
        manager.w13_weight_2[prefetched_gpu].copy_(w13_cpu[prefetched_cpu].cuda())
        manager.w2_weight_2[prefetched_gpu].copy_(w2_cpu[prefetched_cpu].cuda())

    num_cached = max(1, len(cached_ids))
    manager.w13_blocks = torch.empty(
        (num_cached * 2, intermediate_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    manager.w2_blocks = torch.empty(
        (num_cached, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device="cuda",
    )

    gpu_table = torch.full(
        (1, num_experts, 3),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for block_idx, expert_id in enumerate(cached_ids):
        w13_block = block_idx * 2
        manager.w13_blocks[w13_block].copy_(w13_cpu[expert_id, :intermediate_size])
        manager.w13_blocks[w13_block + 1].copy_(
            w13_cpu[expert_id, intermediate_size:]
        )
        manager.w2_blocks[block_idx].copy_(w2_cpu[expert_id])
        gpu_table[0, expert_id] = torch.tensor(
            [w13_block, block_idx, w13_block + 1],
            dtype=torch.int32,
            device="cuda",
        )

    manager.block_table = _DummyBlockTable(gpu_table)
    return manager, layer_prefix


def _measure_segmented(
    manager: BackendExpertManager,
    topk_ids: torch.Tensor,
    *,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    layer_idx = 0
    layer_weights = manager.expert_params[manager.get_layer_prefix(layer_idx)]
    stream = StreamContext.memory_stream

    def run_once() -> dict[str, float]:
        next_buffer_idx = manager.comp_flag
        prefetched_ready_mask, handoff_event = manager.prefetch_daemon.notify_layer_loading(
            layer_idx,
            buffer_idx=next_buffer_idx,
            num_experts=manager.w13_weight_1.size(0),
            device=manager.w13_weight_1.device,
        )
        buffer_idx, w13_weight_comp, w2_weight_comp = manager._acquire_comp_buffers()
        compute_inputs = manager._build_offloaded_compute_inputs(
            buffer_idx,
            w13_weight_comp,
            w2_weight_comp,
        )
        current_stream = torch.cuda.current_stream()

        prepare_start = torch.cuda.Event(enable_timing=True)
        prepare_end = torch.cuda.Event(enable_timing=True)
        miss_start = torch.cuda.Event(enable_timing=True)
        miss_end = torch.cuda.Event(enable_timing=True)
        wait_start = torch.cuda.Event(enable_timing=True)
        wait_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            stream.wait_stream(current_stream)
            if handoff_event is not None:
                stream.wait_event(handoff_event)
            prepare_start.record(stream)
            miss_expert_ids = manager._prepare_offloaded_compute_inputs(
                buffer_idx,
                compute_inputs,
                layer_idx,
                topk_ids,
                prefetched_ready_mask,
            )
            prepare_end.record(stream)

            miss_start.record(stream)
            manager._copy_uncached_expert_ids_to_comp_buffer(
                layer_weights,
                miss_expert_ids,
                w13_weight_comp,
                w2_weight_comp,
            )
            miss_end.record(stream)

        wait_start.record(current_stream)
        current_stream.wait_stream(stream)
        wait_end.record(current_stream)
        wait_end.synchronize()

        return {
            "prepare_ms": prepare_start.elapsed_time(prepare_end),
            "miss_copy_ms": miss_start.elapsed_time(miss_end),
            "stream_wait_ms": wait_start.elapsed_time(wait_end),
        }

    for _ in range(warmup):
        run_once()

    totals = {
        "prepare_ms": 0.0,
        "miss_copy_ms": 0.0,
        "stream_wait_ms": 0.0,
    }
    for _ in range(iters):
        result = run_once()
        for key, value in result.items():
            totals[key] += value

    return {key: value / iters for key, value in totals.items()}


def _measure_public(
    manager: BackendExpertManager,
    layer_prefix: str,
    topk_ids: torch.Tensor,
    *,
    iters: int,
    warmup: int,
) -> float:
    def run_once():
        manager.get_offloaded_compute_inputs(layer_prefix, topk_ids, [])

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    for _ in range(iters):
        start.record()
        run_once()
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end)
    return elapsed / iters


def main():
    parser = FlexibleArgumentParser(
        description="Benchmark get_offloaded_compute_inputs hot-path segments."
    )
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.num_experts < 8:
        raise ValueError("--num-experts must be at least 8 for the fixed scenarios.")

    scenarios = {
        "hit_only": {
            "prefetched_ids": [0, 1],
            "cached_ids": [2, 3],
            "topk_ids": [[0, 2], [3, 1]],
        },
        "two_misses": {
            "prefetched_ids": [0],
            "cached_ids": [2],
            "topk_ids": [[0, 2], [4, 5]],
        },
        "all_misses": {
            "prefetched_ids": [],
            "cached_ids": [],
            "topk_ids": [[4, 5], [6, 7]],
        },
    }

    for name, config in scenarios.items():
        topk_ids = torch.tensor(
            config["topk_ids"],
            dtype=torch.int32,
            device="cuda",
        )
        manager, layer_prefix = _build_manager(
            args.num_experts,
            args.hidden_size,
            args.intermediate_size,
            config["prefetched_ids"],
            config["cached_ids"],
            pin_memory=args.pin_memory,
        )

        public_ms = _measure_public(
            manager,
            layer_prefix,
            topk_ids,
            iters=args.iters,
            warmup=args.warmup_iters,
        )
        segmented = _measure_segmented(
            manager,
            topk_ids,
            iters=args.iters,
            warmup=args.warmup_iters,
        )
        segmented["total_segments_ms"] = sum(segmented.values())

        print(f"[{name}] public_ms={public_ms:.4f}")
        print(
            "  "
            + " ".join(
                f"{key}={value:.4f}"
                for key, value in segmented.items()
            )
        )


if __name__ == "__main__":
    main()

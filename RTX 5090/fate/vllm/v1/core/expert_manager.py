# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
from collections import defaultdict
import copy

from vllm.logger import init_logger
from vllm.v1.core.block_utils import Block, Blocks
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.expert_ARC_cache import ARC_Cache
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import ExpertCacheDelta

logger = init_logger(__name__)


class ExpertManager:

    def __init__(
        self,
        vllm_config: VllmConfig,
        block_pool: BlockPool,
        num_expert_per_huge: int,
        log_stats: bool = False,
    ) -> None:

        self.vllm_config = vllm_config

        # get some data from hf config.
        self.hf_config = vllm_config.model_config.hf_text_config
        self.num_experts = self.hf_config.num_experts \
            if hasattr(self.hf_config, 'num_experts') \
            else self.hf_config.n_routed_experts
        self.first_k_dense_replace = self.hf_config.first_k_dense_replace \
            if hasattr(self.hf_config, 'first_k_dense_replace') else 0
        self.num_hidden_layers = self.hf_config.num_hidden_layers

        # get some data from expert_offload_config.
        self.expert_offload_config = vllm_config.expert_offload_config
        self.cached_num_experts = self.expert_offload_config.cached_num_experts
        self.offload_expert = self.expert_offload_config.offload_expert
        self.offload_expert_limit = self.expert_offload_config.offload_expert_limit

        # Number of slots per block. block_size == expert_size, 
        # and expert use tp, so each block can store tp_size experts.
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size

        # use TP, so each slot size is the moe_intermediate_size divided by tp_size.
        moe_intermediate_size = self.hf_config.moe_intermediate_size
        assert moe_intermediate_size % self.tp_size == 0, \
            f"intermediate_size ({moe_intermediate_size}) must be divisible by tp_size ({self.tp_size})"
        self.expert_slot_size = moe_intermediate_size // self.tp_size

        self.block_pool = block_pool
        self.num_expert_per_huge = num_expert_per_huge
        self.log_stats = log_stats

        # Mapping from (layer_id, expert_id, w1/w2/w3) to block_id to track the blocks
        # allocated for each expert, so that we can free the blocks when the
        # expert is evicted.
        self.expert_to_block: dict[(int, int, str), int] = {}
        # blocks allocated for the experts.
        self.blocks: Blocks = Blocks(tuple([]), )

        # expert cache list in each layer, dict: {layer_prefix, ARC_cache}
        self.cache_list_each_layer = {}

        # --- dynamic cache adjustment state ---
        self.all_expert_ids = list(range(self.num_experts))
        self.blocks_per_expert = 3  # w1 + w2 + w3

        # Layers eligible for dynamic adjustment (excludes dense layers and
        # the first MoE layer which caches all experts).
        self.adjustable_layers: list[int] = []
        self.target_cache_size: dict[int, int] = {}
        self.min_cache_per_layer: int = 1
        self.max_cache_per_layer: int = self.num_experts

        # Cross-step pending queues for batched execution.
        self._pending_load_queue: list[tuple[int, int]] = []
        self._pending_evict_queue: list[tuple[int, int]] = []

        # Dynamic-cache config (may be overridden from ExpertOffloadConfig).
        self.dynamic_cache_enabled: bool = getattr(
            self.expert_offload_config, 'dynamic_cache_enabled', False)
        self.adjust_interval: int = getattr(
            self.expert_offload_config, 'dynamic_cache_adjust_interval', 5)
        self.expand_threshold: int = getattr(
            self.expert_offload_config, 'dynamic_cache_expand_threshold', 5)
        self.shrink_threshold: int = getattr(
            self.expert_offload_config, 'dynamic_cache_shrink_threshold', 2)
        self.step_counter: int = 0

    def initialize_experts(self) -> dict[(int, int, str), int]:
        """Initialize the experts."""
        # TODO(fzy): use expert popularity to init arc cache.
        # init arc cache for each layer
        for layer_id in range(self.first_k_dense_replace, self.num_hidden_layers):
            if layer_id == self.first_k_dense_replace:
                init_list = list(range(self.num_experts))
            else:
                init_list = list(range(self.cached_num_experts))
            self.cache_list_each_layer[layer_id] = ARC_Cache(init_list)

        # allocate slots for each expert
        layer_expert_list = []
        for layer_id in self.cache_list_each_layer:
            expert_ids = self.cache_list_each_layer[layer_id].get()
            for expert_id in expert_ids:
                layer_expert_list.append((layer_id, expert_id))
        new_expert_to_block = self.allocate_blocks(layer_expert_list)

        # --- initialize dynamic-cache metadata ---
        # Only layers *after* the first MoE layer are adjustable.
        self.adjustable_layers = [
            lid for lid in self.cache_list_each_layer
            if lid > self.first_k_dense_replace
        ]
        for lid in self.cache_list_each_layer:
            self.target_cache_size[lid] = \
                self.cache_list_each_layer[lid].current_size()

        num_adjustable = len(self.adjustable_layers)
        if num_adjustable > 0 and self.offload_expert_limit > 0:
            self.min_cache_per_layer = max(
                1, self.num_experts - self.offload_expert_limit)
        else:
            self.min_cache_per_layer = self.cached_num_experts

        return new_expert_to_block

    def allocate_blocks(
        self,
        layer_expert_list: list[tuple[int, int]],
    ) -> dict[(int, int, str), int]:
        """Allocate expert blocks via BlockPool typed allocation.

        Each (layer, expert) needs 3 expert blocks (w1, w2, w3).
        Returns a dict mapping (layer_id, expert_id, w123) -> expert logical ID.
        """
        new_expert_to_block = {}
        num_blocks_to_allocate = len(layer_expert_list) * 3
        new_block_ids = self.block_pool.allocate_expert_blocks(num_blocks_to_allocate)

        idx = 0
        for layer_id, expert_id in layer_expert_list:
            for w in ("w1", "w2", "w3"):
                bid = new_block_ids[idx]
                idx += 1
                self.expert_to_block[(layer_id, expert_id, w)] = bid
                new_expert_to_block[(layer_id, expert_id, w)] = bid

        return new_expert_to_block

    def free_expert_blocks(
        self,
        experts_to_free: list[tuple[int, int]],
    ) -> None:
        """Free expert blocks back to BlockPool via typed free interface."""
        ids_to_free: list[int] = []
        for (layer_id, expert_id) in experts_to_free:
            for w in ("w1", "w2", "w3"):
                key = (layer_id, expert_id, w)
                block_id = self.expert_to_block.pop(key, None)
                if block_id is not None:
                    ids_to_free.append(block_id)
        if ids_to_free:
            self.block_pool.free_expert_blocks(ids_to_free)

    def free(self, experts: tuple) -> None:
        """Free the blocks for the experts (legacy wrapper)."""
        self.free_expert_blocks(list(experts))

    def get_blocks(self, experts: tuple) -> Blocks:
        """
        Get the blocks for the experts.
        """
        return tuple(self.expert_to_block[expert] for expert in experts)

    def get_block_ids(self, experts: tuple) -> tuple[list[int], ...]:
        """Get the block ids of the expert."""
        return tuple(block.block_id for block, _ in self.get_blocks(experts))

    def update_cache_with_topk_ids(self, layer_id: int, topk_ids: list[int]) -> None:
        """
        使用 topk_ids 更新指定层的 ARC cache。
        
        Args:
            layer_id: 层ID
            topk_ids: 当前层选择的专家ID列表
        """
        if layer_id not in self.cache_list_each_layer:
            logger.warning(f"Layer {layer_id} not found in cache_list_each_layer")
            return
        
        # 将 topk_ids 转换并展平为一维序列，保留访问顺序与重复次数。
        if hasattr(topk_ids, 'tolist'):
            expert_ids = topk_ids.tolist()
        elif isinstance(topk_ids, list):
            expert_ids = topk_ids
        else:
            expert_ids = list(topk_ids)

        flat_expert_ids = []
        for item in expert_ids:
            if isinstance(item, list):
                flat_expert_ids.extend(item)
            else:
                flat_expert_ids.append(item)

        if not flat_expert_ids:
            return
        
        # 更新 ARC cache
        cache = self.cache_list_each_layer[layer_id]
        cache.update_list(flat_expert_ids)


    # ------------------------------------------------------------------
    # Dynamic expert-cache adjustment
    # ------------------------------------------------------------------

    def adjust_expert_cache_capacity(
        self,
        num_waiting_reqs: int = 0,
    ) -> Optional[ExpertCacheDelta]:
        """Decide whether to expand/shrink expert cache based on free blocks.

        Called once per engine step.  The method is a no-op when dynamic cache
        is disabled or the offload feature is off.

        Design highlights
        -----------------
        * Per-layer priority derived from ARC ghost pressure + balance penalty.
        * Respects ``offload_expert_limit`` as a global eviction cap.
        * Large deltas are split across multiple steps via pending queues.
        * Expansion is suppressed when ``num_waiting_reqs > 0`` so that
          incoming requests get blocks first.
        * Dense layers and the first MoE layer (full cache) are never touched.
        """
        if not self.dynamic_cache_enabled or not self.offload_expert:
            return None

        self.step_counter += 1

        # Phase 0: drain any leftover pending work from a previous decision.
        delta = self._drain_pending_batch()
        if delta is not None:
            return delta

        if self.step_counter % self.adjust_interval != 0:
            return None
        if not self.adjustable_layers:
            return None

        # Phase 1: collect metrics.
        num_free = self.block_pool.get_num_free_blocks()
        total = len(self.block_pool.blocks) - 1  # exclude null_block
        if total <= 0:
            return None

        total_evicted = sum(
            max(0, self.cached_num_experts
                - self.cache_list_each_layer[lid].current_size())
            for lid in self.adjustable_layers
        )
        free_expert_slots = num_free * self.num_expert_per_huge // self.blocks_per_expert

        # Phase 2: direction.
        if free_expert_slots < self.shrink_threshold:
            direction = "shrink"
        elif (free_expert_slots > self.expand_threshold
              and num_waiting_reqs == 0):
            direction = "expand"
        else:
            return None

        # Phase 3-4: plan.
        layer_priority = self._compute_layer_priority(direction)
        if direction == "expand":
            self._plan_expansion(layer_priority, free_expert_slots)
        else:
            self._plan_shrinkage(layer_priority, total_evicted)

        # Phase 5: take this step's batch from the newly filled queues.
        return self._drain_pending_batch()

    # -- helpers -----------------------------------------------------------

    def _compute_layer_priority(
        self, direction: str,
    ) -> list[tuple[int, float]]:
        """Score each adjustable layer for expansion / shrinkage.

        Expansion:  high ghost pressure (len(B1)+len(B2)) => high score.
        Shrinkage:  low ghost pressure => high score (those layers can afford
                    losing cache capacity).
        A balance penalty is applied so that layers far from the mean cache
        size are de-prioritised, preventing extreme skew.
        """
        sizes = [self.cache_list_each_layer[lid].current_size()
                 for lid in self.adjustable_layers]
        avg_size = sum(sizes) / len(sizes) if sizes else self.cached_num_experts

        result: list[tuple[int, float]] = []
        for lid in self.adjustable_layers:
            cache = self.cache_list_each_layer[lid]
            cur_size = cache.current_size()
            ghost_pressure = len(cache.B1) + len(cache.B2)

            if direction == "expand":
                score = float(ghost_pressure)
                if cur_size > avg_size and avg_size > 0:
                    overshoot = (cur_size - avg_size) / avg_size
                    score *= max(0.1, 1.0 - overshoot)
                if cur_size >= self.max_cache_per_layer:
                    score = -1.0
            else:
                score = 1.0 / (ghost_pressure + 1)
                if cur_size < avg_size and avg_size > 0:
                    undershoot = (avg_size - cur_size) / avg_size
                    score *= max(0.1, 1.0 - undershoot)
                if cur_size <= self.min_cache_per_layer:
                    score = -1.0

            result.append((lid, score))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _plan_expansion(
        self,
        layer_priority: list[tuple[int, float]],
        free_expert_slots: int,
    ) -> None:
        """Fill ``_pending_load_queue`` with experts to load."""
        budget = free_expert_slots
        for lid, score in layer_priority:
            if score <= 0 or budget <= 0:
                break
            cache = self.cache_list_each_layer[lid]
            cur_size = cache.current_size()
            room = min(1, self.max_cache_per_layer - cur_size, budget)
            if room <= 0:
                continue
            candidates = cache.get_load_candidates(room, self.all_expert_ids)
            for eid in candidates:
                self._pending_load_queue.append((lid, eid))
                budget -= 1

    def _plan_shrinkage(
        self,
        layer_priority: list[tuple[int, float]],
        total_evicted: int,
    ) -> None:
        """Fill ``_pending_evict_queue`` with experts to evict.

        The total number of evicted experts across all adjustable layers must
        not exceed ``offload_expert_limit``.
        """
        remaining_quota = self.offload_expert_limit - total_evicted
        if remaining_quota <= 0:
            return

        target_count = max(1, len(self.adjustable_layers) // 4)
        count = min(target_count, remaining_quota)

        for lid, score in layer_priority:
            if score <= 0 or count <= 0:
                break
            cache = self.cache_list_each_layer[lid]
            cur_size = cache.current_size()
            if cur_size <= self.min_cache_per_layer:
                continue

            evicted_ids = cache.resize(cur_size - 1)
            for eid in evicted_ids:
                self._pending_evict_queue.append((lid, eid))
                count -= 1
            self.target_cache_size[lid] = cache.current_size()

    def _drain_pending_batch(self) -> Optional[ExpertCacheDelta]:
        """Pop a bounded batch from pending queues and return a delta.

        Batch sizing adapts to total pending size so that large bursts are
        spread over roughly 2-6 steps.
        """
        if not self._pending_load_queue and not self._pending_evict_queue:
            return None

        total_pending = (len(self._pending_load_queue)
                         + len(self._pending_evict_queue))
        if total_pending <= 3:
            batch_size = total_pending
        elif total_pending <= 10:
            batch_size = 3
        elif total_pending <= 30:
            batch_size = 5
        else:
            batch_size = max(5, total_pending // 6)

        experts_to_load: list[tuple[int, int]] = []
        experts_to_evict: list[tuple[int, int]] = []
        new_expert_to_block: dict[tuple[int, int, str], int] = {}

        # Evictions first (frees blocks for subsequent loads).
        # evict_n = min(len(self._pending_evict_queue), batch_size)
        for _ in range(len(self._pending_evict_queue)):
            experts_to_evict.append(self._pending_evict_queue.pop(0))
        if experts_to_evict:
            self.free_expert_blocks(experts_to_evict)

        # Loads consume remaining budget.
        # remaining = batch_size - evict_n
        load_n = min(len(self._pending_load_queue), batch_size)
        for _ in range(load_n):
            lid, eid = self._pending_load_queue[0]
            if self.block_pool.get_num_free_blocks() < self.blocks_per_expert:
                break
            self._pending_load_queue.pop(0)
            partial = self.allocate_blocks([(lid, eid)])
            new_expert_to_block.update(partial)
            experts_to_load.append((lid, eid))

            cache = self.cache_list_each_layer[lid]
            cache.resize(cache.current_size() + 1)
            cache.update(eid)
            self.target_cache_size[lid] = cache.current_size()

        if not experts_to_load and not experts_to_evict:
            return None

        return ExpertCacheDelta(
            experts_to_load=experts_to_load,
            experts_to_evict=experts_to_evict,
            new_expert_to_block=new_expert_to_block,
        )
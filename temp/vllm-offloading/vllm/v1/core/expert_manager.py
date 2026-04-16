# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Optional

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.block_utils import Blocks
from vllm.v1.core.expert_hotset import (
    ExpertHotsetLayerProfile,
    expert_hotset_collect_mode_enabled,
    expert_hotset_use_configured_cache_enabled,
    load_expert_hotset_profile,
)
from vllm.v1.core.sched.output import ExpertCacheDelta

logger = init_logger(__name__)


class ExpertManager:

    LOW_WATERMARK_EXPERT_BLOCKS = 2
    EXPAND_HEADROOM_THRESHOLD = 3
    AUTO_CACHE_ROUND_MULTIPLE = 5

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
        self.configured_cached_num_experts = \
            self.expert_offload_config.cached_num_experts
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
        # Authoritative physical residency state per layer.
        self.resident_experts_by_layer: dict[int, set[int]] = {}
        # blocks allocated for the experts.
        self.blocks: Blocks = Blocks(tuple([]), )

        self.hotset_collect_mode = expert_hotset_collect_mode_enabled(vllm_config)
        self.use_configured_cache_during_profile = (
            expert_hotset_use_configured_cache_enabled(vllm_config)
        )
        self.hotset_profile = load_expert_hotset_profile(vllm_config)
        self.hotset_layers: dict[int, ExpertHotsetLayerProfile] = {}
        if self.hotset_profile is not None:
            self.hotset_layers = dict(self.hotset_profile.layers)

        # --- dynamic cache adjustment state ---
        self.blocks_per_expert = 3  # w1 + w2 + w3

        # Layers eligible for dynamic adjustment (excludes dense layers and
        # the first MoE layer which caches all experts).
        self.adjustable_layers: list[int] = []
        self.min_cache_per_layer: int = 1
        self.max_cache_per_layer: int = self.num_experts

        # Dynamic-cache config (may be overridden from ExpertOffloadConfig).
        self.dynamic_cache_enabled: bool = getattr(
            self.expert_offload_config, 'dynamic_cache_enabled', False)
        self._next_delta_id: int = 1
        self._pending_release_by_delta: dict[int, list[int]] = {}
        self._inflight_cache_delta_id: int | None = None

    def initialize_experts(self) -> dict[(int, int, str), int]:
        """Initialize the experts."""
        self.expert_to_block = {}
        self.resident_experts_by_layer = {}
        self._pending_release_by_delta = {}
        self._inflight_cache_delta_id = None
        self.adjustable_layers = []

        if not self.offload_expert:
            self.min_cache_per_layer = 1
            self.max_cache_per_layer = self.num_experts
            return {}

        self.cached_num_experts = self._resolve_cached_num_experts()

        # allocate slots for each expert
        layer_expert_list = []
        for layer_id in range(self.first_k_dense_replace, self.num_hidden_layers):
            if layer_id == self.first_k_dense_replace:
                expert_ids = list(range(self.num_experts))
            elif self.hotset_collect_mode:
                if self.use_configured_cache_during_profile:
                    expert_ids = list(
                        range(min(self.cached_num_experts, self.num_experts))
                    )
                else:
                    expert_ids = []
            else:
                layer_profile = self._require_hotset_layer(layer_id)
                expert_ids = list(
                    layer_profile.prefix(
                        min(self.cached_num_experts, self.num_experts))
                )
            for expert_id in expert_ids:
                layer_expert_list.append((layer_id, expert_id))
        new_expert_to_block = self.allocate_blocks(layer_expert_list)

        # --- initialize dynamic-cache metadata ---
        # Only layers *after* the first MoE layer are adjustable.
        self.adjustable_layers = [
            lid for lid in range(self.first_k_dense_replace + 1, self.num_hidden_layers)
        ]
        limit = max(0, self.offload_expert_limit or 0)
        self.min_cache_per_layer = max(1, self.num_experts - limit)
        self.max_cache_per_layer = self.num_experts
        return new_expert_to_block

    def _resolve_cached_num_experts(self) -> int:
        configured_cached_num_experts = self.configured_cached_num_experts
        if configured_cached_num_experts is not None:
            if configured_cached_num_experts < 0:
                raise ValueError(
                    "cached_num_experts must be non-negative when provided, "
                    f"got {configured_cached_num_experts}."
                )
            return min(configured_cached_num_experts, self.num_experts)
        return self._derive_cached_num_experts_from_huge_blocks()

    def _derive_cached_num_experts_from_huge_blocks(self) -> int:
        num_adjustable_layers = (
            self.num_hidden_layers - self.first_k_dense_replace - 1
        )
        if num_adjustable_layers <= 0:
            logger.info(
                "Auto-initialized cached_num_experts=0 because there are no "
                "adjustable MoE layers after the first MoE layer."
            )
            return 0

        total_expert_blocks = self.block_pool.num_huge_blocks * self.num_expert_per_huge
        reserved_first_layer_blocks = self.num_experts * self.blocks_per_expert
        remaining_expert_blocks = max(
            0,
            total_expert_blocks - reserved_first_layer_blocks,
        )
        raw_cached_num_experts = remaining_expert_blocks / (
            self.blocks_per_expert * num_adjustable_layers
        )
        auto_cached_num_experts = self._round_down_auto_cached_num_experts(
            min(raw_cached_num_experts, float(self.num_experts))
        )

        logger.info(
            "Auto-initialized cached_num_experts=%d using total_expert_blocks=%d, "
            "reserved_first_layer_blocks=%d, adjustable_layers=%d, raw_per_layer=%.2f.",
            auto_cached_num_experts,
            total_expert_blocks,
            reserved_first_layer_blocks,
            num_adjustable_layers,
            raw_cached_num_experts,
        )
        return auto_cached_num_experts

    @classmethod
    def _round_down_auto_cached_num_experts(cls, raw_cached_num_experts: float) -> int:
        floored_cached_num_experts = math.floor(raw_cached_num_experts)
        if floored_cached_num_experts <= 0:
            return 0
        return max(
            0,
            ((floored_cached_num_experts - 1) // cls.AUTO_CACHE_ROUND_MULTIPLE)
            * cls.AUTO_CACHE_ROUND_MULTIPLE,
        )

    def allocate_blocks(
        self,
        layer_expert_list: list[tuple[int, int]],
    ) -> dict[(int, int, str), int]:
        """Allocate expert blocks via BlockPool typed allocation.

        Each (layer, expert) needs 3 expert blocks (w1, w2, w3).
        Returns a dict mapping (layer_id, expert_id, w123) -> expert logical ID.
        """
        new_expert_to_block = {}
        seen_experts: set[tuple[int, int]] = set()
        for layer_id, expert_id in layer_expert_list:
            expert_key = (layer_id, expert_id)
            if expert_key in seen_experts:
                raise RuntimeError(
                    "Attempted to allocate duplicate expert blocks for "
                    f"layer={layer_id}, expert={expert_id}."
                )
            seen_experts.add(expert_key)
            has_existing_mapping = any(
                (layer_id, expert_id, w) in self.expert_to_block
                for w in ("w1", "w2", "w3")
            )
            if has_existing_mapping:
                raise RuntimeError(
                    "Attempted to allocate blocks for an already mapped "
                    f"expert layer={layer_id}, expert={expert_id}."
                )
            if self._is_resident(layer_id, expert_id):
                raise RuntimeError(
                    "Attempted to allocate blocks for an already resident "
                    f"expert layer={layer_id}, expert={expert_id}."
                )

        num_blocks_to_allocate = len(layer_expert_list) * 3
        # print(f"allocate_blocks: num_blocks_to_allocate: {num_blocks_to_allocate}")
        new_block_ids = self.block_pool.allocate_expert_blocks(num_blocks_to_allocate)

        idx = 0
        for layer_id, expert_id in layer_expert_list:
            for w in ("w1", "w2", "w3"):
                bid = new_block_ids[idx]
                idx += 1
                self.expert_to_block[(layer_id, expert_id, w)] = bid
                new_expert_to_block[(layer_id, expert_id, w)] = bid
            self._mark_resident(layer_id, expert_id)

        return new_expert_to_block

    def free_expert_blocks(
        self,
        experts_to_free: list[tuple[int, int]],
    ) -> None:
        """Free expert blocks back to BlockPool via typed free interface."""
        ids_to_free: list[int] = []
        for (layer_id, expert_id) in experts_to_free:
            ids_to_free.extend(
                self._pop_resident_expert_block_ids(layer_id, expert_id)
            )
        if ids_to_free:
            self.block_pool.free_expert_blocks(ids_to_free)

    def _reserve_release_expert_blocks(
        self,
        experts_to_free: list[tuple[int, int]],
        delta_id: int,
    ) -> None:
        ids_to_release: list[int] = []
        for (layer_id, expert_id) in experts_to_free:
            ids_to_release.extend(
                self._pop_resident_expert_block_ids(layer_id, expert_id)
            )
        if ids_to_release:
            self._pending_release_by_delta.setdefault(delta_id, []).extend(
                ids_to_release)

    def complete_cache_delta(self, delta_id: int) -> None:
        if (self._inflight_cache_delta_id is not None
                and delta_id != self._inflight_cache_delta_id):
            raise RuntimeError(
                "Received completion for unexpected expert-cache delta "
                f"{delta_id}; active delta is {self._inflight_cache_delta_id}."
            )
        block_ids = self._pending_release_by_delta.pop(delta_id, None)
        if block_ids:
            self.block_pool.free_expert_blocks(block_ids)
        if self._inflight_cache_delta_id == delta_id:
            self._inflight_cache_delta_id = None

    def free(self, experts: tuple) -> None:
        """Free the blocks for the experts (legacy wrapper)."""
        self.free_expert_blocks(list(experts))

    # def get_blocks(self, experts: tuple) -> Blocks:
    #     """
    #     Get the blocks for the experts.
    #     """
    #     return tuple(self.expert_to_block[expert] for expert in experts)

    # def get_block_ids(self, experts: tuple) -> tuple[list[int], ...]:
    #     """Get the block ids of the expert."""
    #     return tuple(block.block_id for block, _ in self.get_blocks(experts))

    def get_num_cached_experts(self) -> int:
        """Return the number of physically resident expert instances."""
        return sum(len(experts)
                   for experts in self.resident_experts_by_layer.values())

    def _require_hotset_layer(self, layer_id: int) -> ExpertHotsetLayerProfile:
        layer_profile = self.hotset_layers.get(layer_id)
        if layer_profile is None:
            raise ValueError(
                "Missing expert hotset data for layer "
                f"{layer_id}. Regenerate the hotset profile before startup."
            )
        return layer_profile

    # ------------------------------------------------------------------
    # Dynamic expert-cache adjustment
    # ------------------------------------------------------------------

    def adjust_expert_cache_capacity(
        self,
        required_expert_shrink_blocks: int = 0,
    ) -> Optional[ExpertCacheDelta]:
        """Decide whether to expand/shrink expert cache based on block pressure."""
        if not self.dynamic_cache_enabled or not self.offload_expert:
            return None
        if self.hotset_collect_mode:
            return None

        if not self.adjustable_layers:
            return None

        if self._inflight_cache_delta_id is not None:
            return None

        free_expert_blocks = self.block_pool.get_num_free_expert_blocks()
        experts_to_evict: list[tuple[int, int]] = []
        experts_to_load: list[tuple[int, int]] = []
        evict_commit_mode = "row"

        if required_expert_shrink_blocks > 0:
            target_release_blocks = required_expert_shrink_blocks + 1
            target_evict_experts = (
                (target_release_blocks + self.blocks_per_expert - 1)
                // self.blocks_per_expert)
            experts_to_evict = self._collect_shrink_targets(target_evict_experts)
            if experts_to_evict:
                evict_commit_mode = "table"
        elif free_expert_blocks < self.LOW_WATERMARK_EXPERT_BLOCKS:
            experts_to_evict = self._collect_shrink_targets(1)
        else:
            reserved_blocks = (
                self.EXPAND_HEADROOM_THRESHOLD * self.blocks_per_expert)
            expand_budget_blocks = max(0, free_expert_blocks - reserved_blocks)
            num_expand = min(5, expand_budget_blocks // self.blocks_per_expert)
            if num_expand > 0:
                experts_to_load = self._collect_expand_targets(num_expand)

        return self._build_cache_delta(
            experts_to_load,
            experts_to_evict,
            evict_commit_mode=evict_commit_mode,
        )

    # -- helpers -----------------------------------------------------------

    def _collect_expand_targets(
        self,
        max_expand: int,
    ) -> list[tuple[int, int]]:
        selected: list[tuple[int, int]] = []
        if max_expand <= 0:
            return selected
        selected_by_layer: dict[int, set[int]] = {}
        projected_sizes = {
            layer_id: len(self._resident_experts(layer_id))
            for layer_id in self.adjustable_layers
        }
        while len(selected) < max_expand:
            best_candidate: tuple[float, int, int] | None = None
            for layer_id in self.adjustable_layers:
                current_size = projected_sizes[layer_id]
                if current_size >= self.max_cache_per_layer:
                    continue
                layer_profile = self._require_hotset_layer(layer_id)
                next_expert = layer_profile.next_expert(current_size)
                if next_expert is None:
                    continue
                gain = layer_profile.expand_gain(current_size)
                candidate = (gain, -layer_id, next_expert)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
            if best_candidate is None or best_candidate[0] < 0:
                break
            layer_id = -best_candidate[1]
            expert_id = best_candidate[2]
            resident_experts = self._resident_experts(layer_id)
            projected_resident = resident_experts.union(
                selected_by_layer.get(layer_id, set()))
            expected_prefix = set(
                self._require_hotset_layer(layer_id).prefix(projected_sizes[layer_id])
            )
            if projected_resident != expected_prefix:
                raise RuntimeError(
                    "Resident experts no longer match the hotset prefix for "
                    f"layer {layer_id}: resident={sorted(projected_resident)}, "
                    f"expected={sorted(expected_prefix)}."
                )
            if expert_id in projected_resident:
                raise RuntimeError(
                    "Attempted to expand an already resident expert "
                    f"layer={layer_id}, expert={expert_id}."
                )
            selected.append((layer_id, expert_id))
            selected_by_layer.setdefault(layer_id, set()).add(expert_id)
            projected_sizes[layer_id] += 1
        return selected

    def _collect_shrink_targets(
        self,
        max_shrink: int,
    ) -> list[tuple[int, int]]:
        selected: list[tuple[int, int]] = []
        if max_shrink <= 0:
            return selected

        selected_by_layer: dict[int, set[int]] = {}
        projected_sizes = {
            layer_id: len(self._resident_experts(layer_id))
            for layer_id in self.adjustable_layers
        }
        while len(selected) < max_shrink:
            best_candidate: tuple[float, int, int] | None = None
            for layer_id in self.adjustable_layers:
                current_size = projected_sizes[layer_id]
                if current_size <= self.min_cache_per_layer:
                    continue
                layer_profile = self._require_hotset_layer(layer_id)
                expert_id = layer_profile.shrink_expert(current_size)
                if expert_id is None:
                    continue
                loss = layer_profile.shrink_loss(current_size)
                candidate = (-loss, -layer_id, expert_id)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
            if best_candidate is None:
                break
            layer_id = -best_candidate[1]
            expert_id = best_candidate[2]
            resident_experts = self._resident_experts(layer_id)
            projected_resident = resident_experts.difference(
                selected_by_layer.get(layer_id, set()))
            expected_prefix = set(
                self._require_hotset_layer(layer_id).prefix(projected_sizes[layer_id])
            )
            if projected_resident != expected_prefix:
                raise RuntimeError(
                    "Resident experts no longer match the hotset prefix for "
                    f"layer {layer_id}: resident={sorted(projected_resident)}, "
                    f"expected={sorted(expected_prefix)}."
                )
            if expert_id not in projected_resident:
                raise RuntimeError(
                    "Attempted to shrink a non-resident expert "
                    f"layer={layer_id}, expert={expert_id}."
                )
            selected.append((layer_id, expert_id))
            selected_by_layer.setdefault(layer_id, set()).add(expert_id)
            projected_sizes[layer_id] -= 1
        return selected

    def _build_cache_delta(
        self,
        experts_to_load: list[tuple[int, int]],
        experts_to_evict: list[tuple[int, int]],
        evict_commit_mode: str = "row",
    ) -> Optional[ExpertCacheDelta]:
        if not experts_to_load and not experts_to_evict:
            return None

        if self._inflight_cache_delta_id is not None:
            raise RuntimeError(
                "Attempted to build a new expert-cache delta while "
                f"delta {self._inflight_cache_delta_id} is still in flight."
            )

        delta_id = self._next_delta_id
        if experts_to_evict:
            self._reserve_release_expert_blocks(experts_to_evict, delta_id)

        new_expert_to_block: dict[tuple[int, int, str], int] = {}
        if experts_to_load:
            new_expert_to_block = self.allocate_blocks(experts_to_load)

        self._inflight_cache_delta_id = delta_id
        self._next_delta_id += 1
        return ExpertCacheDelta(
            delta_id=delta_id,
            experts_to_load=experts_to_load,
            experts_to_evict=experts_to_evict,
            new_expert_to_block=new_expert_to_block,
            evict_commit_mode=evict_commit_mode,
        )

    def _resident_experts(self, layer_id: int) -> set[int]:
        return self.resident_experts_by_layer.setdefault(layer_id, set())

    def _is_resident(self, layer_id: int, expert_id: int) -> bool:
        return expert_id in self._resident_experts(layer_id)

    def _mark_resident(self, layer_id: int, expert_id: int) -> None:
        self._resident_experts(layer_id).add(expert_id)

    def _mark_nonresident(self, layer_id: int, expert_id: int) -> None:
        resident_experts = self._resident_experts(layer_id)
        resident_experts.remove(expert_id)

    def _pop_resident_expert_block_ids(
        self,
        layer_id: int,
        expert_id: int,
    ) -> list[int]:
        if not self._is_resident(layer_id, expert_id):
            raise RuntimeError(
                "Attempted to evict a non-resident expert "
                f"layer={layer_id}, expert={expert_id}."
            )

        block_ids: list[int] = []
        missing_weights: list[str] = []
        for w in ("w1", "w2", "w3"):
            key = (layer_id, expert_id, w)
            block_id = self.expert_to_block.get(key)
            if block_id is None:
                missing_weights.append(w)
            else:
                block_ids.append(block_id)

        if missing_weights:
            raise RuntimeError(
                "Resident expert is missing block mappings for "
                f"layer={layer_id}, expert={expert_id}, weights={missing_weights}."
            )

        for w in ("w1", "w2", "w3"):
            del self.expert_to_block[(layer_id, expert_id, w)]
        self._mark_nonresident(layer_id, expert_id)
        return block_ids

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import Literal, Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

from vllm.v1.core.block_utils import Blocks, get_dtype_size, KVCacheBlocks
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.expert_manager import ExpertManager
from vllm.v1.core.kv_cache_manager import KVCacheManager

logger = init_logger(__name__)


class BlockManager:

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        log_stats: bool = False,
    ) -> None:

        self.vllm_config = vllm_config
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_caching = vllm_config.cache_config.enable_prefix_caching
        speculative_config = vllm_config.speculative_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        max_num_batched_tokens = scheduler_config.max_num_batched_tokens
        self.hidden_size = model_config.hf_text_config.hidden_size
        self.dtype = model_config.hf_text_config.torch_dtype

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        self.kv_events_config = vllm_config.kv_events_config
        enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)
        dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        self.log_stats = log_stats

        # block size: how many KV in a block;
        # huge_block_size: how many blocks in a huge block;
        # self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size # 16
        # self.num_blocks = kv_cache_config.num_blocks
        # self.num_huge_blocks = kv_cache_config.num_huge_blocks
        # self.huge_block_size_kv = self.num_blocks // self.num_huge_blocks

        self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size # 16
        self.num_blocks = kv_cache_config.num_blocks
        self.num_huge_blocks = kv_cache_config.num_huge_blocks
        self.huge_page_size = kv_cache_config.huge_page_size
        self.num_expert_in_huge_block = kv_cache_config.num_expert_in_huge_block
        self.num_kv_in_huge_block = kv_cache_config.num_kv_in_huge_block
        self.num_ssm_in_huge_block = kv_cache_config.num_ssm_in_huge_block
        self.num_conv_in_huge_block = kv_cache_config.num_conv_in_huge_block

        # Detect Mamba and compute conv_kernel_size / ssm_is_l3
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.core.block_utils import get_expert_page_size
        conv_kernel_size = 0
        ssm_is_l3 = False
        for group in kv_cache_config.kv_cache_groups:
            if isinstance(group.kv_cache_spec, MambaSpec):
                conv_kernel_size = group.kv_cache_spec.shapes[0][0]
                expert_page_size = get_expert_page_size(vllm_config)
                _, ssm_page_size = group.kv_cache_spec.split_page_size_bytes()
                ssm_is_l3 = (expert_page_size % ssm_page_size == 0)
                break

        self.block_pool = BlockPool(
            num_huge_blocks=self.num_huge_blocks,
            num_expert_per_huge=self.num_expert_in_huge_block,
            num_ssm_per_huge=self.num_ssm_in_huge_block,
            num_kv_per_huge=self.num_kv_in_huge_block,
            num_conv_per_huge=self.num_conv_in_huge_block,
            conv_kernel_size=conv_kernel_size,
            ssm_is_l3=ssm_is_l3,
            enable_caching=self.enable_caching,
        )

        self.expert_manager = ExpertManager(
            vllm_config=self.vllm_config,
            block_pool=self.block_pool,
            num_expert_per_huge=self.num_expert_in_huge_block,
            log_stats=log_stats,
        )

        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            hash_block_size=self.block_size,
            enable_caching=self.enable_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            block_pool=self.block_pool,
        )

        # if self.enable_caching:
        #     assert len(
        #         set(g.kv_cache_spec.block_size
        #             for g in kv_cache_config.kv_cache_groups)
        #     ) == 1, "Only one block size is supported for now"
        #     self.block_size = kv_cache_config.kv_cache_groups[
        #         0].kv_cache_spec.block_size

        #     if dcp_world_size > 1:
        #         assert len(kv_cache_config.kv_cache_groups) == 1
        #         # Note(hc): need revisit. When both DCP and any future
        #         # PCP are enabled, the block_size may need to be scaled
        #         # by a factor of dcp_size × pcp_size?
        #         self.block_size *= dcp_world_size

        self.kv_cache_config = kv_cache_config
        self._required_expert_shrink_blocks: int = 0

    # shared
    @property
    def usage(self) -> float:
        """Get the block usage.

        Returns:
            The block usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[Blocks] = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ):
        """Allocate slots for a request or an expert.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            type: The type of the block to allocate.
        
        Returns:
            A list of new allocated blocks.
        """
        
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.kv_cache_manager.empty_kv_cache_blocks.blocks
        
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.kv_cache_manager.coordinator.remove_skipped_blocks(request.request_id,
                                               total_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits

        # ] now every single manager will return there own needed blocks, simply add them together will be wrong
        # eg. 10 kv blocks + 1 SSM block + 1 conv block
        # we need to rebuild 'get_num_blocks_to_allocate' this method, and this will also impact the logic to judge 
        # whether the block pool has enough block to allocate
        num_kv_blocks_to_allocate, num_mamba_blocks_to_allocate = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )
        # print(f"[debug] allocate block tuple, kv: {num_kv_blocks_to_allocate}, mamba: {num_mamba_blocks_to_allocate}")

        
        can_allocate = self.block_pool.check_allocation_status(
            num_kv_needed=num_kv_blocks_to_allocate,
            num_conv_needed=num_mamba_blocks_to_allocate,
            num_ssm_needed=num_mamba_blocks_to_allocate,
        )
        if not can_allocate:
            required_expert_blocks = (
                self.block_pool.get_required_expert_blocks_for_allocation(
                    num_kv_needed=num_kv_blocks_to_allocate,
                    num_conv_needed=num_mamba_blocks_to_allocate,
                    num_ssm_needed=num_mamba_blocks_to_allocate,
                ))
            self._required_expert_shrink_blocks = max(
                self._required_expert_shrink_blocks,
                required_expert_blocks,
            )
            return None

        if (
            new_computed_block_list is not self.kv_cache_manager.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.kv_cache_manager.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_kv_blocks = self.kv_cache_manager.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_tokens_main_model, num_encoder_tokens)
        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.kv_cache_manager.create_kv_cache_blocks(new_kv_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.kv_cache_manager.coordinator.cache_blocks(
            request, num_tokens_to_cache)

        return self.kv_cache_manager.create_kv_cache_blocks(new_kv_blocks)

    def reset_required_expert_shrink_blocks(self) -> None:
        self._required_expert_shrink_blocks = 0

    def take_required_expert_shrink_blocks(self) -> int:
        required_blocks = self._required_expert_shrink_blocks
        self._required_expert_shrink_blocks = 0
        return required_blocks

    def free(
        self, 
        request: Request, 
    ) -> None:
        """Free the blocks allocated for the request or an expert.

        Args:
            request: The request to free the blocks.
            type: The type of the block to free. Default is "kv".
            expert: The expert to free the blocks. Default is None.
        """
        self.kv_cache_manager.coordinator.free(request.request_id)
    
    def free_expert_blocks(self, expert: tuple[int, int]) -> None:
        """
        Free the expert blocks for the expert.

        Args:
            expert: The expert to free the blocks.
        """
        self.expert_manager.free(expert)

    # def get_blocks(
    #     self, 
    #     request_id: str, 
    #     type: Literal["kv", "expert"] = "kv",
    #     expert: Optional[tuple[int, int]] = None,
    # ) -> Blocks:
    #     """Get the blocks of a request or an expert."""
    #     if type == "kv":
    #         return self.kv_cache_manager.get_blocks(request_id)
    #     elif type == "expert":
    #         return self.expert_manager.get_blocks(expert)

    # def get_block_ids(
    #     self, 
    #     request_id: str, 
    #     type: Literal["kv", "expert"] = "kv", 
    #     expert: Optional[tuple[int, int]] = None,
    # ) -> tuple[list[int], ...]:
    #     """Get the block ids of a request or an expert."""
    #     if type == "kv":
    #         return self.kv_cache_manager.get_block_ids(request_id)
    #     elif type == "expert":
    #         return self.expert_manager.get_block_ids(expert)

    # for ExpertManager
    def initialize_experts(self) -> Blocks:
        """Initialize the experts."""
        return self.expert_manager.initialize_experts()

    # for KVCacheManager
    def get_computed_blocks(self,
                            request: Request) -> tuple[Blocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """

        return self.kv_cache_manager.get_computed_blocks(request)

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        return self.kv_cache_manager.make_prefix_cache_stats()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        return self.kv_cache_manager.reset_prefix_cache()

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.kv_cache_manager.get_num_common_prefix_blocks(
            request, num_running_requests)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.kv_cache_manager.take_events()
    
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

    def get_memory_breakdown(self) -> tuple[int, int, int]:
        """Return (expert_blocks, kv_blocks, act_blocks) currently in use."""
        num_expert_blocks = len(self.expert_manager.expert_to_block)
        total_used = (len(self.block_pool.blocks)
                      - self.block_pool.free_block_queue.num_free_blocks - 1)
        num_kv_blocks = max(0, total_used - num_expert_blocks)
        return num_expert_blocks, num_kv_blocks

    def create_empty_block_list(self) -> Blocks:
        """Creates a new Blocks instance with no blocks."""
        return self.kv_cache_manager.create_empty_block_list()

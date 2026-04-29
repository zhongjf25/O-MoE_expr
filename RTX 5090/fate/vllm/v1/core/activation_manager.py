# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Optional, List

from vllm.logger import init_logger
from vllm.v1.core.block_utils import Block, Blocks
from vllm.utils.torch_utils import get_dtype_size
from vllm.config import VllmConfig

from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)


class ActivationManager:

    def __init__(
        self,
        vllm_config: VllmConfig,
        block_pool: BlockPool,
        huge_page_size: int,
        num_act_blocks: int,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.hf_config = vllm_config.model_config.hf_text_config
        self.hidden_size = self.hf_config.hidden_size
        self.dtype = self.hf_config.torch_dtype
        self.log_stats = log_stats
        self.block_pool = block_pool
        self.num_all_new_tokens: int = 0
        self.activation_blocks: List[Block] = []
        
        # # consider that page_size = (hidden_size * moe_intermediate_size * dtype) then 
        # # num_slots for hidden state equal page_size / (hidden_size * dtype)
        # moe_intermediate_size = self.hf_config.moe_intermediate_size
        # self.block_size = moe_intermediate_size
        self.huge_page_size = huge_page_size
        self.num_act_blocks = num_act_blocks

        self.rest_act_blocks = self.num_act_blocks

    def allocate_act_blocks(
        self
    ) -> Optional[Blocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
        Blocks layout:
        ```
        ---------------------------------------------------------------------
        |                        KV/Expert                        | < Act.> |
        ---------------------------------------------------------------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        assert self.num_all_new_tokens != 0, "num_all_new_tokens must be greater than 0"

        num_blocks_to_allocate = self.get_num_blocks_to_allocate(self.num_all_new_tokens)
        new_blocks = self.block_pool.get_new_blocks(num_blocks_to_allocate)
        self.activation_blocks.extend(new_blocks)
        return Blocks(new_blocks)

    def add_new_tokens(self, num_new_tokens: int) -> None:
        self.num_all_new_tokens += num_new_tokens
    
    def get_num_blocks_to_allocate(self, num_new_tokens: int) -> int:
        act_size = num_new_tokens * self.hidden_size * get_dtype_size(self.dtype)
        return math.ceil(act_size / self.huge_page_size)
    
    def get_num_act_blocks_need(self, num_new_tokens: int) -> int:
        return self.get_num_blocks_to_allocate(num_new_tokens + self.num_all_new_tokens)

    def get_rest_act_blocks(self) -> int:
        return self.block_pool.free_act_block_queue.num_free_blocks

    def reset(self) -> None:
        self.num_all_new_tokens = 0
        self.rest_act_blocks = self.num_act_blocks
        self.free_all_blocks()

    def free_all_blocks(self) -> None:
        if self.activation_blocks:
            self.block_pool.free_blocks(self.activation_blocks)
            self.activation_blocks = []

    def get_blocks(self) -> Blocks:
        return Blocks(tuple(self.activation_blocks), )

    def get_block_ids(self) -> list[int]:
        """Get the block ids of a request."""
        return [block.block_id for block in self.activation_blocks]

    def create_empty_block_list(self) -> Blocks:
        """Creates a new Blocks instance with no blocks."""
        return Blocks(tuple([]), )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size
from vllm.config.model import ModelConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec

logger = init_logger(__name__)


class ExpertBlockTable:

    def __init__(
        self,
        # block_size: int,
        pin_memory: bool,
        device: torch.device,
        model_config: ModelConfig = None,
    ):
        # self.block_size = block_size
        self.pin_memory = pin_memory
        self.device = device
        self.hf_config = model_config.hf_text_config
        self.num_experts = self.hf_config.num_experts \
            if hasattr(self.hf_config, 'num_experts') \
            else self.hf_config.n_routed_experts
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        self.block_table = self._make_buffer(self.num_hidden_layers,
                                             self.num_experts, 
                                             3,
                                             dtype=torch.int32)
        
        self.WIDX = {
            "w1": 0,
            "w2": 1,
            "w3": 2,
        }
    
    def update_and_commit_experts(self, new_expert_to_block: dict[(int, int, str), int], non_blocking: bool = True) -> None:
        # for layer_id, expert_id, w123 in new_expert_to_block:
        #     self.block_table.np[layer_id, expert_id, self.WIDX[w123]] = new_expert_to_block[(layer_id, expert_id, w123)]
        for layer_id in range(self.num_hidden_layers):
            for expert_id in range(self.num_experts):
                for w123 in self.WIDX:
                    if (layer_id, expert_id, w123) in new_expert_to_block:
                        self.block_table.np[layer_id, expert_id, self.WIDX[w123]] = new_expert_to_block[(layer_id, expert_id, w123)]
                    else:
                        self.block_table.np[layer_id, expert_id, self.WIDX[w123]] = -1
        self.block_table.copy_to_gpu(non_blocking=non_blocking)

    # def clear(self) -> None:
    #     self.block_table.gpu.fill_(0)
    #     self.block_table.cpu.fill_(0)

    def get_device_block_id(self, layer_id: int, expert_id: int) -> tuple[int, int, int]:
        """Returns the device block of the block table."""
        w1_block_id = self.block_table.gpu[layer_id, expert_id, self.WIDX["w1"]]
        w2_block_id = self.block_table.gpu[layer_id, expert_id, self.WIDX["w2"]]
        w3_block_id = self.block_table.gpu[layer_id, expert_id, self.WIDX["w3"]]
        return w1_block_id, w2_block_id, w3_block_id

    def get_cpu_block_id(self, layer_id: int, expert_id: int) -> tuple[int, int, int]:
        """Returns the CPU block of the block table."""
        w1_block_id = self.block_table.cpu[layer_id, expert_id, self.WIDX["w1"]]
        w2_block_id = self.block_table.cpu[layer_id, expert_id, self.WIDX["w2"]]
        w3_block_id = self.block_table.cpu[layer_id, expert_id, self.WIDX["w3"]]
        return w1_block_id, w2_block_id, w3_block_id

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(self, *size: int | torch.SymInt,
                     dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size,
                            dtype=dtype,
                            device=self.device,
                            pin_memory=self.pin_memory)

class KVBlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        model_config: ModelConfig = None,
        kv_cache_config: KVCacheConfig = None,
        table_idx: int = None,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kv_cache_config is not None:
            self.num_layers = len(kv_cache_config.kv_cache_groups[table_idx].layer_names)
        else:
            self.num_layers = model_config.hf_text_config.num_hidden_layers
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.num_layers, self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.num_layers, self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[list[int]],
        row_idx: int,
    ) -> None:
        """Append typed logical KV IDs directly (no re-linearization)."""
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids[0])
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        for i in range(len(block_ids)):
            self.block_table.np[i, row_idx, start:start + num_blocks] = block_ids[i]

    def add_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[:, tgt, :num_blocks] = block_table_np[:, src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[:, src_tgt, :] = self.block_table.np[:, tgt_src, :]

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * total_cp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            # block_table_indices = (
            #     req_indices * self.max_num_blocks_per_req + positions // self.block_size
            # )

            # block_numbers = self.block_table.np.ravel()[block_table_indices]
            # block_offsets = positions % self.block_size
            # np.add(
            #     block_numbers * self.block_size,
            #     block_offsets,
            #     out=self.slot_mapping.np[: req_indices.shape[0]],
            # )
            num_reqs = req_indices.shape[0]
            # [N] -> [L, N]
            req_indices_2d = np.tile(req_indices, (self.num_layers, 1))
    
            layer_indices_1d = np.arange(self.num_layers)
            # [N] -> [L, N]
            layer_indices_2d = np.repeat(layer_indices_1d, num_reqs).reshape(self.num_layers, num_reqs)
            
            block_in_req_indices_1d = positions // self.block_size
            # [N] -> [L, N]
            block_in_req_indices_2d = np.tile(block_in_req_indices_1d, (self.num_layers, 1))

            # block_table: (num_layers, max_reqs, max_blocks)
            block_numbers = self.block_table.np[
                layer_indices_2d,
                req_indices_2d,
                block_in_req_indices_2d
            ]

            # 3. compute slot (Shape: [L, N])
            block_offsets_2d = np.tile(positions % self.block_size, (self.num_layers, 1))
            all_slots = block_numbers * self.block_size + block_offsets_2d
            self.slot_mapping.np[:, :num_reqs] = all_slots

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs, layer_first=True)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping.copy_to_gpu(num_tokens, layer_first=True)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:, :num_reqs, ...]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )

class MambaBlockTable(KVBlockTable):
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        model_config: ModelConfig = None,
        kv_cache_config: KVCacheConfig = None,
        table_idx: int = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        self.hf_config = model_config.hf_text_config
        if kv_cache_config is not None:
            kv_cache_group: KVCacheGroupSpec = kv_cache_config.kv_cache_groups[table_idx]
            kv_cache_spec = kv_cache_group.kv_cache_spec
            assert isinstance(kv_cache_spec, MambaSpec)
            self.num_layers = len(kv_cache_group.layer_names)
        else:
            self.num_layers = model_config.hf_text_config.num_hidden_layers
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        self.block_table = self._make_buffer(
            self.num_layers, self.max_num_reqs, 2, dtype=torch.int32
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping = self._make_buffer(
            self.num_layers, self.max_num_batched_tokens, 2, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        """Append [conv_start, ssm_id] directly (no re-linearization)."""
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids[0])
        assert num_blocks == 2, "mamba block num should be 2 (conv, ssm)"
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        for i in range(len(block_ids)):
            self.block_table.np[i, row_idx, start:start + num_blocks] = block_ids[i]
    
    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * total_cp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            # block_table_indices = (
            #     req_indices * self.max_num_blocks_per_req + positions // self.block_size
            # )

            # block_numbers = self.block_table.np.ravel()[block_table_indices]
            # block_offsets = positions % self.block_size
            # np.add(
            #     block_numbers * self.block_size,
            #     block_offsets,
            #     out=self.slot_mapping.np[: req_indices.shape[0]],
            # )
            return 
            num_reqs = req_indices.shape[0]
            # [N] -> [L, N]
            req_indices_2d = np.tile(req_indices, (self.num_layers, 1))
    
            layer_indices_1d = np.arange(self.num_layers)
            # [N] -> [L, N]
            layer_indices_2d = np.repeat(layer_indices_1d, num_reqs).reshape(self.num_layers, num_reqs)
            
            block_in_req_indices_1d = positions // self.block_size
            # [N] -> [L, N]
            block_in_req_indices_2d = np.tile(block_in_req_indices_1d, (self.num_layers, 1))

            # block_table: (num_layers, max_reqs, max_blocks)
            block_numbers = self.block_table.np[
                layer_indices_2d,
                req_indices_2d,
                block_in_req_indices_2d
            ]

            # 3. compute slot (Shape: [L, N])
            block_offsets_2d = np.tile(positions % self.block_size, (self.num_layers, 1))
            all_slots = block_numbers * self.block_size + block_offsets_2d
            print("[debug] slot mapping", all_slots)
            self.slot_mapping.np[:, :num_reqs, :] = all_slots



class MultiGroupKVBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        model_config: ModelConfig = None,
        kv_cache_config: KVCacheConfig = None,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = self.create_block_tables(
                    block_sizes=block_sizes,
                    max_num_blocks=max_num_blocks,
                    kernel_block_sizes=kernel_block_sizes,
                    max_num_reqs=max_num_reqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    pin_memory=pin_memory,
                    device=device,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                    model_config=model_config,
                    kv_cache_config=kv_cache_config,
                )
        for blk_table in self.block_tables:
            print(type(blk_table))
        # for block_size, kernel_block_size, max_num_blocks_per_req in zip(
        #         block_sizes, kernel_block_sizes, max_num_blocks):
            # print("[debug]", block_size, kernel_block_size, max_num_blocks_per_req)

    def append_row(self, block_ids: tuple[list[list[(int, int)]], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[list[(int, int)]], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def create_block_tables(
        self,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int],
        **common_kwargs
    ) -> list[KVBlockTable]:
        tables = []
        for i, (b_size, k_size, max_blocks) in enumerate(zip(block_sizes, kernel_block_sizes, max_num_blocks)):
            params = {
                "block_size": b_size,
                "kernel_block_size": k_size,
                "max_num_blocks_per_req": max_blocks,
                "table_idx": i,
                **common_kwargs 
            }
            
            if max_blocks == 1:
                table = MambaBlockTable(**params)
            else:
                table = KVBlockTable(**params)
                
            tables.append(table)
            
        return tables

    def __getitem__(self, idx: int) -> "KVBlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]

class BlockTable:
    """The BlockTables for KV, activation, and experts."""

    def __init__(self,
                 max_num_reqs: int,
                 max_model_len: int,
                 max_num_batched_tokens: int,
                 pin_memory: bool,
                 device: torch.device,
                 block_sizes: list[int],
                 kernel_sizes: list[int],
                 max_num_blocks: list[int] | None = None,
                 cp_kv_cache_interleave_size: int = 1,
                 model_config: ModelConfig = None,
                 kv_cache_config: KVCacheConfig = None,
                ) -> None:

        self.expert_block_tables = ExpertBlockTable(
            pin_memory=pin_memory,
            device=device,
            model_config=model_config,
            )

        # moe_intermediate_size = model_config.hf_config.moe_intermediate_size
        # self.activation_block_tables = ActivationBlockTable(
        #     max_num_reqs=max_num_reqs,
        #     max_num_blocks_per_req=cdiv(max_model_len, moe_intermediate_size),
        #     max_num_batched_tokens=max_num_batched_tokens,
        #     pin_memory=pin_memory,
        #     device=device,
        #     block_size=moe_intermediate_size,
        # )

        self.kv_block_tables = MultiGroupKVBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_sizes,
            max_num_blocks=max_num_blocks,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            model_config=model_config,
            kv_cache_config=kv_cache_config
        )

    def append_row(self, block_ids: tuple[list[list[(int, int)]], ...],
                   row_idx: int) -> None:
        for i, block_table in enumerate(self.kv_block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[list[(int, int)]], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.kv_block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.kv_block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.kv_block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        for block_table in self.kv_block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.kv_block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.kv_block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def update_and_commit_experts(self, new_expert_to_block: dict[(int, int, str), int], non_blocking: bool = True) -> None:
        self.expert_block_tables.update_and_commit_experts(new_expert_to_block, non_blocking=non_blocking)

    def clear(self) -> None:
        for block_table in self.kv_block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.kv_block_tables[idx]


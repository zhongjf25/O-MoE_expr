# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical BlockPool: Huge -> L2 (Expert/SSM) -> L3 (KV/Conv/SSM-sub).

All public allocation methods return *typed logical IDs* that directly
index into the corresponding backend tensor views.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.block_utils import (
    Block,
    FreeBlockQueue,
)
from vllm.utils.math_utils import cdiv

logger = init_logger(__name__)


@dataclass
class L2BlockTracker:
    """Track sub-block state of one L2 block (expert or SSM)."""
    huge_block_id: int
    l2_type: str                    # "expert" or "ssm"
    split_into: str | None = None   # None, "kv", "conv", "ssm_sub"
    total_sub_blocks: int = 0
    freed_sub_blocks: int = 0
    is_free: bool = True

@dataclass(frozen=True)
class BlockPoolAllocationSnapshot:
    """Hierarchical BlockPool occupancy: block counts per queue.

    KV / conv / SSM-sub counts reflect **materialized** L2→L3 splits only (on-demand
    from huge), not ``num_huge_blocks * slots_per_huge`` upper bounds.
    """

    num_huge_blocks: int
    num_free_huge_blocks: int
    num_expert_l2_whole_alloc: int
    num_kv_blocks_alloc: int
    num_conv_starts_alloc: int
    num_ssm_blocks_alloc: int

class BlockPool:
    """Hierarchical block pool managing Huge -> L2 -> L3 blocks.

    All typed allocation methods (``allocate_expert_blocks``,
    ``allocate_kv_blocks``, etc.) return plain ``int`` IDs that directly
    index the corresponding backend tensor view.

    Args:
        num_huge_blocks: Number of huge (L1) blocks.
        num_expert_per_huge: Expert L2 blocks per huge block.
        num_ssm_per_huge: SSM blocks per huge block (-1 if no Mamba).
        num_kv_per_huge: KV blocks per huge block.
        num_conv_per_huge: Conv slots per huge block (-1 if no Mamba).
        conv_kernel_size: ``conv.shape[0]`` (0 if no Mamba).
        ssm_is_l3: True when ``expert_page_size % ssm_page_size == 0``,
            meaning SSM is demoted to L3 and can borrow expert blocks.
        enable_caching: Prefix-cache toggle (kept for API compat, ignored).
    """

    def __init__(
        self,
        num_huge_blocks: int,
        num_expert_per_huge: int,
        num_ssm_per_huge: int,
        num_kv_per_huge: int,
        num_conv_per_huge: int,
        conv_kernel_size: int,
        ssm_is_l3: bool,
        enable_caching: bool = False,
    ):
        assert num_huge_blocks > 0
        self.num_huge_blocks = num_huge_blocks
        self.enable_caching = enable_caching

        # ---- page counts ----
        self.num_expert_per_huge = num_expert_per_huge
        self.num_ssm_per_huge = max(num_ssm_per_huge, 0)
        self.num_kv_per_huge = num_kv_per_huge
        self.num_conv_per_huge = max(num_conv_per_huge, 0)
        self.conv_kernel_size = conv_kernel_size
        self.ssm_is_l3 = ssm_is_l3
        self.has_mamba = num_ssm_per_huge > 0

        # ---- derived L3-per-L2 counts ----
        if num_expert_per_huge > 0:
            self.num_kv_per_expert = num_kv_per_huge // num_expert_per_huge
            self.num_conv_per_expert = (
                self.num_conv_per_huge // num_expert_per_huge
                if self.has_mamba else 0
            )
            self.num_ssm_per_expert = (
                self.num_ssm_per_huge // num_expert_per_huge
                if (self.has_mamba and ssm_is_l3) else 0
            )
        else:
            self.num_kv_per_expert = 0
            self.num_conv_per_expert = 0
            self.num_ssm_per_expert = 0

        if self.has_mamba and not ssm_is_l3 and self.num_ssm_per_huge > 0:
            self.num_kv_per_ssm = num_kv_per_huge // self.num_ssm_per_huge
            self.num_conv_per_ssm = self.num_conv_per_huge // self.num_ssm_per_huge
        else:
            self.num_kv_per_ssm = 0
            self.num_conv_per_ssm = 0

        # ---- total typed capacities (informational) ----
        self.total_expert_blocks = num_huge_blocks * num_expert_per_huge
        self.total_kv_blocks = num_huge_blocks * num_kv_per_huge
        self.total_ssm_blocks = (
            num_huge_blocks * self.num_ssm_per_huge if self.has_mamba else 0
        )

        # ---- Block objects  ----
        self.huge_blocks: list[Block] = [
            Block(idx) for idx in range(num_huge_blocks)
        ]
        # self.blocks = self.huge_blocks  # alias for legacy access

        # Null block sentinel (never allocated)
        self.null_block = Block(-1)
        self.null_block.is_null = True

        # ---- Free sets (O(1) add / discard / pop) ----
        self.huge_free_set: set[int] = set(range(num_huge_blocks))
        self.expert_free_set: set[int] = set()
        self.ssm_free_set: set[int] = set()
        self.kv_free_set: set[int] = set()
        self.conv_free_set: set[int] = set()
        self.ssm_sub_free_set: set[int] = set()

        # ---- L2 tracking for hierarchical reclamation ----
        self.huge_to_l2_ids: dict[int, list[int]] = {}
        self.huge_l2_type: dict[int, str] = {}
        self.l2_tracker: dict[int, L2BlockTracker] = {}

        # ---- KV-event stub (prefix cache disabled) ----
        self.enable_kv_cache_events = False
        self.kv_event_queue: list[KVCacheEvent] = []

    def get_num_free_blocks(self) -> int:
        """Number of free huge blocks."""
        return len(self.huge_free_set)

    def get_num_free_kv_blocks(self) -> int:
        """Number of free KV L3 blocks currently in the free set."""
        return len(self.kv_free_set)

    def get_num_free_split_expert_blocks(self) -> int:
        """Number of already-split free expert L2 blocks."""
        return len(self.expert_free_set)

    def get_num_free_expert_blocks(self) -> int:
        """Number of expert L2 blocks immediately allocatable for experts.

        This includes both already-split free expert blocks and free huge
        blocks that can be split into expert blocks on demand.
        """
        return (len(self.expert_free_set)
                + len(self.huge_free_set) * self.num_expert_per_huge)

    def _can_allocate_with_extra_expert_blocks(
        self,
        num_kv_needed: int,
        num_conv_needed: int | None = 0,
        num_ssm_needed: int | None = 0,
        extra_free_expert_blocks: int = 0,
    ) -> bool:
        """Return whether allocation would fit with extra free expert L2 blocks."""
        kv_free_now = len(self.kv_free_set)
        kv_gap = max(0, num_kv_needed - kv_free_now)
        conv_gap = 0
        ssm_gap = 0
        free_l2_ssms = 0

        if self.has_mamba:
            conv_free_now = len(self.conv_free_set)
            conv_gap = max(0, num_conv_needed - conv_free_now)
            if self.ssm_is_l3:
                ssm_free_now = len(self.ssm_sub_free_set)
                ssm_gap = max(0, num_ssm_needed - ssm_free_now)
            else:
                ssm_gap = max(0, num_ssm_needed - len(self.ssm_free_set))
                free_l2_ssms = max(
                    0, len(self.ssm_free_set) - num_ssm_needed)

        free_experts = len(self.expert_free_set) + max(
            0, extra_free_expert_blocks)

        # consider free expert to cover kv block
        needed_exp_for_kv = cdiv(kv_gap, self.num_kv_per_expert)
        used_exp_for_kv = min(free_experts, needed_exp_for_kv)
        free_experts -= used_exp_for_kv
        kv_gap = max(0, kv_gap - used_exp_for_kv * self.num_kv_per_expert)

        # consider free expert to cover conv & L3 ssm block
        if self.has_mamba:
            needed_exp_for_conv = cdiv(conv_gap, self.num_conv_per_expert)
            used_exp_for_conv = min(free_experts, needed_exp_for_conv)
            free_experts -= used_exp_for_conv
            conv_gap = max(0, conv_gap - used_exp_for_conv *
                           self.num_conv_per_expert)

            if self.ssm_is_l3:
                needed_exp_for_ssm = cdiv(ssm_gap, self.num_ssm_per_expert)
                used_exp_for_ssm = min(free_experts, needed_exp_for_ssm)
                free_experts -= used_exp_for_ssm
                ssm_gap = max(0, ssm_gap - used_exp_for_ssm *
                              self.num_ssm_per_expert)

        # if ssm is L2 block than it can also cover kv/conv block
        if self.has_mamba and not self.ssm_is_l3:
            needed_ssm_for_kv = cdiv(kv_gap, self.num_kv_per_ssm)
            used_ssm_for_kv = min(free_l2_ssms, needed_ssm_for_kv)
            free_l2_ssms -= used_ssm_for_kv
            kv_gap = max(0, kv_gap - used_ssm_for_kv * self.num_kv_per_ssm)

            needed_ssm_for_conv = cdiv(conv_gap, self.num_conv_per_ssm)
            used_ssm_for_conv = min(free_l2_ssms, needed_ssm_for_conv)
            free_l2_ssms -= used_ssm_for_conv
            conv_gap = max(0, conv_gap - used_ssm_for_conv *
                           self.num_conv_per_ssm)

        # rest gaps need to cover by huge blocks
        needed_huge = 0
        if kv_gap > 0 or conv_gap > 0 or (self.ssm_is_l3 and ssm_gap > 0):
            rem_exp_needed = cdiv(kv_gap, self.num_kv_per_expert)
            if self.has_mamba:
                rem_exp_needed += cdiv(conv_gap, self.num_conv_per_expert)
                if self.ssm_is_l3:
                    rem_exp_needed += cdiv(ssm_gap, self.num_ssm_per_expert)

            needed_huge += cdiv(rem_exp_needed, self.num_expert_per_huge)

        if self.has_mamba and not self.ssm_is_l3 and ssm_gap > 0:
            needed_huge += cdiv(ssm_gap, self.num_ssm_per_huge)

        free_huge = len(self.huge_free_set)
        return needed_huge <= free_huge

    def check_allocation_status(
        self,
        num_kv_needed: int,
        num_conv_needed: int | None = 0,
        num_ssm_needed: int | None = 0,
    ) -> bool:
        """
        Checks if there are sufficient blocks for the request.

        Returns:
            is_possible (bool): True if the request can be satisfied (with or without preemption).
        """
        return self._can_allocate_with_extra_expert_blocks(
            num_kv_needed=num_kv_needed,
            num_conv_needed=num_conv_needed,
            num_ssm_needed=num_ssm_needed,
        )

    def get_required_expert_blocks_for_allocation(
        self,
        num_kv_needed: int,
        num_conv_needed: int | None = 0,
        num_ssm_needed: int | None = 0,
    ) -> int:
        """Return minimal extra free expert L2 blocks needed for allocation."""
        if self.total_expert_blocks <= 0:
            return 0

        if self._can_allocate_with_extra_expert_blocks(
                num_kv_needed=num_kv_needed,
                num_conv_needed=num_conv_needed,
                num_ssm_needed=num_ssm_needed):
            return 0

        low = 1
        high = self.total_expert_blocks
        required: int | None = None

        while low <= high:
            mid = (low + high) // 2
            if self._can_allocate_with_extra_expert_blocks(
                    num_kv_needed=num_kv_needed,
                    num_conv_needed=num_conv_needed,
                    num_ssm_needed=num_ssm_needed,
                    extra_free_expert_blocks=mid):
                required = mid
                high = mid - 1
            else:
                low = mid + 1

        return 0 if required is None else required

    def get_new_blocks(self, num_blocks: int) -> list[Block]:
        """Allocate raw huge blocks (used by ActivationManager)."""
        if num_blocks > len(self.huge_free_set):
            raise ValueError(
                f"Cannot get {num_blocks} huge blocks from pool "
                f"(only {len(self.huge_free_set)} free)"
            )
        ret: list[Block] = []
        for _ in range(num_blocks):
            hid = self.huge_free_set.pop()
            blk = self.huge_blocks[hid]
            blk.ref_cnt += 1
            ret.append(blk)
        return ret

    def free_blocks(self, ordered_blocks: Iterable[Block]) -> None:
        """Free raw huge blocks back to pool (used by ActivationManager)."""
        for blk in ordered_blocks:
            blk.ref_cnt -= 1
            if blk.ref_cnt == 0 and not blk.is_null:
                hid = blk.block_id
                self.huge_free_set.add(hid)

    # ------------------------------------------------------------------
    # Internal: Huge -> L2 splitting
    # ------------------------------------------------------------------

    def _split_huge_to_expert(self, huge_id: int) -> list[int]:
        base = huge_id * self.num_expert_per_huge
        ids = list(range(base, base + self.num_expert_per_huge))
        self.huge_to_l2_ids[huge_id] = ids
        self.huge_l2_type[huge_id] = "expert"
        for eid in ids:
            self.l2_tracker[eid] = L2BlockTracker(
                huge_block_id=huge_id, l2_type="expert",
            )
        return ids

    def _split_huge_to_ssm(self, huge_id: int) -> list[int]:
        base = huge_id * self.num_ssm_per_huge
        ids = list(range(base, base + self.num_ssm_per_huge))
        self.huge_to_l2_ids[huge_id] = ids
        self.huge_l2_type[huge_id] = "ssm"
        for sid in ids:
            self.l2_tracker[sid] = L2BlockTracker(
                huge_block_id=huge_id, l2_type="ssm",
            )
        return ids

    # ------------------------------------------------------------------
    # Internal: L2 -> L3 secondary splitting
    # ------------------------------------------------------------------

    def _split_expert_to_kv(self, expert_id: int) -> list[int]:
        base = expert_id * self.num_kv_per_expert
        ids = list(range(base, base + self.num_kv_per_expert))
        t = self.l2_tracker[expert_id]
        t.split_into = "kv"
        t.total_sub_blocks = self.num_kv_per_expert
        t.freed_sub_blocks = 0
        t.is_free = False
        return ids

    def _split_expert_to_conv(self, expert_id: int) -> list[int]:
        base = expert_id * self.num_conv_per_expert * self.conv_kernel_size
        ids = [
            base + j * self.conv_kernel_size
            for j in range(self.num_conv_per_expert)
        ]
        t = self.l2_tracker[expert_id]
        t.split_into = "conv"
        t.total_sub_blocks = self.num_conv_per_expert
        t.freed_sub_blocks = 0
        t.is_free = False
        return ids

    def _split_expert_to_ssm_sub(self, expert_id: int) -> list[int]:
        base = expert_id * self.num_ssm_per_expert
        ids = list(range(base, base + self.num_ssm_per_expert))
        t = self.l2_tracker[expert_id]
        t.split_into = "ssm_sub"
        t.total_sub_blocks = self.num_ssm_per_expert
        t.freed_sub_blocks = 0
        t.is_free = False
        return ids

    def _split_ssm_to_kv(self, ssm_id: int) -> list[int]:
        base = ssm_id * self.num_kv_per_ssm
        ids = list(range(base, base + self.num_kv_per_ssm))
        t = self.l2_tracker[ssm_id]
        t.split_into = "kv"
        t.total_sub_blocks = self.num_kv_per_ssm
        t.freed_sub_blocks = 0
        t.is_free = False
        return ids

    def _split_ssm_to_conv(self, ssm_id: int) -> list[int]:
        base = ssm_id * self.num_conv_per_ssm * self.conv_kernel_size
        ids = [
            base + j * self.conv_kernel_size
            for j in range(self.num_conv_per_ssm)
        ]
        t = self.l2_tracker[ssm_id]
        t.split_into = "conv"
        t.total_sub_blocks = self.num_conv_per_ssm
        t.freed_sub_blocks = 0
        t.is_free = False
        return ids

    # ------------------------------------------------------------------
    # Internal: ensure / borrow helpers
    # ------------------------------------------------------------------

    def _pop_huge_and_split_to_expert(self) -> list[int]:
        """Pop one huge block, split into expert L2, return expert IDs."""
        if not self.huge_free_set:
            raise ValueError("No huge blocks available")
        hid = self.huge_free_set.pop()
        self.huge_blocks[hid].ref_cnt += 1
        return self._split_huge_to_expert(hid)

    def _ensure_expert_blocks(self, needed: int) -> None:
        while len(self.expert_free_set) < needed:
            new_ids = self._pop_huge_and_split_to_expert()
            self.expert_free_set.update(new_ids)

    def _ensure_ssm_l2_blocks(self, needed: int) -> None:
        while len(self.ssm_free_set) < needed:
            if not self.huge_free_set:
                raise ValueError("No huge blocks available for SSM L2")
            hid = self.huge_free_set.pop()
            self.huge_blocks[hid].ref_cnt += 1
            new_ids = self._split_huge_to_ssm(hid)
            self.ssm_free_set.update(new_ids)

    def _borrow_l2_for_kv(self) -> list[int]:
        """Borrow one L2 block, secondary-split into KV IDs."""
        if self.expert_free_set:
            eid = self.expert_free_set.pop()
            return self._split_expert_to_kv(eid)
        if not self.ssm_is_l3 and self.ssm_free_set:
            sid = self.ssm_free_set.pop()
            return self._split_ssm_to_kv(sid)
        new_expert = self._pop_huge_and_split_to_expert()
        eid = new_expert[0]
        self.expert_free_set.update(new_expert[1:])
        return self._split_expert_to_kv(eid)

    def _borrow_l2_for_conv(self) -> list[int]:
        """Borrow one L2 block, secondary-split into conv starts."""
        if self.expert_free_set:
            eid = self.expert_free_set.pop()
            return self._split_expert_to_conv(eid)
        if not self.ssm_is_l3 and self.ssm_free_set:
            sid = self.ssm_free_set.pop()
            return self._split_ssm_to_conv(sid)
        new_expert = self._pop_huge_and_split_to_expert()
        eid = new_expert[0]
        self.expert_free_set.update(new_expert[1:])
        return self._split_expert_to_conv(eid)

    def _borrow_expert_for_ssm_sub(self) -> list[int]:
        """Borrow one expert L2 block, secondary-split into SSM sub IDs."""
        if self.expert_free_set:
            eid = self.expert_free_set.pop()
            return self._split_expert_to_ssm_sub(eid)
        new_expert = self._pop_huge_and_split_to_expert()
        eid = new_expert[0]
        self.expert_free_set.update(new_expert[1:])
        return self._split_expert_to_ssm_sub(eid)

    # ------------------------------------------------------------------
    # Public typed allocation
    # ------------------------------------------------------------------

    def allocate_expert_blocks(self, n: int) -> list[int]:
        """Return *n* expert typed logical IDs."""
        self._ensure_expert_blocks(n)
        result: list[int] = []
        for _ in range(n):
            eid = self.expert_free_set.pop()
            self.l2_tracker[eid].is_free = False
            result.append(eid)
        return result

    def allocate_kv_blocks(self, n: int) -> list[int]:
        """Return *n* KV typed logical IDs."""
        if n <= 0:
            return []
        result: list[int] = []
        while len(result) < n:
            take = min(n - len(result), len(self.kv_free_set))
            for _ in range(take):
                result.append(self.kv_free_set.pop())
            if len(result) >= n:
                break
            new_ids = self._borrow_l2_for_kv()
            self.kv_free_set.update(new_ids)
        return result

    def allocate_ssm_blocks(self, n: int) -> list[int]:
        """Return *n* SSM typed logical IDs."""
        if n <= 0:
            return []
        if self.ssm_is_l3:
            result: list[int] = []
            while len(result) < n:
                take = min(n - len(result), len(self.ssm_sub_free_set))
                for _ in range(take):
                    result.append(self.ssm_sub_free_set.pop())
                if len(result) >= n:
                    break
                new_ids = self._borrow_expert_for_ssm_sub()
                self.ssm_sub_free_set.update(new_ids)
            return result
        else:
            self._ensure_ssm_l2_blocks(n)
            result: list[int] = []
            for _ in range(n):
                sid = self.ssm_free_set.pop()
                self.l2_tracker[sid].is_free = False
                result.append(sid)
            return result

    def allocate_conv_starts(self, n: int) -> list[int]:
        """Return *n* conv start row indices."""
        if n <= 0:
            return []
        result: list[int] = []
        while len(result) < n:
            take = min(n - len(result), len(self.conv_free_set))
            for _ in range(take):
                result.append(self.conv_free_set.pop())
            if len(result) >= n:
                break
            new_ids = self._borrow_l2_for_conv()
            self.conv_free_set.update(new_ids)
        return result

    # ------------------------------------------------------------------
    # Public typed free
    # ------------------------------------------------------------------

    def free_expert_blocks(self, expert_ids: list[int]) -> None:
        for eid in expert_ids:
            tracker = self.l2_tracker.get(eid)
            if tracker is None:
                continue
            tracker.is_free = True
            self.expert_free_set.add(eid)
            self._try_reclaim_huge(tracker.huge_block_id)

    def free_kv_blocks(self, kv_ids: list[int]) -> None:
        for kid in kv_ids:
            self.kv_free_set.add(kid)
            huge_id = kid // self.num_kv_per_huge
            l2_type = self.huge_l2_type.get(huge_id)
            if l2_type == "expert":
                l2_id = kid // self.num_kv_per_expert
            elif l2_type == "ssm":
                l2_id = kid // self.num_kv_per_ssm
            else:
                continue
            tracker = self.l2_tracker.get(l2_id)
            if tracker is not None and tracker.split_into == "kv":
                tracker.freed_sub_blocks += 1
                if self._all_l3_sub_blocks_free(l2_id, tracker):
                    self._reclaim_l2(l2_id, tracker)

    def free_ssm_blocks(self, ssm_ids: list[int]) -> None:
        if self.ssm_is_l3:
            for sid in ssm_ids:
                self.ssm_sub_free_set.add(sid)
                l2_id = sid // self.num_ssm_per_expert
                tracker = self.l2_tracker.get(l2_id)
                if tracker is not None and tracker.split_into == "ssm_sub":
                    tracker.freed_sub_blocks += 1
                    if self._all_l3_sub_blocks_free(l2_id, tracker):
                        self._reclaim_l2(l2_id, tracker)
        else:
            for sid in ssm_ids:
                tracker = self.l2_tracker.get(sid)
                if tracker is None:
                    continue
                tracker.is_free = True
                self.ssm_free_set.add(sid)
                self._try_reclaim_huge(tracker.huge_block_id)

    def free_conv_starts(self, conv_starts: list[int]) -> None:
        for cs in conv_starts:
            self.conv_free_set.add(cs)
            huge_id = cs // (self.num_conv_per_huge * self.conv_kernel_size)
            l2_type = self.huge_l2_type.get(huge_id)
            if l2_type == "expert":
                l2_id = cs // (self.num_conv_per_expert *
                               self.conv_kernel_size)
            elif l2_type == "ssm":
                l2_id = cs // (self.num_conv_per_ssm * self.conv_kernel_size)
            else:
                continue
            tracker = self.l2_tracker.get(l2_id)
            if tracker is not None and tracker.split_into == "conv":
                tracker.freed_sub_blocks += 1
                if self._all_l3_sub_blocks_free(l2_id, tracker):
                    self._reclaim_l2(l2_id, tracker)

    # ------------------------------------------------------------------
    # Reclamation
    # ------------------------------------------------------------------

    def _reclaim_l2(self, l2_id: int, tracker: L2BlockTracker) -> None:
        """Reclaim an L2 block whose L3 sub-blocks are all freed."""
        self._remove_l3_from_free_sets(l2_id, tracker)
        tracker.split_into = None
        tracker.total_sub_blocks = 0
        tracker.freed_sub_blocks = 0
        tracker.is_free = True
        if tracker.l2_type == "expert":
            self.expert_free_set.add(l2_id)
        elif tracker.l2_type == "ssm":
            self.ssm_free_set.add(l2_id)
        self._try_reclaim_huge(tracker.huge_block_id)

    def _remove_l3_from_free_sets(
        self, l2_id: int, tracker: L2BlockTracker
    ) -> None:
        if tracker.l2_type == "expert":
            if tracker.split_into == "kv":
                base = l2_id * self.num_kv_per_expert
                for j in range(self.num_kv_per_expert):
                    self.kv_free_set.discard(base + j)
            elif tracker.split_into == "conv":
                base = l2_id * self.num_conv_per_expert * self.conv_kernel_size
                for j in range(self.num_conv_per_expert):
                    self.conv_free_set.discard(
                        base + j * self.conv_kernel_size)
            elif tracker.split_into == "ssm_sub":
                base = l2_id * self.num_ssm_per_expert
                for j in range(self.num_ssm_per_expert):
                    self.ssm_sub_free_set.discard(base + j)
        elif tracker.l2_type == "ssm":
            if tracker.split_into == "kv":
                base = l2_id * self.num_kv_per_ssm
                for j in range(self.num_kv_per_ssm):
                    self.kv_free_set.discard(base + j)
            elif tracker.split_into == "conv":
                base = l2_id * self.num_conv_per_ssm * self.conv_kernel_size
                for j in range(self.num_conv_per_ssm):
                    self.conv_free_set.discard(
                        base + j * self.conv_kernel_size
                    )

    def _all_l3_sub_blocks_free(self, l2_id: int, tracker: L2BlockTracker) -> bool:
        """Return whether all L3 sub-blocks of an L2 block are currently free."""
        if tracker.l2_type == "expert":
            if tracker.split_into == "kv":
                base = l2_id * self.num_kv_per_expert
                for j in range(self.num_kv_per_expert):
                    if (base + j) not in self.kv_free_set:
                        return False
                return True
            if tracker.split_into == "conv":
                base = l2_id * self.num_conv_per_expert * self.conv_kernel_size
                for j in range(self.num_conv_per_expert):
                    if (base + j * self.conv_kernel_size) not in self.conv_free_set:
                        return False
                return True
            if tracker.split_into == "ssm_sub":
                base = l2_id * self.num_ssm_per_expert
                for j in range(self.num_ssm_per_expert):
                    if (base + j) not in self.ssm_sub_free_set:
                        return False
                return True
            return False

        if tracker.l2_type == "ssm":
            if tracker.split_into == "kv":
                base = l2_id * self.num_kv_per_ssm
                for j in range(self.num_kv_per_ssm):
                    if (base + j) not in self.kv_free_set:
                        return False
                return True
            if tracker.split_into == "conv":
                base = l2_id * self.num_conv_per_ssm * self.conv_kernel_size
                for j in range(self.num_conv_per_ssm):
                    if (base + j * self.conv_kernel_size) not in self.conv_free_set:
                        return False
                return True
            return False

        return False

    def _try_reclaim_huge(self, huge_id: int) -> None:
        """If every L2 block in *huge_id* is free, reclaim the whole thing."""
        l2_ids = self.huge_to_l2_ids.get(huge_id)
        if l2_ids is None:
            return
        for l2_id in l2_ids:
            tracker = self.l2_tracker.get(l2_id)
            if tracker is None or not tracker.is_free:
                return
        l2_type = self.huge_l2_type[huge_id]
        for l2_id in l2_ids:
            if l2_type == "expert":
                self.expert_free_set.discard(l2_id)
            elif l2_type == "ssm":
                self.ssm_free_set.discard(l2_id)
            del self.l2_tracker[l2_id]
        del self.huge_to_l2_ids[huge_id]
        del self.huge_l2_type[huge_id]
        blk = self.huge_blocks[huge_id]
        blk.ref_cnt -= 1
        self.huge_free_set.add(huge_id)

    # ------------------------------------------------------------------
    # Utility / compat
    # ------------------------------------------------------------------

    def estimate_available_kv_capacity(self) -> int:
        """Conservative estimate of how many KV blocks can still be allocated."""
        avail = len(self.kv_free_set)
        avail += len(self.expert_free_set) * self.num_kv_per_expert
        if not self.ssm_is_l3 and self.num_kv_per_ssm > 0:
            avail += len(self.ssm_free_set) * self.num_kv_per_ssm
        avail += len(self.huge_free_set) * self.num_kv_per_huge
        return avail

    def get_usage(self) -> float:
        # print("[debug] free huge blocks:", len(self.huge_free_set), self.num_huge_blocks)
        # print("[debug] free expert blocks:", len(self.expert_free_set))
        # print("[debug] free ssm blocks:", len(self.ssm_free_set))
        # print("[debug] free kv blocks:", len(self.kv_free_set))
        # print("[debug] free conv blocks:", len(self.conv_free_set))
        # print("[debug] free ssm sub blocks:", len(self.ssm_sub_free_set))
        if self.num_huge_blocks == 0:
            return 0.0
        return 1.0 - (len(self.huge_free_set) / self.num_huge_blocks)
    

    def snapshot_allocation_stats(self) -> BlockPoolAllocationSnapshot:
        """Return per-queue allocated block counts for observability.

        Huge is split on demand; only L2 blocks that have been secondary-split
        contribute KV / conv / SSM-sub slots. In-use count is
        ``materialized_slots - len(free_set)`` for each L3 type.
        SSM as L2 uses whole-block alloc (``split_into is None``).
        """
        num_expert_whole = 0
        materialized_kv = 0
        materialized_conv = 0
        materialized_ssm_sub = 0
        num_ssm_l2_whole_alloc = 0

        for tracker in self.l2_tracker.values():
            if (
                tracker.l2_type == "expert"
                and not tracker.is_free
                and tracker.split_into is None
            ):
                num_expert_whole += 1

            split = tracker.split_into
            if split == "kv":
                materialized_kv += tracker.total_sub_blocks
            elif split == "conv":
                materialized_conv += tracker.total_sub_blocks
            elif split == "ssm_sub":
                materialized_ssm_sub += tracker.total_sub_blocks
            elif (
                self.has_mamba
                and not self.ssm_is_l3
                and tracker.l2_type == "ssm"
                and split is None
                and not tracker.is_free
            ):
                num_ssm_l2_whole_alloc += 1

        num_kv_alloc = max(0, materialized_kv - len(self.kv_free_set))

        if self.has_mamba and self.num_conv_per_huge > 0:
            num_conv_alloc = max(0, materialized_conv - len(self.conv_free_set))
        else:
            num_conv_alloc = 0

        if not self.has_mamba:
            num_ssm_alloc = 0
        elif self.ssm_is_l3:
            num_ssm_alloc = max(0,
                                 materialized_ssm_sub - len(self.ssm_sub_free_set))
        else:
            num_ssm_alloc = num_ssm_l2_whole_alloc

        return BlockPoolAllocationSnapshot(
            num_huge_blocks=self.num_huge_blocks,
            num_free_huge_blocks=len(self.huge_free_set),
            num_expert_l2_whole_alloc=num_expert_whole,
            num_kv_blocks_alloc=num_kv_alloc,
            num_conv_starts_alloc=num_conv_alloc,
            num_ssm_blocks_alloc=num_ssm_alloc,
        )

    def take_events(self) -> list[KVCacheEvent]:
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    # Prefix-cache stubs (disabled per design, kept for API compat)
    def get_cached_block(self, block_hash: Any, kv_cache_group_ids: list[int]) -> None:
        return None

    def cache_full_blocks(self, *args: Any, **kwargs: Any) -> None:
        pass

    def touch(self, blocks: Sequence[Any]) -> None:
        pass

    def evict_blocks(self, block_ids: set[int]) -> None:
        pass

    def reset_prefix_cache(self) -> bool:
        return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hierarchical BlockPool: Huge -> L2 (Expert/SSM) -> L3 (KV/Conv/SSM-sub).

All public allocation methods return *typed logical IDs* that directly
index into the corresponding backend tensor views.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import json
import os
import threading
import time
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


class BlockPool:
    """Hierarchical block pool managing Huge -> L2 -> L3 blocks.

    All typed allocation methods (``allocate_expert_blocks``,
    ``allocate_kv_blocks``, etc.) return plain ``int`` IDs that directly
    index the corresponding backend tensor view.

    Args:
        num_huge_blocks: Number of huge (L1) blocks.
        num_act_blocks: Number of activation blocks (reserved region).
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
        num_act_blocks: int,
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
        self.num_act_blocks = num_act_blocks
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

        # ---- Block objects (kept for ActivationManager compat) ----
        self.huge_blocks: list[Block] = [
            Block(idx) for idx in range(num_huge_blocks)
        ]
        self.blocks = self.huge_blocks  # alias for legacy access
        self.act_blocks: list[Block] = [
            Block(idx)
            for idx in range(num_huge_blocks, num_huge_blocks + num_act_blocks)
        ]
        self.free_act_block_queue = FreeBlockQueue(self.act_blocks)

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

        # ---- debug trace (always-on, fixed path) ----
        self._bp_log_path = "/root/workspace/mycode/results/blockpool_trace.log"
        self._bp_log_lock = threading.Lock()
        self._bp_log_fh = None
        self._dbg_seq = 0
        self._dbg_alloc_expert: set[int] = set()
        self._dbg_alloc_kv: set[int] = set()

        try:
            os.makedirs(os.path.dirname(self._bp_log_path), exist_ok=True)
            self._bp_log_fh = open(self._bp_log_path, "a", buffering=1)
            self._bp_log(
                "INIT",
                num_huge_blocks=num_huge_blocks,
                num_act_blocks=num_act_blocks,
                num_expert_per_huge=num_expert_per_huge,
                num_ssm_per_huge=num_ssm_per_huge,
                num_kv_per_huge=num_kv_per_huge,
                num_conv_per_huge=num_conv_per_huge,
                conv_kernel_size=conv_kernel_size,
                ssm_is_l3=ssm_is_l3,
            )
        except Exception as e:
            # Never fail allocation due to trace init problems.
            logger.warning("BlockPool trace init failed: %s", e)
            self._bp_log_fh = None

    def _bp_log(self, op: str, **fields: Any) -> None:
        """Best-effort JSONL trace to fixed file. Must not raise."""
        fh = self._bp_log_fh
        if fh is None:
            return
        try:
            with self._bp_log_lock:
                self._dbg_seq += 1
                payload = {
                    "ts_ns": time.time_ns(),
                    "seq": self._dbg_seq,
                    "op": op,
                    **fields,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _bp_stomp(self, reason: str, **fields: Any) -> None:
        self._bp_log("STOMP", reason=reason, **fields)

    # ------------------------------------------------------------------
    # Huge-block interface (used by ActivationManager)
    # ------------------------------------------------------------------

    # [TODO] this method will be deprecated
    def get_num_free_blocks(self) -> int:
        """Number of free huge blocks."""
        return len(self.huge_free_set)

    # [TODO] this method will return whether needed blocks can be allocated
    def check_allocation_status(self, num_kv_needed, num_mamba_needed, num_act_needed):
        """
        Checks if there are sufficient blocks for the request.

        Returns:
            is_possible (bool): True if the request can be satisfied (with or without preemption).
            should_preempt (bool): True if activation blocks need to be preempted to fulfill the request.
        """
        # calculate kv gap, conv gap/ssm gap(maybe)
        kv_gap = max(0, num_kv_needed - len(self.kv_free_set))
        conv_gap = 0
        ssm_gap = 0
        free_l2_ssms = 0

        if self.has_mamba:
            conv_gap = max(0, num_mamba_needed - len(self.conv_free_set))
            if self.ssm_is_l3:
                ssm_gap = max(0, num_mamba_needed - len(self.ssm_sub_free_set))
            else:
                ssm_gap = max(0, num_mamba_needed - len(self.ssm_free_set))
                free_l2_ssms = max(0, len(self.ssm_free_set) - num_mamba_needed)
        
        free_experts = len(self.expert_free_set)
        
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
            conv_gap = max(0, conv_gap - used_exp_for_conv * self.num_conv_per_expert)
            
            if self.ssm_is_l3:
                needed_exp_for_ssm = cdiv(ssm_gap, self.num_ssm_per_expert)
                used_exp_for_ssm = min(free_experts, needed_exp_for_ssm)
                free_experts -= used_exp_for_ssm
                ssm_gap = max(0, ssm_gap - used_exp_for_ssm * self.num_ssm_per_expert)

        # if ssm is L2 block than it can also cover kv/conv block
        if self.has_mamba and not self.ssm_is_l3:
            needed_ssm_for_kv = cdiv(kv_gap, self.num_kv_per_ssm)
            used_ssm_for_kv = min(free_l2_ssms, needed_ssm_for_kv)
            free_l2_ssms -= used_ssm_for_kv
            kv_gap = max(0, kv_gap - used_ssm_for_kv * self.num_kv_per_ssm) 

            needed_ssm_for_conv = cdiv(conv_gap, self.num_conv_per_ssm)
            used_ssm_for_conv = min(free_l2_ssms, needed_ssm_for_conv)
            free_l2_ssms -= used_ssm_for_conv
            conv_gap = max(0, conv_gap - used_ssm_for_conv * self.num_conv_per_ssm) 
            
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
            needed_huge += cdiv(ssm_gap, self.num_ssm_per_huge )

        free_huge = len(self.huge_free_set)
        if needed_huge <= free_huge:
            return True, False
        
        # [TODO] for now we disable preempt logic 
        # huge_gap = needed_huge - free_huge
        # rest_act_blocks = self.activation_manager.get_rest_act_blocks()
        
        # if huge_gap + num_act_needed <= rest_act_blocks:
        #     return True, True
            
        return False, False


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
                self.huge_free_set.add(blk.block_id)

    def preempt_new_blocks(self, block_ids: list[int]) -> list[Block]:
        """Take blocks from the activation-reserved pool."""
        ret: list[Block] = []
        for bid in block_ids:
            blk = self.act_blocks[bid]
            self.free_act_block_queue.remove(blk)
            blk.ref_cnt += 1
            ret.append(blk)
        return ret

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
            t = self.l2_tracker[eid]
            stomp = False
            reason = None
            if eid in self._dbg_alloc_expert:
                stomp = True
                reason = "L2_in_use_as_expert_but_split_to_kv"
            elif (t.is_free is False) and (t.split_into is None):
                stomp = True
                reason = "L2_tracker_in_use_as_expert_but_split_to_kv"
            self._bp_log(
                "BORROW_L2_FOR_KV",
                l2_source="expert",
                l2_id=eid,
                huge_id=t.huge_block_id,
                l2_type=t.l2_type,
                is_free=t.is_free,
                split_into=t.split_into,
                freed_sub_blocks=t.freed_sub_blocks,
                total_sub_blocks=t.total_sub_blocks,
                expert_free=len(self.expert_free_set),
                ssm_free=len(self.ssm_free_set),
                kv_free=len(self.kv_free_set),
                stomp=stomp,
                reason=reason,
            )
            if stomp and reason is not None:
                self._bp_stomp(
                    reason,
                    l2_id=eid,
                    huge_id=t.huge_block_id,
                    l2_type=t.l2_type,
                    is_free=t.is_free,
                    split_into=t.split_into,
                )
            return self._split_expert_to_kv(eid)
        if not self.ssm_is_l3 and self.ssm_free_set:
            sid = self.ssm_free_set.pop()
            t = self.l2_tracker[sid]
            self._bp_log(
                "BORROW_L2_FOR_KV",
                l2_source="ssm",
                l2_id=sid,
                huge_id=t.huge_block_id,
                l2_type=t.l2_type,
                is_free=t.is_free,
                split_into=t.split_into,
                freed_sub_blocks=t.freed_sub_blocks,
                total_sub_blocks=t.total_sub_blocks,
                expert_free=len(self.expert_free_set),
                ssm_free=len(self.ssm_free_set),
                kv_free=len(self.kv_free_set),
            )
            return self._split_ssm_to_kv(sid)
        new_expert = self._pop_huge_and_split_to_expert()
        eid = new_expert[0]
        t = self.l2_tracker[eid]
        self._bp_log(
            "BORROW_L2_FOR_KV",
            l2_source="new_expert",
            l2_id=eid,
            huge_id=t.huge_block_id,
            l2_type=t.l2_type,
            is_free=t.is_free,
            split_into=t.split_into,
            expert_free=len(self.expert_free_set),
            ssm_free=len(self.ssm_free_set),
            kv_free=len(self.kv_free_set),
        )
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
            tracker = self.l2_tracker[eid]
            if eid in self._dbg_alloc_expert:
                self._bp_stomp(
                    "duplicate_alloc_expert",
                    eid=eid,
                    huge_id=tracker.huge_block_id,
                    split_into=tracker.split_into,
                    is_free=tracker.is_free,
                )
            if tracker.split_into is not None:
                # This L2 is already split into L3 blocks but is now allocated
                # as an expert L2. This indicates overlapping ownership.
                self._bp_stomp(
                    "alloc_expert_on_split_l2",
                    eid=eid,
                    huge_id=tracker.huge_block_id,
                    split_into=tracker.split_into,
                    is_free=tracker.is_free,
                    freed_sub_blocks=tracker.freed_sub_blocks,
                    total_sub_blocks=tracker.total_sub_blocks,
                )
            tracker.is_free = False
            self._dbg_alloc_expert.add(eid)
            result.append(eid)
        self._bp_log(
            "ALLOC_EXPERT",
            n=n,
            ids_head=result[:16],
            ids_len=len(result),
            expert_free=len(self.expert_free_set),
            huge_free=len(self.huge_free_set),
        )
        return result

    def allocate_kv_blocks(self, n: int) -> list[int]:
        """Return *n* KV typed logical IDs."""
        result: list[int] = []
        while len(result) < n:
            take = min(n - len(result), len(self.kv_free_set))
            for _ in range(take):
                result.append(self.kv_free_set.pop())
            if len(result) >= n:
                break
            new_ids = self._borrow_l2_for_kv()
            self.kv_free_set.update(new_ids)

        # Debug checks (post-allocation).
        dup_in_call = set()
        for kid in result:
            if kid in dup_in_call:
                self._bp_stomp("duplicate_kv_id_within_call", kid=kid)
            else:
                dup_in_call.add(kid)
            if kid in self._dbg_alloc_kv:
                self._bp_stomp("duplicate_alloc_kv", kid=kid)
            self._dbg_alloc_kv.add(kid)

            # Optional consistency: the owning L2 must be split into KV.
            if self.num_kv_per_huge > 0:
                huge_id = kid // self.num_kv_per_huge
                l2_type = self.huge_l2_type.get(huge_id)
                l2_id = None
                if l2_type == "expert" and self.num_kv_per_expert > 0:
                    l2_id = kid // self.num_kv_per_expert
                elif l2_type == "ssm" and self.num_kv_per_ssm > 0:
                    l2_id = kid // self.num_kv_per_ssm
                if l2_id is not None:
                    t = self.l2_tracker.get(l2_id)
                    if t is None or t.split_into != "kv":
                        self._bp_stomp(
                            "kv_alloc_from_non_kv_split_l2",
                            kid=kid,
                            huge_id=huge_id,
                            l2_type=l2_type,
                            l2_id=l2_id,
                            tracker_split_into=(None if t is None else t.split_into),
                            tracker_is_free=(None if t is None else t.is_free),
                        )

        self._bp_log(
            "ALLOC_KV",
            n=n,
            ids_head=result[:16],
            ids_len=len(result),
            kv_free=len(self.kv_free_set),
            expert_free=len(self.expert_free_set),
            ssm_free=len(self.ssm_free_set),
            huge_free=len(self.huge_free_set),
        )
        return result

    def allocate_ssm_blocks(self, n: int) -> list[int]:
        """Return *n* SSM typed logical IDs."""
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
        self._bp_log(
            "FREE_EXPERT",
            n=len(expert_ids),
            ids_head=expert_ids[:16],
            ids_len=len(expert_ids),
        )
        for eid in expert_ids:
            tracker = self.l2_tracker.get(eid)
            if tracker is None:
                continue
            if eid not in self._dbg_alloc_expert:
                self._bp_stomp(
                    "free_expert_without_alloc",
                    eid=eid,
                    huge_id=tracker.huge_block_id,
                    split_into=tracker.split_into,
                    is_free=tracker.is_free,
                )
            else:
                self._dbg_alloc_expert.discard(eid)
            tracker.is_free = True
            self.expert_free_set.add(eid)
            self._try_reclaim_huge(tracker.huge_block_id)

    def free_kv_blocks(self, kv_ids: list[int]) -> None:
        self._bp_log(
            "FREE_KV",
            n=len(kv_ids),
            ids_head=kv_ids[:16],
            ids_len=len(kv_ids),
        )
        for kid in kv_ids:
            if kid not in self._dbg_alloc_kv:
                self._bp_stomp("free_kv_without_alloc", kid=kid)
            else:
                self._dbg_alloc_kv.discard(kid)
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
                self._bp_log(
                    "L2_KV_PROGRESS",
                    l2_id=l2_id,
                    huge_id=tracker.huge_block_id,
                    l2_type=tracker.l2_type,
                    freed_sub_blocks=tracker.freed_sub_blocks,
                    total_sub_blocks=tracker.total_sub_blocks,
                )
                if tracker.freed_sub_blocks == tracker.total_sub_blocks:
                    self._reclaim_l2(l2_id, tracker)

    def free_ssm_blocks(self, ssm_ids: list[int]) -> None:
        if self.ssm_is_l3:
            for sid in ssm_ids:
                self.ssm_sub_free_set.add(sid)
                l2_id = sid // self.num_ssm_per_expert
                tracker = self.l2_tracker.get(l2_id)
                if tracker is not None and tracker.split_into == "ssm_sub":
                    tracker.freed_sub_blocks += 1
                    if tracker.freed_sub_blocks == tracker.total_sub_blocks:
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
                l2_id = cs // (self.num_conv_per_expert * self.conv_kernel_size)
            elif l2_type == "ssm":
                l2_id = cs // (self.num_conv_per_ssm * self.conv_kernel_size)
            else:
                continue
            tracker = self.l2_tracker.get(l2_id)
            if tracker is not None and tracker.split_into == "conv":
                tracker.freed_sub_blocks += 1
                if tracker.freed_sub_blocks == tracker.total_sub_blocks:
                    self._reclaim_l2(l2_id, tracker)

    # ------------------------------------------------------------------
    # Reclamation
    # ------------------------------------------------------------------

    def _reclaim_l2(self, l2_id: int, tracker: L2BlockTracker) -> None:
        """Reclaim an L2 block whose L3 sub-blocks are all freed."""
        self._bp_log(
            "RECLAIM_L2_BEGIN",
            l2_id=l2_id,
            huge_id=tracker.huge_block_id,
            l2_type=tracker.l2_type,
            split_into=tracker.split_into,
            freed_sub_blocks=tracker.freed_sub_blocks,
            total_sub_blocks=tracker.total_sub_blocks,
            expert_free=len(self.expert_free_set),
            ssm_free=len(self.ssm_free_set),
            kv_free=len(self.kv_free_set),
            conv_free=len(self.conv_free_set),
            ssm_sub_free=len(self.ssm_sub_free_set),
        )
        self._remove_l3_from_free_sets(l2_id, tracker)
        tracker.split_into = None
        tracker.total_sub_blocks = 0
        tracker.freed_sub_blocks = 0
        tracker.is_free = True
        if tracker.l2_type == "expert":
            self.expert_free_set.add(l2_id)
        elif tracker.l2_type == "ssm":
            self.ssm_free_set.add(l2_id)
        self._bp_log(
            "RECLAIM_L2_END",
            l2_id=l2_id,
            huge_id=tracker.huge_block_id,
            l2_type=tracker.l2_type,
            expert_free=len(self.expert_free_set),
            ssm_free=len(self.ssm_free_set),
            huge_free=len(self.huge_free_set),
        )
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
                    self.conv_free_set.discard(base + j * self.conv_kernel_size)
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
        self._bp_log(
            "RECLAIM_HUGE_BEGIN",
            huge_id=huge_id,
            l2_type=l2_type,
            l2_ids_head=l2_ids[:16],
            l2_ids_len=len(l2_ids),
            huge_ref_cnt=self.huge_blocks[huge_id].ref_cnt,
        )
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
        self._bp_log(
            "RECLAIM_HUGE_END",
            huge_id=huge_id,
            l2_type=l2_type,
            huge_ref_cnt=blk.ref_cnt,
            huge_free=len(self.huge_free_set),
            expert_free=len(self.expert_free_set),
            ssm_free=len(self.ssm_free_set),
        )

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
        if self.num_huge_blocks == 0:
            return 0.0
        return 1.0 - (len(self.huge_free_set) / self.num_huge_blocks)

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

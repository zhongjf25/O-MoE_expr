# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

from vllm.config.utils import config


@config
class ExpertOffloadConfig:
    """Configuration for expert offloading."""

    cached_num_experts: int | None = None
    """The number of experts to cache.

    If ``None``, derive the initial per-layer cache size from the available
    expert-block budget after reserving the first MoE layer.
    """

    offload_expert: bool = False
    """If True, offload the expert."""

    offload_expert_limit: int = 0
    """The limit of the expert to offload."""

    dynamic_cache_enabled: bool = False
    """Enable dynamic expert cache sizing based on free block headroom."""

    expert_no_copy_compute: bool = False
    """If True, use the no-copy Triton compute path for offloaded experts."""

    expert_numa_binding: bool = False
    """If True, bind expert offload CPU memory and prefetch thread to the NUMA
    node near this worker's GPU (Linux: libnuma ``numa_set_preferred`` on the
    main thread before load, ``sched_setaffinity`` on the prefetch thread;
    requires ``libnuma`` / ``numactl`` package)."""

    def compute_hash(self) -> str:
        """Compute the hash of the expert offload config."""
        return hashlib.md5(str(self).encode(), usedforsecurity=False).hexdigest()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class ExpertOffloadConfig:
    """Configuration for expert offloading."""

    cached_num_experts: int = 0
    """The number of experts to cache."""

    offload_expert: bool = False
    """If True, offload the expert."""

    offload_expert_limit: int = 0
    """The limit of the expert to offload."""

    dynamic_cache_enabled: bool = False
    """Enable dynamic expert cache sizing based on free block headroom."""

    dynamic_cache_adjust_interval: int = 5
    """How often (in engine steps) to re-evaluate expert cache capacity."""

    dynamic_cache_expand_threshold: int = 5
    """Expand expert cache when free block headroom exceeds this many experts."""

    dynamic_cache_shrink_threshold: int = 2
    """Shrink expert cache when free block headroom drops below this many experts."""

    def compute_hash(self) -> str:
        """Compute the hash of the expert offload config."""
        return hashlib.md5(str(self).encode(), usedforsecurity=False).hexdigest()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NUMA placement helpers for expert CPU offload (GPU-local node + CPU set)."""

from __future__ import annotations

import ctypes
import os
import platform
from collections import Counter
from dataclasses import dataclass
from vllm.logger import init_logger

logger = init_logger(__name__)

@dataclass(frozen=True)
class ExpertNumaContext:
    """CPU IDs near the local GPU (for sched_setaffinity) and a NUMA node id."""

    cpu_ids: frozenset[int]
    preferred_numa_node: int | None


def _affinity_ulonglongs_to_cpu_set(
    affinity: ctypes.Array,
) -> frozenset[int]:
    cpus: set[int] = set()
    for idx, word in enumerate(affinity):
        w = int(word)
        if w == 0:
            continue
        base = idx * 64
        for bit in range(64):
            if w & (1 << bit):
                cpus.add(base + bit)
    return frozenset(cpus)


def _nvml_resolve(
    physical_device_id: int,
) -> tuple[frozenset[int], int | None] | None:
    try:
        from vllm.platforms.cuda import with_nvml_context
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()

        @with_nvml_context
        def _query() -> tuple[frozenset[int], int | None]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            num_cpus = os.cpu_count() or 256
            set_size = max(1, (num_cpus + 63) // 64)
            aff = pynvml.nvmlDeviceGetCpuAffinity(handle, set_size)
            raw_cpus = _affinity_ulonglongs_to_cpu_set(aff)

            nvml_node: int | None = None
            try:
                n = int(pynvml.nvmlDeviceGetNumaNodeId(handle))
                if n >= 0:
                    nvml_node = n
            except Exception:
                pass
            return raw_cpus, nvml_node

        return _query()
    except Exception as e:
        logger.warning("expert_numa: NVML resolve failed: %s", e)
        return None


def _fallback_resolve(local_rank: int) -> tuple[frozenset[int], int | None] | None:
    try:
        from vllm.platforms.cpu import CpuPlatform

        allowed_nodes, logical_list = CpuPlatform.get_allowed_cpu_core_node_list()
        if not allowed_nodes:
            return None
        node = allowed_nodes[local_rank % len(allowed_nodes)]
        cpus = frozenset(
            x.id
            for x in logical_list
            if x.numa_node == node and x.id >= 0
        )
        return cpus, node
    except Exception as e:
        logger.warning("expert_numa: CPU topology fallback failed: %s", e)
        return None


def _majority_numa_from_cpus(
    cpu_ids: frozenset[int],
    logical_list: list,
) -> int | None:
    id_to_node = {x.id: x.numa_node for x in logical_list if x.id >= 0}
    nodes = [id_to_node[c] for c in cpu_ids if c in id_to_node]
    if not nodes:
        return None
    counts = Counter(nodes)
    best = counts.most_common(1)[0][0]
    if len(counts) > 1:
        logger.warning(
            "expert_numa: GPU CPU affinity spans NUMA nodes %s; using mode node %s",
            sorted(counts.keys()),
            best,
        )
    return best


def resolve_expert_numa_context(
    *,
    local_cuda_index: int,
    local_rank: int,
    physical_device_id: int | None = None,
) -> ExpertNumaContext | None:
    """Resolve NUMA node and CPU set for the given local CUDA device.

    ``physical_device_id`` should follow ``device_id_to_physical_device_id`` when
    provided; otherwise ``local_cuda_index`` is used as the NVML device index.
    """
    allowed_sched = os.sched_getaffinity(0)

    nvml_node: int | None = None
    raw_cpus: frozenset[int] | None = None

    phys = (
        physical_device_id
        if physical_device_id is not None
        else local_cuda_index
    )
    nv = _nvml_resolve(phys)
    if nv is not None:
        raw_cpus, nvml_node = nv

    if not raw_cpus:
        fb = _fallback_resolve(local_rank)
        if fb is None:
            logger.warning("expert_numa: could not resolve CPU affinity")
            return None
        raw_cpus, nvml_node = fb

    cpu_ids = frozenset(c for c in raw_cpus if c in allowed_sched)
    if not cpu_ids:
        logger.warning(
            "expert_numa: intersection of NVML affinity and sched_getaffinity is empty"
        )
        return None

    preferred_node: int | None = nvml_node
    if preferred_node is None:
        try:
            from vllm.platforms.cpu import CpuPlatform

            _allowed_nodes, logical_list = (
                CpuPlatform.get_allowed_cpu_core_node_list()
            )
            preferred_node = _majority_numa_from_cpus(cpu_ids, logical_list)
        except Exception as e:
            logger.debug("expert_numa: could not derive NUMA node from CPUs: %s", e)

    if preferred_node is None:
        logger.warning(
            "expert_numa: no preferred NUMA node; skipping mem policy (prefetch "
            "CPU binding still applied)"
        )

    return ExpertNumaContext(cpu_ids=cpu_ids, preferred_numa_node=preferred_node)


def _load_libnuma():
    for name in ("libnuma.so.1", "libnuma.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def apply_expert_numa_memory_policy(preferred_numa_node: int | None) -> bool:
    """Set preferred NUMA node for the calling thread (Linux, ``libnuma``).

    Uses ``numa_set_preferred(3)`` instead of ``set_mempolicy`` via ``libc``:
    ``set_mempolicy`` is often not exported as a plain symbol in ``libc.so.6`` for
    ``ctypes``, which causes ``AttributeError`` at runtime.
    """
    if platform.system() != "Linux":
        return False
    if preferred_numa_node is None or preferred_numa_node < 0:
        return False

    libnuma = _load_libnuma()
    if libnuma is None:
        logger.warning(
            "expert_numa: libnuma not found (install the ``numactl`` package on "
            "Debian/Ubuntu); skipping preferred NUMA memory policy"
        )
        return False

    libnuma.numa_available.argtypes = []
    libnuma.numa_available.restype = ctypes.c_int
    if int(libnuma.numa_available()) < 0:
        logger.warning("expert_numa: libnuma reports NUMA not available on this system")
        return False

    libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
    libnuma.numa_set_preferred.restype = None
    libnuma.numa_set_preferred(int(preferred_numa_node))

    logger.info(
        "expert_numa: numa_set_preferred for NUMA node %s",
        preferred_numa_node,
    )
    return True


def maybe_apply_expert_numa_for_worker(
    vllm_config,
    *,
    local_cuda_index: int,
    local_rank: int,
) -> None:
    """Apply memory policy before model load when expert NUMA binding is enabled."""
    eo = getattr(vllm_config, "expert_offload_config", None)
    if eo is None or not getattr(eo, "offload_expert", False):
        return
    if not getattr(eo, "expert_numa_binding", False):
        return
    if platform.system() != "Linux":
        logger.warning("expert_numa_binding is Linux-only; skipping")
        return

    try:
        from vllm.platforms import current_platform

        phys = current_platform.device_id_to_physical_device_id(local_cuda_index)
    except Exception as e:
        logger.warning("expert_numa: device_id_to_physical_device_id failed: %s", e)
        phys = local_cuda_index

    ctx = resolve_expert_numa_context(
        local_cuda_index=local_cuda_index,
        local_rank=local_rank,
        physical_device_id=phys,
    )
    if ctx is None:
        return
    if ctx.preferred_numa_node is None:
        return
    apply_expert_numa_memory_policy(ctx.preferred_numa_node)


def maybe_bind_prefetch_thread_expert_numa(
    vllm_config,
    *,
    local_cuda_index: int,
    local_rank: int,
) -> None:
    """Bind the calling thread to GPUs' local CPUs (``sched_setaffinity``)."""
    eo = getattr(vllm_config, "expert_offload_config", None)
    if eo is None or not getattr(eo, "offload_expert", False):
        return
    if not getattr(eo, "expert_numa_binding", False):
        return
    if platform.system() != "Linux" or not hasattr(os, "sched_setaffinity"):
        return

    try:
        from vllm.platforms import current_platform

        phys = current_platform.device_id_to_physical_device_id(local_cuda_index)
    except Exception:
        phys = local_cuda_index

    ctx = resolve_expert_numa_context(
        local_cuda_index=local_cuda_index,
        local_rank=local_rank,
        physical_device_id=phys,
    )
    if ctx is None or not ctx.cpu_ids:
        return
    try:
        os.sched_setaffinity(0, ctx.cpu_ids)
    except OSError as e:
        logger.warning("expert_numa: sched_setaffinity for prefetch thread failed: %s", e)
        return
    logger.info_once(
        "expert_numa: prefetch thread CPU affinity set (%d CPUs)",
        len(ctx.cpu_ids),
        scope="local",
    )

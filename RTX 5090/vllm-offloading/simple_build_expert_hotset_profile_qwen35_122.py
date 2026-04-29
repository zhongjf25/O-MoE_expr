#!/usr/bin/env python3

"""Build an offline expert hotset profile for MoE expert caching."""

import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.benchmarks.datasets import ShareGPTDataset
from vllm.v1.core.expert_hotset import (
    EXPERT_HOTSET_COLLECT_MODE_KEY,
    EXPERT_HOTSET_USE_CONFIGURED_CACHE_KEY,
    build_expert_hotset_profile,
    get_expert_hotset_profile_path,
    write_expert_hotset_profile,
)

# Edit these defaults in-place when switching models or profiling settings.
MODEL = "/root/autodl-tmp/models/qwen35_122b"
DATASET_PATH = "/root/autodl-tmp/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
GPU_MEMORY_UTILIZATION = 0.7
CACHED_NUM_EXPERTS = 128
OFFLOAD_EXPERT_LIMIT = 0
TP_SIZE = 8
ENFORCE_EAGER = True
ENABLE_PREFIX_CACHING = False
ENABLE_CHUNKED_PREFILL = False
EXPERT_NO_COPY_COMPUTE = False
ASYNC_SCHEDULING = False
# In collect mode, cache expert ids [0, CACHED_NUM_EXPERTS) for each
# adjustable layer. Setting CACHED_NUM_EXPERTS=num_experts is equivalent
# to caching all experts.
USE_CONFIGURED_CACHE_DURING_PROFILE = True
REQUEST_SEED = 0
MIN_REQUESTS = 100
CHECK_INTERVAL = 25
MAX_REQUESTS = 500
TV_THRESHOLD = 0.02
STABLE_CHECKPOINTS_REQUIRED = 2
TEMPERATURE = 0.0
TOP_P = 1.0
PROFILE_BATCH_SIZE = 8
MAX_MODEL_LEN = 4096

def _request_prompt(sample):
    if sample.multi_modal_data is None:
        return sample.prompt
    return {
        "prompt": sample.prompt,
        "multi_modal_data": sample.multi_modal_data,
    }


def _build_sampling_params(sample) -> SamplingParams:
    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=sample.expected_output_len,
    )


def _initialize_layer_counts(vllm_config) -> dict[int, list[int]]:
    hf_config = vllm_config.model_config.hf_text_config
    num_experts = (
        hf_config.num_experts
        if hasattr(hf_config, "num_experts")
        else hf_config.n_routed_experts
    )
    first_k_dense_replace = (
        hf_config.first_k_dense_replace
        if hasattr(hf_config, "first_k_dense_replace")
        else 0
    )
    return {
        layer_id: [0] * num_experts
        for layer_id in range(first_k_dense_replace, hf_config.num_hidden_layers)
    }


def _copy_counts(layer_counts: dict[int, list[int]]) -> dict[int, list[int]]:
    return {layer_id: counts[:] for layer_id, counts in layer_counts.items()}


def _top_prefix(counts: list[int], size: int) -> tuple[int, ...]:
    ranking = sorted(range(len(counts)), key=lambda expert_id: (-counts[expert_id], expert_id))
    return tuple(ranking[:max(0, size)])


def _probabilities(counts: list[int]) -> np.ndarray:
    array = np.asarray(counts, dtype=np.float64)
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Hotset counts must be positive before convergence checks.")
    return array / total


def _fetch_layer_counts(
    llm: LLM,
    template_counts: dict[int, list[int]],
) -> dict[int, list[int]]:
    merged_counts = {
        layer_id: [0] * len(counts) for layer_id, counts in template_counts.items()
    }
    worker_counts = llm.collective_rpc("get_expert_hotset_counts")
    for counts_by_layer in worker_counts:
        for layer_id, counts in counts_by_layer.items():
            if layer_id not in merged_counts:
                continue
            if len(counts) != len(merged_counts[layer_id]):
                raise RuntimeError(
                    "Mismatched expert hotset count length for layer "
                    f"{layer_id}: expected {len(merged_counts[layer_id])}, "
                    f"found {len(counts)}."
                )
            layer_total = merged_counts[layer_id]
            for expert_id, count in enumerate(counts):
                layer_total[expert_id] += int(count)
    return merged_counts


def _check_convergence(
    previous_counts: dict[int, list[int]],
    current_counts: dict[int, list[int]],
    adjustable_layers: list[int],
) -> tuple[bool, dict[str, float], dict[str, list[int]]]:
    per_layer_tv: dict[str, float] = {}
    prefixes: dict[str, list[int]] = {}
    stable = True
    for layer_id in adjustable_layers:
        previous_probs = _probabilities(previous_counts[layer_id])
        current_probs = _probabilities(current_counts[layer_id])
        tv_distance = 0.5 * float(np.abs(current_probs - previous_probs).sum())
        per_layer_tv[str(layer_id)] = tv_distance
        previous_prefix = _top_prefix(previous_counts[layer_id], CACHED_NUM_EXPERTS)
        current_prefix = _top_prefix(current_counts[layer_id], CACHED_NUM_EXPERTS)
        prefixes[str(layer_id)] = list(current_prefix)
        if tv_distance > TV_THRESHOLD or current_prefix != previous_prefix:
            stable = False
    return stable, per_layer_tv, prefixes


def _progress_postfix(
    *,
    processed: int,
    next_check: int,
    stable_checkpoints: int,
    latest_tv: dict[str, float],
    converged: bool,
) -> dict[str, str]:
    postfix = {
        "next_check": str(next_check),
        "stable": f"{stable_checkpoints}/{STABLE_CHECKPOINTS_REQUIRED}",
    }
    if latest_tv:
        postfix["max_tv"] = f"{max(latest_tv.values()):.4f}"
    if converged:
        postfix["status"] = "converged"
    else:
        postfix["sampled"] = str(processed)
    return postfix


def main() -> None:
    additional_config = {EXPERT_HOTSET_COLLECT_MODE_KEY: True}
    if USE_CONFIGURED_CACHE_DURING_PROFILE:
        additional_config[EXPERT_HOTSET_USE_CONFIGURED_CACHE_KEY] = True

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=ENFORCE_EAGER,
        tensor_parallel_size=TP_SIZE,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        async_scheduling=ASYNC_SCHEDULING,
        cached_num_experts=CACHED_NUM_EXPERTS,
        offload_expert=True,
        offload_expert_limit=OFFLOAD_EXPERT_LIMIT,
        enable_dynamic_cache=False,
        expert_no_copy_compute=EXPERT_NO_COPY_COMPUTE,
        additional_config=additional_config,
        disable_log_stats=True,
        max_model_len=MAX_MODEL_LEN,
    )

    vllm_config = llm.llm_engine.vllm_config
    hf_config = vllm_config.model_config.hf_text_config
    num_experts = (
        hf_config.num_experts
        if hasattr(hf_config, "num_experts")
        else hf_config.n_routed_experts
    )
    first_k_dense_replace = (
        hf_config.first_k_dense_replace
        if hasattr(hf_config, "first_k_dense_replace")
        else 0
    )
    num_hidden_layers = hf_config.num_hidden_layers
    adjustable_layers = list(range(first_k_dense_replace + 1, num_hidden_layers))
    profile_path = get_expert_hotset_profile_path(vllm_config)

    dataset = ShareGPTDataset(
        dataset_path=DATASET_PATH,
        random_seed=REQUEST_SEED,
    )
    samples = dataset.sample(
        llm.get_tokenizer(),
        num_requests=MAX_REQUESTS,
        request_id_prefix="hotset-",
    )
    if len(samples) < MIN_REQUESTS:
        raise ValueError(
            f"Dataset only produced {len(samples)} valid requests; "
            f"need at least {MIN_REQUESTS}."
        )

    layer_counts = _initialize_layer_counts(vllm_config)
    llm.collective_rpc("reset_expert_hotset_counts")
    previous_checkpoint_counts: dict[int, list[int]] | None = None
    latest_tv: dict[str, float] = {}
    latest_prefixes: dict[str, list[int]] = {}
    stable_checkpoints = 0
    converged = False
    processed = 0
    total_requests = min(len(samples), MAX_REQUESTS)
    batch_size = max(1, PROFILE_BATCH_SIZE)

    with tqdm(
        total=total_requests,
        desc="Building hotset profile",
        unit="req",
        dynamic_ncols=True,
    ) as progress:
        progress.set_postfix(
            _progress_postfix(
                processed=processed,
                next_check=min(MIN_REQUESTS, total_requests),
                stable_checkpoints=stable_checkpoints,
                latest_tv=latest_tv,
                converged=converged,
            )
        )

        while processed < total_requests:
            target = MIN_REQUESTS if processed == 0 else min(
                processed + CHECK_INTERVAL,
                total_requests,
            )
            while processed < target:
                batch_end = min(processed + batch_size, target)
                batch = samples[processed:batch_end]
                prompts = [_request_prompt(sample) for sample in batch]
                sampling_params = [_build_sampling_params(sample) for sample in batch]
                llm.generate(prompts, sampling_params, use_tqdm=False)
                progress.update(batch_end - processed)
                processed = batch_end
                progress.set_postfix(
                    _progress_postfix(
                        processed=processed,
                        next_check=target,
                        stable_checkpoints=stable_checkpoints,
                        latest_tv=latest_tv,
                        converged=converged,
                    )
                )

            if not adjustable_layers:
                converged = processed >= MIN_REQUESTS
                progress.set_postfix(
                    _progress_postfix(
                        processed=processed,
                        next_check=processed,
                        stable_checkpoints=stable_checkpoints,
                        latest_tv=latest_tv,
                        converged=converged,
                    )
                )
                if converged:
                    break
                continue

            if processed < MIN_REQUESTS:
                continue

            layer_counts = _fetch_layer_counts(llm, layer_counts)

            if previous_checkpoint_counts is None:
                previous_checkpoint_counts = _copy_counts(layer_counts)
                next_check = min(processed + CHECK_INTERVAL, total_requests)
                progress.set_postfix(
                    _progress_postfix(
                        processed=processed,
                        next_check=next_check,
                        stable_checkpoints=stable_checkpoints,
                        latest_tv=latest_tv,
                        converged=converged,
                    )
                )
                continue

            stable, latest_tv, latest_prefixes = _check_convergence(
                previous_checkpoint_counts,
                layer_counts,
                adjustable_layers,
            )
            stable_checkpoints = stable_checkpoints + 1 if stable else 0
            previous_checkpoint_counts = _copy_counts(layer_counts)
            if stable_checkpoints >= STABLE_CHECKPOINTS_REQUIRED:
                converged = True

            next_check = processed if converged else min(
                processed + CHECK_INTERVAL,
                total_requests,
            )
            progress.set_postfix(
                _progress_postfix(
                    processed=processed,
                    next_check=next_check,
                    stable_checkpoints=stable_checkpoints,
                    latest_tv=latest_tv,
                    converged=converged,
                )
            )
            if converged:
                break

    layer_counts = _fetch_layer_counts(llm, layer_counts)
    convergence = {
        "min_requests": MIN_REQUESTS,
        "check_interval": CHECK_INTERVAL,
        "max_requests": MAX_REQUESTS,
        "tv_threshold": TV_THRESHOLD,
        "stable_checkpoints_required": STABLE_CHECKPOINTS_REQUIRED,
        "stable_checkpoints_observed": stable_checkpoints,
        "sampled_requests": processed,
        "latest_tv_by_layer": latest_tv,
        "latest_top_cached_prefix_by_layer": latest_prefixes,
    }
    profile = build_expert_hotset_profile(
        vllm_config=vllm_config,
        layer_counts=layer_counts,
        sampled_requests=processed,
        converged=converged,
        convergence=convergence,
    )
    write_expert_hotset_profile(profile, profile_path)

    print(f"Wrote expert hotset profile to {profile_path}")
    print(
        "sampled_requests=%d converged=%s stable_checkpoints=%d"
        % (processed, converged, stable_checkpoints)
    )


if __name__ == "__main__":
    main()

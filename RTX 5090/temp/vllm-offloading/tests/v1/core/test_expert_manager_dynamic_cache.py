import types

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.expert_hotset import (
    EXPERT_HOTSET_COLLECT_MODE_KEY,
    ExpertHotsetLayerProfile,
    ExpertHotsetProfile,
    get_expert_hotset_metadata,
)
from vllm.v1.core.expert_manager import ExpertManager

pytestmark = pytest.mark.cpu_test


def _make_vllm_config(
    *,
    num_experts: int = 4,
    cached_num_experts: int | None = 2,
    offload_expert_limit: int = 3,
    num_hidden_layers: int = 4,
):
    hf_config = types.SimpleNamespace(
        num_experts=num_experts,
        first_k_dense_replace=0,
        num_hidden_layers=num_hidden_layers,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
    )
    expert_offload_config = types.SimpleNamespace(
        cached_num_experts=cached_num_experts,
        offload_expert=True,
        offload_expert_limit=offload_expert_limit,
        dynamic_cache_enabled=True,
    )
    parallel_config = types.SimpleNamespace(
        tensor_parallel_size=1,
    )
    model_config = types.SimpleNamespace(
        hf_text_config=hf_config,
        model="test-moe",
        revision="main",
        compute_hash=lambda: "test-model-hash",
    )
    return types.SimpleNamespace(
        model_config=model_config,
        expert_offload_config=expert_offload_config,
        parallel_config=parallel_config,
        additional_config={EXPERT_HOTSET_COLLECT_MODE_KEY: True},
    )


def _make_hotset_profile(vllm_config) -> ExpertHotsetProfile:
    metadata = get_expert_hotset_metadata(vllm_config)
    num_experts = metadata["num_experts"]
    assert num_experts == 4
    layer_counts = {
        0: [25, 25, 25, 25],
        1: [30, 50, 15, 5],
        2: [20, 30, 40, 10],
        3: [10, 70, 15, 5],
    }
    layers = {}
    for layer_id in range(metadata["first_k_dense_replace"], metadata["num_hidden_layers"]):
        counts = layer_counts.get(layer_id, layer_counts[1])
        total = sum(counts)
        ranking = tuple(
            sorted(range(num_experts), key=lambda expert_id: (-counts[expert_id], expert_id))
        )
        probabilities = tuple(count / total for count in counts)
        layers[layer_id] = ExpertHotsetLayerProfile(
            layer_id=layer_id,
            ranked_expert_ids=ranking,
            counts=tuple(counts),
            probabilities=probabilities,
        )
    return ExpertHotsetProfile(
        version=1,
        model_fingerprint=metadata["model_fingerprint"],
        model=metadata["model"],
        revision=metadata["revision"],
        num_hidden_layers=metadata["num_hidden_layers"],
        num_experts=metadata["num_experts"],
        first_k_dense_replace=metadata["first_k_dense_replace"],
        num_experts_per_tok=metadata["num_experts_per_tok"],
        sampled_requests=200,
        converged=True,
        convergence={"sampled_requests": 200},
        layers=layers,
    )


def _make_manager(
    *,
    num_huge_blocks: int,
    num_expert_per_huge: int = 6,
    **config_kwargs,
) -> ExpertManager:
    block_pool = BlockPool(
        num_huge_blocks=num_huge_blocks,
        num_expert_per_huge=num_expert_per_huge,
        num_ssm_per_huge=0,
        num_kv_per_huge=0,
        num_conv_per_huge=0,
        conv_kernel_size=0,
        ssm_is_l3=False,
    )
    vllm_config = _make_vllm_config(**config_kwargs)
    manager = ExpertManager(
        vllm_config,
        block_pool=block_pool,
        num_expert_per_huge=num_expert_per_huge,
    )
    manager.hotset_collect_mode = False
    manager.hotset_profile = _make_hotset_profile(vllm_config)
    manager.hotset_layers = dict(manager.hotset_profile.layers)
    manager.initialize_experts()
    return manager


def _make_manager_for_auto_sizing(
    *,
    num_huge_blocks: int,
    num_expert_per_huge: int = 1,
    **config_kwargs,
) -> ExpertManager:
    vllm_config = _make_vllm_config(**config_kwargs)
    block_pool = types.SimpleNamespace(num_huge_blocks=num_huge_blocks)
    return ExpertManager(
        vllm_config,
        block_pool=block_pool,
        num_expert_per_huge=num_expert_per_huge,
    )


def test_deferred_free_blocks_are_not_reused_until_delta_completion():
    block_pool = BlockPool(
        num_huge_blocks=2,
        num_expert_per_huge=6,
        num_ssm_per_huge=0,
        num_kv_per_huge=0,
        num_conv_per_huge=0,
        conv_kernel_size=0,
        ssm_is_l3=False,
    )
    manager = ExpertManager(
        _make_vllm_config(num_hidden_layers=2),
        block_pool=block_pool,
        num_expert_per_huge=6,
    )
    manager.hotset_collect_mode = False
    manager.hotset_profile = _make_hotset_profile(manager.vllm_config)
    manager.hotset_layers = dict(manager.hotset_profile.layers)

    old_mapping = manager.allocate_blocks([(1, 0)])
    old_block_ids = set(old_mapping.values())

    manager._reserve_release_expert_blocks([(1, 0)], delta_id=9)
    assert old_block_ids.issubset(set(manager._pending_release_by_delta[9]))
    assert old_block_ids.isdisjoint(block_pool.expert_free_set)

    new_mapping = manager.allocate_blocks([(1, 1)])
    new_block_ids = set(new_mapping.values())
    assert old_block_ids.isdisjoint(new_block_ids)

    manager.complete_cache_delta(9)
    assert old_block_ids.issubset(block_pool.expert_free_set)


def test_initialize_experts_uses_hotset_prefix():
    manager = _make_manager(num_huge_blocks=10, num_hidden_layers=4)

    assert manager.resident_experts_by_layer[0] == {0, 1, 2, 3}
    assert manager.resident_experts_by_layer[1] == {0, 1}
    assert manager.resident_experts_by_layer[2] == {1, 2}
    assert manager.resident_experts_by_layer[3] == {1, 2}


def test_initialize_experts_respects_explicit_zero_cached_num_experts():
    manager = _make_manager(
        num_huge_blocks=10,
        num_hidden_layers=4,
        cached_num_experts=0,
    )

    assert manager.resident_experts_by_layer[0] == {0, 1, 2, 3}
    assert manager.resident_experts_by_layer.get(1, set()) == set()
    assert manager.resident_experts_by_layer.get(2, set()) == set()
    assert manager.resident_experts_by_layer.get(3, set()) == set()


def test_auto_cached_num_experts_rounds_down_to_lower_multiple_of_five():
    manager = _make_manager_for_auto_sizing(
        num_huge_blocks=10475,
        num_expert_per_huge=1,
        num_experts=160,
        num_hidden_layers=28,
        cached_num_experts=None,
    )

    assert manager._resolve_cached_num_experts() == 120


def test_auto_cached_num_experts_keeps_extra_headroom_on_exact_multiple():
    manager = _make_manager_for_auto_sizing(
        num_huge_blocks=8643,
        num_expert_per_huge=1,
        num_experts=160,
        num_hidden_layers=28,
        cached_num_experts=None,
    )

    assert manager._resolve_cached_num_experts() == 95


def test_initialize_experts_is_noop_when_offload_disabled():
    block_pool = BlockPool(
        num_huge_blocks=2,
        num_expert_per_huge=6,
        num_ssm_per_huge=0,
        num_kv_per_huge=0,
        num_conv_per_huge=0,
        conv_kernel_size=0,
        ssm_is_l3=False,
    )
    vllm_config = _make_vllm_config(num_hidden_layers=4)
    vllm_config.expert_offload_config.offload_expert = False
    manager = ExpertManager(
        vllm_config,
        block_pool=block_pool,
        num_expert_per_huge=6,
    )
    manager.hotset_collect_mode = False
    manager.hotset_profile = None
    manager.hotset_layers = {}

    assert manager.initialize_experts() == {}
    assert manager.resident_experts_by_layer == {}


def test_inflight_delta_blocks_further_adjustments_until_completion():
    manager = _make_manager(num_huge_blocks=7, num_hidden_layers=4)

    first_delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)
    assert first_delta is not None

    second_delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)
    assert second_delta is None

    manager.complete_cache_delta(first_delta.delta_id)
    manager.block_pool.get_num_free_expert_blocks = lambda: 18

    third_delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)
    assert third_delta is not None


def test_shortfall_triggered_shrink_releases_coldest_prefix_experts():
    manager = _make_manager(num_huge_blocks=7, num_hidden_layers=4)

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=7)

    assert delta is not None
    assert delta.experts_to_load == []
    assert delta.experts_to_evict == [(3, 2), (1, 0), (2, 1)]
    assert delta.evict_commit_mode == "table"


def test_low_watermark_shrink_evicts_exactly_one_expert():
    manager = _make_manager(num_huge_blocks=7, num_hidden_layers=4)
    manager.block_pool.get_num_free_expert_blocks = lambda: 1

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is not None
    assert delta.experts_to_load == []
    assert delta.experts_to_evict == [(3, 2)]
    assert delta.evict_commit_mode == "row"


def test_shortfall_shrink_takes_precedence_over_low_watermark():
    manager = _make_manager(num_huge_blocks=7, num_hidden_layers=4)
    manager.block_pool.get_num_free_expert_blocks = lambda: 1

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=7)

    assert delta is not None
    assert len(delta.experts_to_evict) == 3
    assert delta.evict_commit_mode == "table"


@pytest.mark.parametrize(
    ("free_expert_blocks", "expected_loads"),
    [
        (12, 1),
        (15, 2),
        (18, 3),
    ],
)
def test_expand_budget_caps_growth_to_at_most_three_experts(
    free_expert_blocks: int,
    expected_loads: int,
):
    manager = _make_manager(num_huge_blocks=7, num_hidden_layers=4)
    manager.block_pool.get_num_free_expert_blocks = lambda: free_expert_blocks

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is not None
    assert delta.experts_to_evict == []
    assert len(delta.experts_to_load) == expected_loads
    assert len(delta.new_expert_to_block) == expected_loads * 3


def test_expand_prefers_largest_marginal_gain():
    manager = _make_manager(num_huge_blocks=10, num_hidden_layers=4)
    manager.block_pool.get_num_free_expert_blocks = lambda: 12

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is not None
    assert delta.experts_to_load == [(2, 0)]


def test_shrink_respects_per_layer_min_bound():
    manager = _make_manager(
        num_huge_blocks=6,
        num_hidden_layers=4,
        cached_num_experts=1,
        offload_expert_limit=3,
    )

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=1)

    assert delta is None


def test_expand_respects_per_layer_max_bound():
    manager = _make_manager(
        num_huge_blocks=10,
        num_hidden_layers=4,
        cached_num_experts=4,
        offload_expert_limit=3,
    )

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is None


def test_headroom_decision_uses_free_expert_blocks_not_free_huge_blocks():
    manager = _make_manager(num_huge_blocks=10, num_hidden_layers=4)
    manager.block_pool.get_num_free_blocks = lambda: 0
    manager.block_pool.get_num_free_expert_blocks = lambda: 9

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is None


def test_expand_can_use_free_expert_blocks_even_when_free_huge_blocks_are_zero():
    manager = _make_manager(num_huge_blocks=10, num_hidden_layers=4)
    manager.block_pool.get_num_free_blocks = lambda: 0
    manager.block_pool.get_num_free_expert_blocks = lambda: 12

    delta = manager.adjust_expert_cache_capacity(
        required_expert_shrink_blocks=0)

    assert delta is not None
    assert delta.experts_to_evict == []
    assert delta.experts_to_load == [(2, 0)]

import types

import pytest

from vllm.v1.core.block_manager import BlockManager

pytestmark = pytest.mark.cpu_test


def _make_block_manager(required_expert_blocks):
    manager = object.__new__(BlockManager)
    manager.max_model_len = 32
    manager.enable_caching = False
    manager._required_expert_shrink_blocks = 0

    coordinator = types.SimpleNamespace(
        remove_skipped_blocks=lambda request_id, total_tokens: None,
        get_num_blocks_to_allocate=lambda **kwargs: (8, 0),
        allocate_new_computed_blocks=lambda **kwargs: None,
        allocate_new_blocks=lambda *args, **kwargs: (),
        cache_blocks=lambda *args, **kwargs: None,
    )
    manager.kv_cache_manager = types.SimpleNamespace(
        coordinator=coordinator,
        empty_kv_cache_blocks=types.SimpleNamespace(blocks=()),
        create_kv_cache_blocks=lambda new_kv_blocks: new_kv_blocks,
    )
    manager.block_pool = types.SimpleNamespace(
        check_allocation_status=lambda **kwargs: False,
        get_required_expert_blocks_for_allocation=required_expert_blocks,
    )
    return manager


def _make_request():
    return types.SimpleNamespace(
        request_id="req-0",
        num_computed_tokens=0,
        num_tokens=8,
    )


def test_allocate_slots_records_required_expert_shrink_blocks_on_shortage():
    manager = _make_block_manager(
        required_expert_blocks=lambda **kwargs: 5,
    )

    blocks = BlockManager.allocate_slots(
        manager,
        _make_request(),
        num_new_tokens=8,
    )

    assert blocks is None
    assert manager.take_required_expert_shrink_blocks() == 5
    assert manager.take_required_expert_shrink_blocks() == 0


def test_allocate_slots_keeps_max_required_expert_shrink_blocks():
    required_values = iter([2, 5])
    manager = _make_block_manager(
        required_expert_blocks=lambda **kwargs: next(required_values),
    )

    BlockManager.allocate_slots(manager, _make_request(), num_new_tokens=8)
    BlockManager.allocate_slots(manager, _make_request(), num_new_tokens=8)

    assert manager.take_required_expert_shrink_blocks() == 5

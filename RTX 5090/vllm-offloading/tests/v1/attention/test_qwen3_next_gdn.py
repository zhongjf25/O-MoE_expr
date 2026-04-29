import types

import torch

import vllm.model_executor.models.qwen3_next as qwen3_next_module
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def _make_dummy_layer():
    def _rearrange_mixed_qkv(mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        tensor = torch.ones((1, 1, 1, 1), dtype=torch.float32)
        return tensor, tensor, tensor

    return types.SimpleNamespace(
        prefix="gdn_test",
        layer_idx=0,
        conv1d=types.SimpleNamespace(
            weight=torch.zeros((1, 1, 2), dtype=torch.float32),
            bias=torch.zeros((1,), dtype=torch.float32),
        ),
        activation="silu",
        kv_cache=(
            torch.zeros((4, 1), dtype=torch.float32),
            torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        ),
        A_log=torch.zeros((1,), dtype=torch.float32),
        dt_bias=torch.zeros((1,), dtype=torch.float32),
        rearrange_mixed_qkv=_rearrange_mixed_qkv,
        chunk_gated_delta_rule=None,
    )


def _patch_runtime(monkeypatch, metadata):
    monkeypatch.setattr(
        qwen3_next_module,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            attn_metadata={"gdn_test": metadata},
            virtual_engine=0,
        ),
    )
    monkeypatch.setattr(
        qwen3_next_module,
        "causal_conv1d_update",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        qwen3_next_module,
        "fused_gdn_gating",
        lambda *args, **kwargs: (
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        qwen3_next_module,
        "fused_recurrent_gated_delta_rule",
        lambda *args, **kwargs: (
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
            torch.ones((1, 1, 1, 1), dtype=torch.float32),
        ),
    )


def test_forward_core_handles_no_spec_metadata(monkeypatch):
    layer = _make_dummy_layer()
    non_spec_indices = torch.tensor([[0, 0]], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
        spec_query_start_loc=None,
        non_spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=(non_spec_indices,),
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
    )
    _patch_runtime(monkeypatch, metadata)

    core_attn_out = torch.zeros((1, 1, 1), dtype=torch.float32)
    Qwen3NextGatedDeltaNet._forward_core(
        layer,
        mixed_qkv=torch.ones((1, 3), dtype=torch.float32),
        b=torch.ones((1, 1), dtype=torch.float32),
        a=torch.ones((1, 1), dtype=torch.float32),
        core_attn_out=core_attn_out,
    )

    torch.testing.assert_close(core_attn_out, torch.ones_like(core_attn_out))


def test_forward_core_handles_all_spec_metadata(monkeypatch):
    layer = _make_dummy_layer()
    spec_indices = torch.tensor([[0, 0]], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=1,
        num_actual_tokens=1,
        spec_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        non_spec_query_start_loc=None,
        spec_state_indices_tensor=spec_indices,
        non_spec_state_indices_tensor=(None,),
        spec_sequence_masks=torch.tensor([True]),
        spec_token_indx=torch.tensor([0], dtype=torch.int32),
        non_spec_token_indx=torch.empty((0,), dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )
    _patch_runtime(monkeypatch, metadata)

    core_attn_out = torch.zeros((1, 1, 1), dtype=torch.float32)
    Qwen3NextGatedDeltaNet._forward_core(
        layer,
        mixed_qkv=torch.ones((1, 3), dtype=torch.float32),
        b=torch.ones((1, 1), dtype=torch.float32),
        a=torch.ones((1, 1), dtype=torch.float32),
        core_attn_out=core_attn_out,
    )

    torch.testing.assert_close(core_attn_out, torch.ones_like(core_attn_out))

import types

import pytest

from vllm.v1.core import expert_hotset


def _make_vllm_config(*, collect_mode: bool = False):
    hf_config = types.SimpleNamespace(
        num_experts=4,
        first_k_dense_replace=0,
        num_hidden_layers=4,
        num_experts_per_tok=2,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=hf_config,
            model="test-moe",
            revision="main",
            compute_hash=lambda: "test-model-hash",
        ),
        expert_offload_config=types.SimpleNamespace(offload_expert=True),
        additional_config=(
            {expert_hotset.EXPERT_HOTSET_COLLECT_MODE_KEY: True}
            if collect_mode
            else {}
        ),
    )


def test_load_expert_hotset_profile_missing_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(expert_hotset, "EXPERT_HOTSET_PROFILE_DIR", tmp_path)
    vllm_config = _make_vllm_config()
    expected_path = expert_hotset.get_expert_hotset_profile_path(vllm_config)

    with pytest.raises(ValueError, match="simple_build_expert_hotset_profile.py"):
        expert_hotset.load_expert_hotset_profile(vllm_config)

    assert str(expected_path).endswith(".json")
    assert expected_path.name == "test-moe.json"


def test_profile_path_uses_model_basename(monkeypatch, tmp_path):
    monkeypatch.setattr(expert_hotset, "EXPERT_HOTSET_PROFILE_DIR", tmp_path)
    vllm_config = _make_vllm_config()
    vllm_config.model_config.model = "/root/workspace/model_weights/test-moe/"

    profile_path = expert_hotset.get_expert_hotset_profile_path(vllm_config)

    assert profile_path == tmp_path / "test-moe.json"


def test_collect_mode_skips_hotset_profile_requirement(monkeypatch, tmp_path):
    monkeypatch.setattr(expert_hotset, "EXPERT_HOTSET_PROFILE_DIR", tmp_path)
    vllm_config = _make_vllm_config(collect_mode=True)

    assert not expert_hotset.requires_expert_hotset_profile(vllm_config)
    assert expert_hotset.load_expert_hotset_profile(vllm_config) is None


def test_build_and_load_expert_hotset_profile_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(expert_hotset, "EXPERT_HOTSET_PROFILE_DIR", tmp_path)
    vllm_config = _make_vllm_config()
    layer_counts = {
        0: [25, 25, 25, 25],
        1: [30, 50, 15, 5],
        2: [20, 30, 40, 10],
        3: [10, 70, 15, 5],
    }

    profile = expert_hotset.build_expert_hotset_profile(
        vllm_config=vllm_config,
        layer_counts=layer_counts,
        sampled_requests=150,
        converged=False,
        convergence={"sampled_requests": 150},
    )
    profile_path = expert_hotset.get_expert_hotset_profile_path(vllm_config)
    expert_hotset.write_expert_hotset_profile(profile, profile_path)

    loaded_profile = expert_hotset.load_expert_hotset_profile(vllm_config)

    assert loaded_profile is not None
    assert loaded_profile.model_fingerprint == profile.model_fingerprint
    assert loaded_profile.sampled_requests == 150
    assert loaded_profile.converged is False


def test_load_expert_hotset_profile_ignores_fingerprint_mismatch_for_same_model_name(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(expert_hotset, "EXPERT_HOTSET_PROFILE_DIR", tmp_path)
    vllm_config = _make_vllm_config()
    layer_counts = {
        0: [25, 25, 25, 25],
        1: [30, 50, 15, 5],
        2: [20, 30, 40, 10],
        3: [10, 70, 15, 5],
    }

    profile = expert_hotset.build_expert_hotset_profile(
        vllm_config=vllm_config,
        layer_counts=layer_counts,
        sampled_requests=150,
        converged=True,
        convergence={"sampled_requests": 150},
    )
    profile_path = expert_hotset.get_expert_hotset_profile_path(vllm_config)
    expert_hotset.write_expert_hotset_profile(profile, profile_path)

    vllm_config.model_config.compute_hash = lambda: "different-hash"

    loaded_profile = expert_hotset.load_expert_hotset_profile(vllm_config)

    assert loaded_profile is not None
    assert loaded_profile.sampled_requests == 150

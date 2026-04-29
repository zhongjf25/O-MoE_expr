"""Helpers for offline expert hotset profiles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

EXPERT_HOTSET_PROFILE_VERSION = 1
EXPERT_HOTSET_COLLECT_MODE_KEY = "expert_hotset_collect_mode"
EXPERT_HOTSET_USE_CONFIGURED_CACHE_KEY = "expert_hotset_use_configured_cache"
EXPERT_HOTSET_PROFILE_DIR = Path(__file__).resolve().parents[3] / (
    "expert_hotset_profiles"
)


@dataclass(frozen=True)
class ExpertHotsetLayerProfile:
    layer_id: int
    ranked_expert_ids: tuple[int, ...]
    counts: tuple[int, ...]
    probabilities: tuple[float, ...]

    def prefix(self, size: int) -> tuple[int, ...]:
        return self.ranked_expert_ids[:max(0, size)]

    def expand_gain(self, current_size: int) -> float:
        expert_id = self.next_expert(current_size)
        if expert_id is None:
            return -1.0
        return float(self.probabilities[expert_id])

    def shrink_loss(self, current_size: int) -> float:
        expert_id = self.shrink_expert(current_size)
        if expert_id is None:
            return float("inf")
        return float(self.probabilities[expert_id])

    def next_expert(self, current_size: int) -> int | None:
        if current_size >= len(self.ranked_expert_ids):
            return None
        return int(self.ranked_expert_ids[current_size])

    def shrink_expert(self, current_size: int) -> int | None:
        if current_size <= 0:
            return None
        return int(self.ranked_expert_ids[current_size - 1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_expert_ids": list(self.ranked_expert_ids),
            "counts": list(self.counts),
            "probabilities": list(self.probabilities),
        }

    @classmethod
    def from_dict(
        cls,
        layer_id: int,
        data: dict[str, Any],
    ) -> "ExpertHotsetLayerProfile":
        return cls(
            layer_id=int(layer_id),
            ranked_expert_ids=tuple(int(x) for x in data["ranked_expert_ids"]),
            counts=tuple(int(x) for x in data["counts"]),
            probabilities=tuple(float(x) for x in data["probabilities"]),
        )


@dataclass(frozen=True)
class ExpertHotsetProfile:
    version: int
    model_fingerprint: str
    model: str
    revision: str | None
    num_hidden_layers: int
    num_experts: int
    first_k_dense_replace: int
    num_experts_per_tok: int
    sampled_requests: int
    converged: bool
    convergence: dict[str, Any]
    layers: dict[int, ExpertHotsetLayerProfile]

    def validate_against(self, vllm_config: VllmConfig) -> None:
        metadata = get_expert_hotset_metadata(vllm_config)
        if self.version != EXPERT_HOTSET_PROFILE_VERSION:
            raise ValueError(
                "Unsupported expert hotset profile version "
                f"{self.version}; expected {EXPERT_HOTSET_PROFILE_VERSION}."
            )
        actual_model_name = _normalize_model_name(self.model)
        if actual_model_name != metadata["model_name"]:
            raise ValueError(
                "Expert hotset profile model name mismatch: "
                f"expected {metadata['model_name']}, "
                f"found {actual_model_name}."
            )
        if self.num_hidden_layers != metadata["num_hidden_layers"]:
            raise ValueError(
                "Expert hotset profile layer count mismatch: "
                f"expected {metadata['num_hidden_layers']}, "
                f"found {self.num_hidden_layers}."
            )
        if self.num_experts != metadata["num_experts"]:
            raise ValueError(
                "Expert hotset profile expert count mismatch: "
                f"expected {metadata['num_experts']}, found {self.num_experts}."
            )
        if self.first_k_dense_replace != metadata["first_k_dense_replace"]:
            raise ValueError(
                "Expert hotset profile first_k_dense_replace mismatch: "
                f"expected {metadata['first_k_dense_replace']}, "
                f"found {self.first_k_dense_replace}."
            )
        if self.num_experts_per_tok != metadata["num_experts_per_tok"]:
            raise ValueError(
                "Expert hotset profile num_experts_per_tok mismatch: "
                f"expected {metadata['num_experts_per_tok']}, "
                f"found {self.num_experts_per_tok}."
            )

        expected_layers = set(
            range(self.first_k_dense_replace, self.num_hidden_layers)
        )
        if set(self.layers) != expected_layers:
            raise ValueError(
                "Expert hotset profile layer set mismatch: "
                f"expected {sorted(expected_layers)}, found {sorted(self.layers)}."
            )

        expected_ranking = list(range(self.num_experts))
        expected_set = set(expected_ranking)
        for layer_id, layer_profile in self.layers.items():
            ranked = list(layer_profile.ranked_expert_ids)
            counts = list(layer_profile.counts)
            probabilities = list(layer_profile.probabilities)
            if len(ranked) != self.num_experts:
                raise ValueError(
                    "Expert hotset profile ranking length mismatch for layer "
                    f"{layer_id}: expected {self.num_experts}, found {len(ranked)}."
                )
            if len(counts) != self.num_experts:
                raise ValueError(
                    "Expert hotset profile count length mismatch for layer "
                    f"{layer_id}: expected {self.num_experts}, found {len(counts)}."
                )
            if len(probabilities) != self.num_experts:
                raise ValueError(
                    "Expert hotset profile probability length mismatch for layer "
                    f"{layer_id}: expected {self.num_experts}, "
                    f"found {len(probabilities)}."
                )
            if set(ranked) != expected_set:
                raise ValueError(
                    "Expert hotset profile ranking is not a permutation of expert "
                    f"ids for layer {layer_id}: {ranked}."
                )
            if any(count < 0 for count in counts):
                raise ValueError(
                    f"Expert hotset profile has negative counts for layer {layer_id}."
                )
            total_count = sum(counts)
            if total_count <= 0:
                raise ValueError(
                    "Expert hotset profile has zero observations for layer "
                    f"{layer_id}."
                )
            probability_sum = sum(probabilities)
            if abs(probability_sum - 1.0) > 1e-4:
                raise ValueError(
                    "Expert hotset profile probabilities must sum to 1 for layer "
                    f"{layer_id}; got {probability_sum}."
                )
            expected_sorted = sorted(
                expected_ranking,
                key=lambda expert_id: (-counts[expert_id], expert_id),
            )
            if ranked != expected_sorted:
                raise ValueError(
                    "Expert hotset profile ranking is inconsistent with counts "
                    f"for layer {layer_id}."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_fingerprint": self.model_fingerprint,
            "model": self.model,
            "revision": self.revision,
            "num_hidden_layers": self.num_hidden_layers,
            "num_experts": self.num_experts,
            "first_k_dense_replace": self.first_k_dense_replace,
            "num_experts_per_tok": self.num_experts_per_tok,
            "sampled_requests": self.sampled_requests,
            "converged": self.converged,
            "convergence": self.convergence,
            "layers": {
                str(layer_id): layer_profile.to_dict()
                for layer_id, layer_profile in sorted(self.layers.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpertHotsetProfile":
        return cls(
            version=int(data["version"]),
            model_fingerprint=str(data["model_fingerprint"]),
            model=str(data["model"]),
            revision=data.get("revision"),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_experts=int(data["num_experts"]),
            first_k_dense_replace=int(data["first_k_dense_replace"]),
            num_experts_per_tok=int(data["num_experts_per_tok"]),
            sampled_requests=int(data["sampled_requests"]),
            converged=bool(data["converged"]),
            convergence=dict(data.get("convergence", {})),
            layers={
                int(layer_id): ExpertHotsetLayerProfile.from_dict(layer_id, layer_data)
                for layer_id, layer_data in data["layers"].items()
            },
        )


def _hf_expert_metadata(vllm_config: VllmConfig) -> dict[str, int]:
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
    num_experts_per_tok = (
        hf_config.num_experts_per_tok
        if hasattr(hf_config, "num_experts_per_tok")
        else getattr(hf_config, "top_k", 0)
    )
    return {
        "num_experts": int(num_experts),
        "first_k_dense_replace": int(first_k_dense_replace),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "num_experts_per_tok": int(num_experts_per_tok),
    }


def _normalize_model_name(model: str) -> str:
    normalized = str(model).rstrip("/")
    if not normalized:
        raise ValueError("Model name/path cannot be empty when building hotset profile.")
    return normalized.split("/")[-1]


def get_expert_hotset_model_name(vllm_config: VllmConfig) -> str:
    return _normalize_model_name(vllm_config.model_config.model)


def get_expert_hotset_metadata(vllm_config: VllmConfig) -> dict[str, Any]:
    model_config = vllm_config.model_config
    expert_metadata = _hf_expert_metadata(vllm_config)
    payload = {
        "model_hash": model_config.compute_hash(),
        "model": model_config.model,
        "model_name": get_expert_hotset_model_name(vllm_config),
        "revision": model_config.revision,
        **expert_metadata,
    }
    payload["model_fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def get_expert_hotset_profile_path(vllm_config: VllmConfig) -> Path:
    model_name = get_expert_hotset_model_name(vllm_config)
    return EXPERT_HOTSET_PROFILE_DIR / f"{model_name}.json"


def expert_hotset_collect_mode_enabled(vllm_config: VllmConfig) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return False
    return bool(additional_config.get(EXPERT_HOTSET_COLLECT_MODE_KEY, False))


def expert_hotset_use_configured_cache_enabled(vllm_config: VllmConfig) -> bool:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return False
    return bool(additional_config.get(EXPERT_HOTSET_USE_CONFIGURED_CACHE_KEY, False))


def requires_expert_hotset_profile(vllm_config: VllmConfig) -> bool:
    expert_offload_config = vllm_config.expert_offload_config
    if not getattr(expert_offload_config, "offload_expert", False):
        return False
    metadata = _hf_expert_metadata(vllm_config)
    if metadata["num_hidden_layers"] <= metadata["first_k_dense_replace"] + 1:
        return False
    return not expert_hotset_collect_mode_enabled(vllm_config)


def load_expert_hotset_profile(vllm_config: VllmConfig) -> ExpertHotsetProfile | None:
    if not requires_expert_hotset_profile(vllm_config):
        return None

    profile_path = get_expert_hotset_profile_path(vllm_config)
    if not profile_path.exists():
        raise ValueError(
            "Missing expert hotset profile at "
            f"{profile_path}. Run `python simple_build_expert_hotset_profile.py` "
            "to generate it before starting the server."
        )

    try:
        with profile_path.open(encoding="utf-8") as f:
            profile = ExpertHotsetProfile.from_dict(json.load(f))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid expert hotset profile at {profile_path}: {exc}"
        ) from exc
    profile.validate_against(vllm_config)
    if not profile.converged:
        logger.warning(
            "Loading non-converged expert hotset profile from %s "
            "(sampled_requests=%d).",
            profile_path,
            profile.sampled_requests,
        )
    return profile


def build_expert_hotset_profile(
    vllm_config: VllmConfig,
    layer_counts: dict[int, list[int]],
    sampled_requests: int,
    converged: bool,
    convergence: dict[str, Any],
) -> ExpertHotsetProfile:
    metadata = get_expert_hotset_metadata(vllm_config)
    num_experts = metadata["num_experts"]
    layers: dict[int, ExpertHotsetLayerProfile] = {}
    for layer_id in range(
        metadata["first_k_dense_replace"], metadata["num_hidden_layers"]
    ):
        counts = [int(x) for x in layer_counts[layer_id]]
        if len(counts) != num_experts:
            raise ValueError(
                "Layer count length mismatch while building expert hotset profile "
                f"for layer {layer_id}: expected {num_experts}, found {len(counts)}."
            )
        total = sum(counts)
        if total <= 0:
            raise ValueError(
                "Cannot build expert hotset profile with zero observations for "
                f"layer {layer_id}."
            )
        ranked_expert_ids = tuple(
            sorted(range(num_experts), key=lambda expert_id: (-counts[expert_id], expert_id))
        )
        probabilities = tuple(count / total for count in counts)
        layers[layer_id] = ExpertHotsetLayerProfile(
            layer_id=layer_id,
            ranked_expert_ids=ranked_expert_ids,
            counts=tuple(counts),
            probabilities=probabilities,
        )
    profile = ExpertHotsetProfile(
        version=EXPERT_HOTSET_PROFILE_VERSION,
        model_fingerprint=metadata["model_fingerprint"],
        model=metadata["model"],
        revision=metadata["revision"],
        num_hidden_layers=metadata["num_hidden_layers"],
        num_experts=num_experts,
        first_k_dense_replace=metadata["first_k_dense_replace"],
        num_experts_per_tok=metadata["num_experts_per_tok"],
        sampled_requests=int(sampled_requests),
        converged=bool(converged),
        convergence=convergence,
        layers=layers,
    )
    profile.validate_against(vllm_config)
    return profile


def write_expert_hotset_profile(
    profile: ExpertHotsetProfile,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")

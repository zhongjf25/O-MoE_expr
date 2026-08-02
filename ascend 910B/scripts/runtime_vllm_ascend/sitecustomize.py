"""Optional Qwen3.5 compatibility shims for the baseline Ascend runtime."""

import os


if os.environ.get("BASELINE_QWEN35_TUPLE_SHARD_COMPAT") == "1":
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    _original_merged_weight_loader = MergedColumnParallelLinear.weight_loader

    def _tuple_compatible_merged_weight_loader(
        self,
        param,
        loaded_weight,
        loaded_shard_id=None,
    ):
        if not isinstance(loaded_shard_id, tuple):
            return _original_merged_weight_loader(
                self,
                param,
                loaded_weight,
                loaded_shard_id,
            )

        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            raise ValueError("Tuple shard loading requires an output_dim")

        loaded_offset = 0
        for shard_id in loaded_shard_id:
            shard_size = self.output_sizes[shard_id]
            loaded_weight_shard = loaded_weight.narrow(
                output_dim,
                loaded_offset,
                shard_size,
            )
            _original_merged_weight_loader(
                self,
                param,
                loaded_weight_shard,
                shard_id,
            )
            loaded_offset += shard_size

        if loaded_offset != loaded_weight.shape[output_dim]:
            raise ValueError(
                "Tuple shard sizes do not match the loaded weight: "
                f"{loaded_offset} != {loaded_weight.shape[output_dim]}"
            )

    MergedColumnParallelLinear.weight_loader = (
        _tuple_compatible_merged_weight_loader
    )


if os.environ.get("BASELINE_QWEN35_ASCEND_BF16_SSM") == "1":
    from vllm.model_executor.models.config import (
        Qwen3_5ForConditionalGenerationConfig,
    )

    if not getattr(
        Qwen3_5ForConditionalGenerationConfig,
        "_vllm_ascend_bf16_ssm_patch",
        False,
    ):
        _original_qwen35_verify = (
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config
        )

        def _ascend_compatible_qwen35_verify(vllm_config):
            cache_config = vllm_config.cache_config
            if cache_config.mamba_ssm_cache_dtype == "auto":
                # CANN 8.5.2 RecurrentGatedDeltaRule requires BF16 state.
                return
            _original_qwen35_verify(vllm_config)

        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config = (
            staticmethod(_ascend_compatible_qwen35_verify)
        )
        Qwen3_5ForConditionalGenerationConfig._vllm_ascend_bf16_ssm_patch = (
            True
        )

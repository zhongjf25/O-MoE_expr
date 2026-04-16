#include <torch/all.h>
#include <torch/library.h>

void copy_uncached_experts_to_comp(torch::Tensor& w13_dst,
                                   torch::Tensor& w2_dst,
                                   const torch::Tensor& w13_src,
                                   const torch::Tensor& w2_src,
                                   const torch::Tensor& expert_ids);
void prepare_offloaded_compute_inputs(torch::Tensor& expert_source,
                                      torch::Tensor& cache_w1_block_ids,
                                      torch::Tensor& cache_w2_block_ids,
                                      torch::Tensor& cache_w3_block_ids,
                                      torch::Tensor& miss_expert_ids,
                                      torch::Tensor& miss_count,
                                      torch::Tensor& seen_buffer,
                                      const torch::Tensor& layer_table,
                                      const torch::Tensor& topk_ids,
                                      const torch::Tensor& ready_mask,
                                      int64_t current_epoch);
void prepare_and_copy_offloaded_compute_inputs(
    torch::Tensor& expert_source, torch::Tensor& cache_w1_block_ids,
    torch::Tensor& cache_w2_block_ids, torch::Tensor& cache_w3_block_ids,
    torch::Tensor& miss_expert_ids, torch::Tensor& miss_count,
    torch::Tensor& seen_buffer, const torch::Tensor& layer_table,
    const torch::Tensor& topk_ids, const torch::Tensor& ready_mask,
    torch::Tensor& w13_dst, torch::Tensor& w2_dst,
    const torch::Tensor& w13_src, const torch::Tensor& w2_src,
    int64_t current_epoch);
void prepare_and_copy_offloaded_compute_inputs_compact(
    torch::Tensor& expert_source, torch::Tensor& comp_expert_to_slot,
    torch::Tensor& cache_w1_block_ids, torch::Tensor& cache_w2_block_ids,
    torch::Tensor& cache_w3_block_ids, torch::Tensor& miss_expert_ids,
    torch::Tensor& miss_slot_ids, torch::Tensor& miss_count,
    torch::Tensor& seen_buffer, torch::Tensor& slot_live_flags,
    const torch::Tensor& layer_table, const torch::Tensor& topk_ids,
    const torch::Tensor& ready_mask, const torch::Tensor& pending_mask,
    torch::Tensor& slot_to_expert, int64_t assigned_slot_count,
    torch::Tensor& w13_dst, torch::Tensor& w2_dst,
    const torch::Tensor& w13_src, const torch::Tensor& w2_src,
    int64_t current_epoch);

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "copy_uncached_experts_to_comp(Tensor! w13_dst, Tensor! w2_dst, "
      "Tensor w13_src, Tensor w2_src, Tensor expert_ids) -> ()");
  m.impl("copy_uncached_experts_to_comp", c10::kCUDA,
         &copy_uncached_experts_to_comp);
  m.def(
      "prepare_offloaded_compute_inputs(Tensor! expert_source, "
      "Tensor! cache_w1_block_ids, Tensor! cache_w2_block_ids, "
      "Tensor! cache_w3_block_ids, Tensor! miss_expert_ids, "
      "Tensor! miss_count, Tensor! seen_buffer, Tensor layer_table, "
      "Tensor topk_ids, Tensor ready_mask, int current_epoch) -> ()");
  m.impl("prepare_offloaded_compute_inputs", c10::kCUDA,
         &prepare_offloaded_compute_inputs);
  m.def(
      "prepare_and_copy_offloaded_compute_inputs(Tensor! expert_source, "
      "Tensor! cache_w1_block_ids, Tensor! cache_w2_block_ids, "
      "Tensor! cache_w3_block_ids, Tensor! miss_expert_ids, "
      "Tensor! miss_count, Tensor! seen_buffer, Tensor layer_table, "
      "Tensor topk_ids, Tensor ready_mask, Tensor! w13_dst, Tensor! w2_dst, "
      "Tensor w13_src, Tensor w2_src, int current_epoch) -> ()");
  m.impl("prepare_and_copy_offloaded_compute_inputs", c10::kCUDA,
         &prepare_and_copy_offloaded_compute_inputs);
  m.def(
      "prepare_and_copy_offloaded_compute_inputs_compact("
      "Tensor! expert_source, Tensor! comp_expert_to_slot, "
      "Tensor! cache_w1_block_ids, Tensor! cache_w2_block_ids, "
      "Tensor! cache_w3_block_ids, Tensor! miss_expert_ids, "
      "Tensor! miss_slot_ids, Tensor! miss_count, Tensor! seen_buffer, "
      "Tensor! slot_live_flags, Tensor layer_table, Tensor topk_ids, "
      "Tensor ready_mask, Tensor pending_mask, Tensor! slot_to_expert, "
      "int assigned_slot_count, Tensor! w13_dst, Tensor! w2_dst, "
      "Tensor w13_src, Tensor w2_src, int current_epoch) -> ()");
  m.impl("prepare_and_copy_offloaded_compute_inputs_compact", c10::kCUDA,
         &prepare_and_copy_offloaded_compute_inputs_compact);
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <algorithm>
#include <assert.h>
#include <cstdint>

#include "cuda_utils.h"

namespace {

constexpr int kCopyThreads = 256;
constexpr int64_t kVecBytes = 16;
constexpr int64_t kVecsPerThread = 8;
constexpr int64_t kScalarBytesPerThread = 16;
constexpr int kPrepareThreads = 256;

__host__ __device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & (kVecBytes - 1)) == 0;
}

template <bool Vec16>
__global__ void copy_selected_expert_rows_kernel(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    const int64_t* __restrict__ src_row_ids,
    const int64_t* __restrict__ dst_row_ids,
    const int32_t* __restrict__ active_row_count,
    int64_t max_selected_rows,
    int64_t dst_stride_bytes,
    int64_t src_stride_bytes,
    int64_t bytes_per_row,
    int64_t blocks_per_row) {
  int64_t num_selected_rows = max_selected_rows;
  if (active_row_count != nullptr) {
    const int64_t active_rows =
        std::max<int64_t>(0, static_cast<int64_t>(*active_row_count));
    num_selected_rows = std::min(max_selected_rows, active_rows);
  }
  if (num_selected_rows == 0) {
    return;
  }

  const int64_t total_tiles = num_selected_rows * blocks_per_row;
  for (int64_t task_idx = static_cast<int64_t>(blockIdx.x); task_idx < total_tiles;
       task_idx += static_cast<int64_t>(gridDim.x)) {
    const int64_t row_pos = task_idx / blocks_per_row;
    const int64_t tile_idx = task_idx - row_pos * blocks_per_row;
    const int64_t src_row = src_row_ids[row_pos];
    const int64_t dst_row = dst_row_ids == nullptr ? src_row : dst_row_ids[row_pos];
    if (src_row < 0 || dst_row < 0) {
      continue;
    }

    uint8_t* dst_row_ptr = dst + dst_row * dst_stride_bytes;
    const uint8_t* src_row_ptr = src + src_row * src_stride_bytes;

    if constexpr (Vec16) {
      constexpr int64_t kTileVecs = kCopyThreads * kVecsPerThread;
      const int64_t num_vecs = bytes_per_row / kVecBytes;
      const int64_t vec_start = tile_idx * kTileVecs;
      if (vec_start >= num_vecs) {
        continue;
      }
      const int64_t vec_end = std::min(num_vecs, vec_start + kTileVecs);
      auto* dst_vec = reinterpret_cast<int4*>(dst_row_ptr);
      const auto* src_vec = reinterpret_cast<const int4*>(src_row_ptr);
      for (int64_t vec_idx = vec_start + threadIdx.x; vec_idx < vec_end;
           vec_idx += blockDim.x) {
        dst_vec[vec_idx] = src_vec[vec_idx];
      }
    } else {
      constexpr int64_t kTileBytes = kCopyThreads * kScalarBytesPerThread;
      const int64_t byte_start = tile_idx * kTileBytes;
      if (byte_start >= bytes_per_row) {
        continue;
      }
      const int64_t byte_end =
          std::min(bytes_per_row, byte_start + kTileBytes);
      for (int64_t byte_idx = byte_start + threadIdx.x; byte_idx < byte_end;
           byte_idx += blockDim.x) {
        dst_row_ptr[byte_idx] = src_row_ptr[byte_idx];
      }
    }
  }
}

template <typename expert_id_t>
__global__ void prepare_offloaded_compute_inputs_kernel(
    int32_t* __restrict__ expert_source,
    int32_t* __restrict__ cache_w1_block_ids,
    int32_t* __restrict__ cache_w2_block_ids,
    int32_t* __restrict__ cache_w3_block_ids,
    int64_t* __restrict__ miss_expert_ids,
    int32_t* __restrict__ miss_count,
    int32_t* __restrict__ seen_buffer,
    const int32_t* __restrict__ layer_table,
    const expert_id_t* __restrict__ topk_ids,
    const bool* __restrict__ ready_mask,
    int64_t num_topk_ids,
    int64_t num_experts,
    int32_t current_epoch) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_topk_ids) {
    return;
  }

  const int64_t expert_id_i64 = static_cast<int64_t>(topk_ids[idx]);
  if (expert_id_i64 < 0 || expert_id_i64 >= num_experts) {
    return;
  }
  const int32_t expert_id = static_cast<int32_t>(expert_id_i64);

  if (atomicExch(seen_buffer + expert_id, current_epoch) == current_epoch) {
    return;
  }

  expert_source[expert_id] = 0;
  if (ready_mask != nullptr && ready_mask[expert_id]) {
    return;
  }

  const int64_t row_offset = static_cast<int64_t>(expert_id) * 3;
  const int32_t w1_block_id = layer_table[row_offset];
  if (w1_block_id != -1) {
    expert_source[expert_id] = 1;
    cache_w1_block_ids[expert_id] = w1_block_id;
    cache_w2_block_ids[expert_id] = layer_table[row_offset + 1];
    cache_w3_block_ids[expert_id] = layer_table[row_offset + 2];
    return;
  }

  const int32_t miss_pos = atomicAdd(miss_count, 1);
  miss_expert_ids[miss_pos] = static_cast<int64_t>(expert_id);
}

template <typename expert_id_t>
__global__ void prepare_offloaded_compute_inputs_compact_kernel(
    int32_t* __restrict__ expert_source,
    const int32_t* __restrict__ comp_expert_to_slot,
    int32_t* __restrict__ cache_w1_block_ids,
    int32_t* __restrict__ cache_w2_block_ids,
    int32_t* __restrict__ cache_w3_block_ids,
    int64_t* __restrict__ miss_expert_ids,
    int32_t* __restrict__ miss_count,
    int32_t* __restrict__ seen_buffer,
    int32_t* __restrict__ slot_live_flags,
    const int32_t* __restrict__ layer_table,
    const expert_id_t* __restrict__ topk_ids,
    const bool* __restrict__ ready_mask,
    int64_t num_topk_ids,
    int64_t num_experts,
    int64_t comp_capacity,
    int32_t current_epoch) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_topk_ids) {
    return;
  }

  const int64_t expert_id_i64 = static_cast<int64_t>(topk_ids[idx]);
  if (expert_id_i64 < 0 || expert_id_i64 >= num_experts) {
    return;
  }
  const int32_t expert_id = static_cast<int32_t>(expert_id_i64);

  if (atomicExch(seen_buffer + expert_id, current_epoch) == current_epoch) {
    return;
  }

  expert_source[expert_id] = -1;
  if (ready_mask != nullptr && ready_mask[expert_id]) {
    const int32_t slot = comp_expert_to_slot[expert_id];
    assert(slot >= 0 && slot < comp_capacity);
    expert_source[expert_id] = 0;
    atomicExch(slot_live_flags + slot, 1);
    return;
  }

  const int64_t row_offset = static_cast<int64_t>(expert_id) * 3;
  const int32_t w1_block_id = layer_table[row_offset];
  if (w1_block_id != -1) {
    expert_source[expert_id] = 1;
    cache_w1_block_ids[expert_id] = w1_block_id;
    cache_w2_block_ids[expert_id] = layer_table[row_offset + 1];
    cache_w3_block_ids[expert_id] = layer_table[row_offset + 2];
    return;
  }

  const int32_t miss_pos = atomicAdd(miss_count, 1);
  miss_expert_ids[miss_pos] = static_cast<int64_t>(expert_id);
}

__global__ void assign_compact_slots_kernel(
    int32_t* __restrict__ expert_source,
    int32_t* __restrict__ comp_expert_to_slot,
    int32_t* __restrict__ slot_to_expert,
    const int64_t* __restrict__ miss_expert_ids,
    int64_t* __restrict__ miss_slot_ids,
    const int32_t* __restrict__ miss_count,
    const int32_t* __restrict__ slot_live_flags,
    int32_t assigned_slot_count,
    int32_t comp_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int32_t miss_total = *miss_count;
  if (miss_total < 0) {
    miss_total = 0;
  }

  int32_t assigned = 0;
  for (int32_t slot = 0; slot < comp_capacity && assigned < miss_total; ++slot) {
    if (slot_live_flags[slot] != 0) {
      continue;
    }

    const int64_t new_expert = miss_expert_ids[assigned];
    assert(new_expert >= 0);

    if (slot < assigned_slot_count) {
      const int32_t old_expert = slot_to_expert[slot];
      if (old_expert >= 0) {
        comp_expert_to_slot[old_expert] = -1;
      }
    }

    comp_expert_to_slot[new_expert] = slot;
    slot_to_expert[slot] = static_cast<int32_t>(new_expert);
    miss_slot_ids[assigned] = static_cast<int64_t>(slot);
    expert_source[new_expert] = 0;
    ++assigned;
  }

  assert(assigned == miss_total);
}

void check_rank3_copy_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank-3");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_copy_layout_compatibility(const torch::Tensor& dst,
                                     const torch::Tensor& src,
                                     const char* dst_name,
                                     const char* src_name,
                                     bool require_same_rows) {
  check_rank3_copy_tensor(dst, dst_name);
  check_rank3_copy_tensor(src, src_name);
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(), dst_name, " and ",
              src_name, " must have the same dtype");
  TORCH_CHECK(dst.device() == src.device(), dst_name, " and ", src_name,
              " must be on the same device");
  TORCH_CHECK(dst.size(1) == src.size(1) && dst.size(2) == src.size(2),
              dst_name, " and ", src_name,
              " must have the same per-expert shape");
  if (require_same_rows) {
    TORCH_CHECK(dst.size(0) == src.size(0), dst_name, " and ", src_name,
                " must have the same number of experts");
  }
}

void check_miss_count_tensor(const torch::Tensor& miss_count) {
  TORCH_CHECK(miss_count.is_cuda(), "miss_count must be on a GPU");
  TORCH_CHECK(miss_count.dim() == 1 && miss_count.size(0) == 1,
              "miss_count must be a rank-1 tensor with one element");
  TORCH_CHECK(miss_count.is_contiguous(), "miss_count must be contiguous");
  TORCH_CHECK(miss_count.scalar_type() == torch::kInt32,
              "miss_count must have dtype int32");
}

void check_row_id_tensor(const torch::Tensor& tensor,
                         const char* name,
                         torch::Device device) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name,
              " must have dtype int64");
  TORCH_CHECK(tensor.device() == device, name,
              " must be on the same device as the destination");
}

template <bool Vec16>
int get_copy_launch_blocks(int64_t max_selected_rows,
                           int64_t blocks_per_row) {
  const int64_t max_tiles = max_selected_rows * blocks_per_row;
  if (max_tiles <= 0) {
    return 0;
  }

  int blocks_per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, copy_selected_expert_rows_kernel<Vec16>, kCopyThreads, 0));
  const int device_blocks = std::max(
      1, at::cuda::getCurrentDeviceProperties()->multiProcessorCount *
             std::max(1, blocks_per_sm));
  return static_cast<int>(std::min<int64_t>(max_tiles, device_blocks));
}

void launch_copy_selected_expert_rows(torch::Tensor& dst,
                                      const torch::Tensor& src,
                                      const torch::Tensor& src_row_ids,
                                      const torch::Tensor* dst_row_ids,
                                      const torch::Tensor* active_row_count,
                                      cudaStream_t stream) {
  if (src_row_ids.numel() == 0) {
    return;
  }

  check_row_id_tensor(src_row_ids, "src_row_ids", dst.device());
  if (dst_row_ids != nullptr) {
    check_row_id_tensor(*dst_row_ids, "dst_row_ids", dst.device());
    TORCH_CHECK(dst_row_ids->numel() == src_row_ids.numel(),
                "dst_row_ids must have the same length as src_row_ids");
  }

  const int64_t num_selected_rows = src_row_ids.size(0);
  const int64_t dst_stride_bytes = dst.stride(0) * dst.element_size();
  const int64_t src_stride_bytes = src.stride(0) * src.element_size();
  const int64_t bytes_per_row =
      (dst.numel() / dst.size(0)) * dst.element_size();
  const int64_t src_bytes_per_row =
      (src.numel() / src.size(0)) * src.element_size();
  const int32_t* active_row_count_ptr = nullptr;

  if (active_row_count != nullptr) {
    check_miss_count_tensor(*active_row_count);
    TORCH_CHECK(active_row_count->device() == dst.device(),
                "active_row_count must be on the same device as dst");
    active_row_count_ptr = active_row_count->data_ptr<int32_t>();
  }

  TORCH_CHECK(dst_stride_bytes >= bytes_per_row,
              "dst stride must cover a row slice");
  TORCH_CHECK(src_stride_bytes >= src_bytes_per_row,
              "src stride must cover a row slice");
  TORCH_CHECK(bytes_per_row == src_bytes_per_row,
              "source and destination row sizes must match");

  const bool can_vec16 =
      bytes_per_row % kVecBytes == 0 &&
      dst_stride_bytes % kVecBytes == 0 &&
      src_stride_bytes % kVecBytes == 0 &&
      is_16byte_aligned(dst.data_ptr()) &&
      is_16byte_aligned(src.data_ptr());

  int64_t blocks_per_row;
  if (can_vec16) {
    const int64_t num_vecs = bytes_per_row / kVecBytes;
    constexpr int64_t kTileVecs = kCopyThreads * kVecsPerThread;
    blocks_per_row = cuda_utils::ceil_div(num_vecs, kTileVecs);
    const int launch_blocks =
        get_copy_launch_blocks<true>(num_selected_rows, blocks_per_row);
    if (launch_blocks == 0) {
      return;
    }
    copy_selected_expert_rows_kernel<true>
        <<<launch_blocks, kCopyThreads, 0, stream>>>(
            reinterpret_cast<uint8_t*>(dst.data_ptr()),
            reinterpret_cast<const uint8_t*>(src.data_ptr()),
            src_row_ids.data_ptr<int64_t>(),
            dst_row_ids == nullptr ? nullptr : dst_row_ids->data_ptr<int64_t>(),
            active_row_count_ptr,
            num_selected_rows,
            dst_stride_bytes,
            src_stride_bytes,
            bytes_per_row,
            blocks_per_row);
  } else {
    constexpr int64_t kTileBytes = kCopyThreads * kScalarBytesPerThread;
    blocks_per_row = cuda_utils::ceil_div(bytes_per_row, kTileBytes);
    const int launch_blocks =
        get_copy_launch_blocks<false>(num_selected_rows, blocks_per_row);
    if (launch_blocks == 0) {
      return;
    }
    copy_selected_expert_rows_kernel<false>
        <<<launch_blocks, kCopyThreads, 0, stream>>>(
            reinterpret_cast<uint8_t*>(dst.data_ptr()),
            reinterpret_cast<const uint8_t*>(src.data_ptr()),
            src_row_ids.data_ptr<int64_t>(),
            dst_row_ids == nullptr ? nullptr : dst_row_ids->data_ptr<int64_t>(),
            active_row_count_ptr,
            num_selected_rows,
            dst_stride_bytes,
            src_stride_bytes,
            bytes_per_row,
            blocks_per_row);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_metadata_tensor(const torch::Tensor& tensor,
                           const char* name,
                           torch::ScalarType expected_dtype,
                           int64_t expected_numel) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype, name,
              " has an unexpected dtype");
  TORCH_CHECK(tensor.size(0) == expected_numel, name,
              " has an unexpected size");
}

int64_t check_prepare_inputs(const torch::Tensor& expert_source,
                             const torch::Tensor& cache_w1_block_ids,
                             const torch::Tensor& cache_w2_block_ids,
                             const torch::Tensor& cache_w3_block_ids,
                             const torch::Tensor& seen_buffer,
                             const torch::Tensor& layer_table,
                             const torch::Tensor& topk_ids,
                             const torch::Tensor& ready_mask,
                             int64_t current_epoch) {
  TORCH_CHECK(current_epoch > 0 && current_epoch <= INT32_MAX,
              "current_epoch must fit in int32 and be positive");
  TORCH_CHECK(layer_table.is_cuda(), "layer_table must be on a GPU");
  TORCH_CHECK(layer_table.dim() == 2, "layer_table must be rank-2");
  TORCH_CHECK(layer_table.is_contiguous(), "layer_table must be contiguous");
  TORCH_CHECK(layer_table.scalar_type() == torch::kInt32,
              "layer_table must have dtype int32");
  TORCH_CHECK(layer_table.size(1) == 3,
              "layer_table must have shape [num_experts, 3]");

  const int64_t num_experts = layer_table.size(0);
  check_metadata_tensor(expert_source, "expert_source", torch::kInt32,
                        num_experts);
  check_metadata_tensor(cache_w1_block_ids, "cache_w1_block_ids",
                        torch::kInt32, num_experts);
  check_metadata_tensor(cache_w2_block_ids, "cache_w2_block_ids",
                        torch::kInt32, num_experts);
  check_metadata_tensor(cache_w3_block_ids, "cache_w3_block_ids",
                        torch::kInt32, num_experts);
  check_metadata_tensor(seen_buffer, "seen_buffer", torch::kInt32,
                        num_experts);

  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be on a GPU");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32
                  || topk_ids.scalar_type() == torch::kInt64,
              "topk_ids must have dtype int32 or int64");
  TORCH_CHECK(topk_ids.device() == expert_source.device(),
              "topk_ids must be on the same device as the metadata tensors");
  TORCH_CHECK(layer_table.device() == expert_source.device(),
              "layer_table must be on the same device as the metadata tensors");

  TORCH_CHECK(ready_mask.is_cuda(), "ready_mask must be on a GPU");
  TORCH_CHECK(ready_mask.dim() == 1, "ready_mask must be 1D");
  TORCH_CHECK(ready_mask.is_contiguous(), "ready_mask must be contiguous");
  TORCH_CHECK(ready_mask.scalar_type() == torch::kBool,
              "ready_mask must have dtype bool");
  TORCH_CHECK(ready_mask.device() == expert_source.device(),
              "ready_mask must be on the same device as the metadata tensors");
  TORCH_CHECK(ready_mask.numel() == 0 || ready_mask.size(0) == num_experts,
              "ready_mask must be empty or match num_experts");
  return num_experts;
}

}  // namespace

void copy_uncached_experts_to_comp(torch::Tensor& w13_dst,
                                   torch::Tensor& w2_dst,
                                   const torch::Tensor& w13_src,
                                   const torch::Tensor& w2_src,
                                   const torch::Tensor& expert_ids) {
  check_copy_layout_compatibility(w13_dst, w13_src, "w13_dst", "w13_src", true);
  check_copy_layout_compatibility(w2_dst, w2_src, "w2_dst", "w2_src", true);
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be on a GPU");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be 1D");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(expert_ids.device() == w13_dst.device(),
              "expert_ids must be on the same device as the destination");
  TORCH_CHECK(w13_dst.size(0) == w2_dst.size(0),
              "w13_dst and w2_dst must have the same number of experts");
  TORCH_CHECK(w13_src.size(0) == w2_src.size(0),
              "w13_src and w2_src must have the same number of experts");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(w13_dst));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_copy_selected_expert_rows(
      w13_dst, w13_src, expert_ids, nullptr, nullptr, stream);
  launch_copy_selected_expert_rows(
      w2_dst, w2_src, expert_ids, nullptr, nullptr, stream);
}

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
                                      int64_t current_epoch) {
  const int64_t num_experts = check_prepare_inputs(
      expert_source, cache_w1_block_ids, cache_w2_block_ids,
      cache_w3_block_ids, seen_buffer, layer_table, topk_ids, ready_mask,
      current_epoch);
  check_metadata_tensor(miss_expert_ids, "miss_expert_ids", torch::kInt64,
                        num_experts);
  check_miss_count_tensor(miss_count);

  miss_expert_ids.fill_(-1);
  miss_count.zero_();

  const int64_t num_topk_ids = topk_ids.numel();
  if (num_topk_ids == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_source));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool has_ready_mask = ready_mask.numel() > 0;
  const int blocks =
      cuda_utils::ceil_div(num_topk_ids, static_cast<int64_t>(kPrepareThreads));

  AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(),
                             "prepare_offloaded_compute_inputs", [&] {
    prepare_offloaded_compute_inputs_kernel<scalar_t>
        <<<blocks, kPrepareThreads, 0, stream>>>(
            expert_source.data_ptr<int32_t>(),
            cache_w1_block_ids.data_ptr<int32_t>(),
            cache_w2_block_ids.data_ptr<int32_t>(),
            cache_w3_block_ids.data_ptr<int32_t>(),
            miss_expert_ids.data_ptr<int64_t>(),
            miss_count.data_ptr<int32_t>(),
            seen_buffer.data_ptr<int32_t>(),
            layer_table.data_ptr<int32_t>(),
            topk_ids.data_ptr<scalar_t>(),
            has_ready_mask ? ready_mask.data_ptr<bool>() : nullptr,
            num_topk_ids,
            num_experts,
            static_cast<int32_t>(current_epoch));
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_and_copy_offloaded_compute_inputs(
    torch::Tensor& expert_source, torch::Tensor& cache_w1_block_ids,
    torch::Tensor& cache_w2_block_ids, torch::Tensor& cache_w3_block_ids,
    torch::Tensor& miss_expert_ids, torch::Tensor& miss_count,
    torch::Tensor& seen_buffer, const torch::Tensor& layer_table,
    const torch::Tensor& topk_ids, const torch::Tensor& ready_mask,
    torch::Tensor& w13_dst, torch::Tensor& w2_dst,
    const torch::Tensor& w13_src, const torch::Tensor& w2_src,
    int64_t current_epoch) {
  const int64_t num_experts = check_prepare_inputs(
      expert_source, cache_w1_block_ids, cache_w2_block_ids,
      cache_w3_block_ids, seen_buffer, layer_table, topk_ids, ready_mask,
      current_epoch);
  check_miss_count_tensor(miss_count);
  check_copy_layout_compatibility(w13_dst, w13_src, "w13_dst", "w13_src", true);
  check_copy_layout_compatibility(w2_dst, w2_src, "w2_dst", "w2_src", true);
  TORCH_CHECK(w13_dst.size(0) == num_experts,
              "w13_dst must have the same number of experts as layer_table");
  TORCH_CHECK(w2_dst.size(0) == num_experts,
              "w2_dst must have the same number of experts as layer_table");

  TORCH_CHECK(miss_expert_ids.is_cuda(), "miss_expert_ids must be on a GPU");
  TORCH_CHECK(miss_expert_ids.dim() == 1, "miss_expert_ids must be 1D");
  TORCH_CHECK(miss_expert_ids.is_contiguous(),
              "miss_expert_ids must be contiguous");
  TORCH_CHECK(miss_expert_ids.scalar_type() == torch::kInt64,
              "miss_expert_ids must have dtype int64");
  TORCH_CHECK(miss_expert_ids.device() == expert_source.device(),
              "miss_expert_ids must be on the same device as metadata tensors");

  const int64_t num_topk_ids = topk_ids.numel();
  TORCH_CHECK(miss_expert_ids.size(0) >= num_topk_ids,
              "miss_expert_ids must provide at least topk_ids.numel() slots");

  auto miss_expert_ids_view = miss_expert_ids.narrow(0, 0, num_topk_ids);
  miss_expert_ids_view.fill_(-1);
  miss_count.zero_();
  if (num_topk_ids == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_source));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool has_ready_mask = ready_mask.numel() > 0;
  const int blocks =
      cuda_utils::ceil_div(num_topk_ids, static_cast<int64_t>(kPrepareThreads));

  AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(),
                             "prepare_and_copy_offloaded_compute_inputs", [&] {
    prepare_offloaded_compute_inputs_kernel<scalar_t>
        <<<blocks, kPrepareThreads, 0, stream>>>(
            expert_source.data_ptr<int32_t>(),
            cache_w1_block_ids.data_ptr<int32_t>(),
            cache_w2_block_ids.data_ptr<int32_t>(),
            cache_w3_block_ids.data_ptr<int32_t>(),
            miss_expert_ids_view.data_ptr<int64_t>(),
            miss_count.data_ptr<int32_t>(),
            seen_buffer.data_ptr<int32_t>(),
            layer_table.data_ptr<int32_t>(),
            topk_ids.data_ptr<scalar_t>(),
            has_ready_mask ? ready_mask.data_ptr<bool>() : nullptr,
            num_topk_ids,
            num_experts,
            static_cast<int32_t>(current_epoch));
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  launch_copy_selected_expert_rows(
      w13_dst, w13_src, miss_expert_ids_view, nullptr, &miss_count, stream);
  launch_copy_selected_expert_rows(
      w2_dst, w2_src, miss_expert_ids_view, nullptr, &miss_count, stream);
}

void prepare_and_copy_offloaded_compute_inputs_compact(
    torch::Tensor& expert_source,
    torch::Tensor& comp_expert_to_slot,
    torch::Tensor& cache_w1_block_ids,
    torch::Tensor& cache_w2_block_ids,
    torch::Tensor& cache_w3_block_ids,
    torch::Tensor& miss_expert_ids,
    torch::Tensor& miss_slot_ids,
    torch::Tensor& miss_count,
    torch::Tensor& seen_buffer,
    torch::Tensor& slot_live_flags,
    const torch::Tensor& layer_table,
    const torch::Tensor& topk_ids,
    const torch::Tensor& ready_mask,
    const torch::Tensor& pending_mask,
    torch::Tensor& slot_to_expert,
    int64_t assigned_slot_count,
    torch::Tensor& w13_dst,
    torch::Tensor& w2_dst,
    const torch::Tensor& w13_src,
    const torch::Tensor& w2_src,
    int64_t current_epoch) {
  const int64_t num_experts = check_prepare_inputs(
      expert_source, cache_w1_block_ids, cache_w2_block_ids,
      cache_w3_block_ids, seen_buffer, layer_table, topk_ids, ready_mask,
      current_epoch);
  check_metadata_tensor(comp_expert_to_slot, "comp_expert_to_slot",
                        torch::kInt32, num_experts);
  check_miss_count_tensor(miss_count);
  check_copy_layout_compatibility(w13_dst, w13_src, "w13_dst", "w13_src", false);
  check_copy_layout_compatibility(w2_dst, w2_src, "w2_dst", "w2_src", false);
  TORCH_CHECK(w13_dst.size(0) == w2_dst.size(0),
              "w13_dst and w2_dst must have the same compact capacity");
  TORCH_CHECK(w13_src.size(0) == num_experts,
              "w13_src must have one row per expert");
  TORCH_CHECK(w2_src.size(0) == num_experts,
              "w2_src must have one row per expert");

  const int64_t compact_capacity = w13_dst.size(0);
  TORCH_CHECK(compact_capacity > 0,
              "compact destination must expose at least one slot");
  TORCH_CHECK(assigned_slot_count >= 0 && assigned_slot_count <= compact_capacity,
              "assigned_slot_count must stay within compact capacity");
  check_metadata_tensor(slot_to_expert, "slot_to_expert", torch::kInt32,
                        compact_capacity);
  check_metadata_tensor(slot_live_flags, "slot_live_flags", torch::kInt32,
                        compact_capacity);

  TORCH_CHECK(miss_expert_ids.is_cuda(), "miss_expert_ids must be on a GPU");
  TORCH_CHECK(miss_expert_ids.dim() == 1, "miss_expert_ids must be 1D");
  TORCH_CHECK(miss_expert_ids.is_contiguous(),
              "miss_expert_ids must be contiguous");
  TORCH_CHECK(miss_expert_ids.scalar_type() == torch::kInt64,
              "miss_expert_ids must have dtype int64");
  TORCH_CHECK(miss_expert_ids.device() == expert_source.device(),
              "miss_expert_ids must be on the same device as metadata tensors");
  TORCH_CHECK(miss_slot_ids.is_cuda(), "miss_slot_ids must be on a GPU");
  TORCH_CHECK(miss_slot_ids.dim() == 1, "miss_slot_ids must be 1D");
  TORCH_CHECK(miss_slot_ids.is_contiguous(), "miss_slot_ids must be contiguous");
  TORCH_CHECK(miss_slot_ids.scalar_type() == torch::kInt64,
              "miss_slot_ids must have dtype int64");
  TORCH_CHECK(miss_slot_ids.device() == expert_source.device(),
              "miss_slot_ids must be on the same device as metadata tensors");

  TORCH_CHECK(pending_mask.is_cuda(), "pending_mask must be on a GPU");
  TORCH_CHECK(pending_mask.dim() == 1, "pending_mask must be 1D");
  TORCH_CHECK(pending_mask.is_contiguous(), "pending_mask must be contiguous");
  TORCH_CHECK(pending_mask.scalar_type() == torch::kBool,
              "pending_mask must have dtype bool");
  TORCH_CHECK(pending_mask.device() == expert_source.device(),
              "pending_mask must be on the same device as metadata tensors");
  TORCH_CHECK(pending_mask.numel() == 0 || pending_mask.size(0) == num_experts,
              "pending_mask must be empty or match num_experts");

  const int64_t num_topk_ids = topk_ids.numel();
  TORCH_CHECK(miss_expert_ids.size(0) >= num_topk_ids,
              "miss_expert_ids must provide at least topk_ids.numel() slots");
  TORCH_CHECK(miss_slot_ids.size(0) >= num_topk_ids,
              "miss_slot_ids must provide at least topk_ids.numel() slots");

  auto miss_expert_ids_view = miss_expert_ids.narrow(0, 0, num_topk_ids);
  auto miss_slot_ids_view = miss_slot_ids.narrow(0, 0, num_topk_ids);
  miss_expert_ids_view.fill_(-1);
  miss_slot_ids_view.fill_(-1);
  miss_count.zero_();
  slot_live_flags.zero_();
  (void)pending_mask;

  if (num_topk_ids == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_source));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool has_ready_mask = ready_mask.numel() > 0;
  const int blocks =
      cuda_utils::ceil_div(num_topk_ids, static_cast<int64_t>(kPrepareThreads));

  AT_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(),
      "prepare_and_copy_offloaded_compute_inputs_compact", [&] {
        prepare_offloaded_compute_inputs_compact_kernel<scalar_t>
            <<<blocks, kPrepareThreads, 0, stream>>>(
                expert_source.data_ptr<int32_t>(),
                comp_expert_to_slot.data_ptr<int32_t>(),
                cache_w1_block_ids.data_ptr<int32_t>(),
                cache_w2_block_ids.data_ptr<int32_t>(),
                cache_w3_block_ids.data_ptr<int32_t>(),
                miss_expert_ids_view.data_ptr<int64_t>(),
                miss_count.data_ptr<int32_t>(),
                seen_buffer.data_ptr<int32_t>(),
                slot_live_flags.data_ptr<int32_t>(),
                layer_table.data_ptr<int32_t>(),
                topk_ids.data_ptr<scalar_t>(),
                has_ready_mask ? ready_mask.data_ptr<bool>() : nullptr,
                num_topk_ids,
                num_experts,
                compact_capacity,
                static_cast<int32_t>(current_epoch));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  assign_compact_slots_kernel<<<1, 1, 0, stream>>>(
      expert_source.data_ptr<int32_t>(),
      comp_expert_to_slot.data_ptr<int32_t>(),
      slot_to_expert.data_ptr<int32_t>(),
      miss_expert_ids_view.data_ptr<int64_t>(),
      miss_slot_ids_view.data_ptr<int64_t>(),
      miss_count.data_ptr<int32_t>(),
      slot_live_flags.data_ptr<int32_t>(),
      static_cast<int32_t>(assigned_slot_count),
      static_cast<int32_t>(compact_capacity));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  launch_copy_selected_expert_rows(
      w13_dst, w13_src, miss_expert_ids_view, &miss_slot_ids_view, &miss_count,
      stream);
  launch_copy_selected_expert_rows(
      w2_dst, w2_src, miss_expert_ids_view, &miss_slot_ids_view, &miss_count,
      stream);
}

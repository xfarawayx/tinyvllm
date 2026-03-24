// ---------------------------------------------------------------------------
// nf4_dequant.cu
//
// NF4 double-quantized dequantization CUDA kernel for tinyvllm.
//
// Adapted from Learning-CUDA/03_nf4_dequant/xfarawayx/kernel/nf4_dequant_kernel.cuh
//
// Thread mapping: 1 thread -> 4 packed bytes -> 8 output elements
// absmax_real = code2[absmax_q[block_idx]] * absmax2[group_idx] + offset
// output[i]   = NF4_TABLE[index] * absmax_real
// ---------------------------------------------------------------------------

#include "tvllm/fused_kernels.h"
#include "tvllm/cuda_common.h"

namespace tvllm {
namespace cuda {

// NF4 lookup table (bitsandbytes create_normal_map)
__constant__ float TVLLM_NF4_TABLE[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

// float -> half / __nv_bfloat16 template cast
template <typename OutT>
__device__ __forceinline__ OutT nf4_cast_from_float(float x);

template <>
__device__ __forceinline__ half nf4_cast_from_float<half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 nf4_cast_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// Raw 16-bit representation
template <typename OutT>
__device__ __forceinline__ uint16_t nf4_raw_bits(OutT v) {
    return *reinterpret_cast<uint16_t*>(&v);
}

// log2(x), requires x to be a power of 2
static inline int log2_pow2(int x) {
    int r = 0;
    while (x > 1) { x >>= 1; r++; }
    return r;
}

template <typename OutT>
__global__ void nf4_dequantize_kernel(
    const uint8_t*  __restrict__ packed_weights,  // [n/2]
    const uint8_t*  __restrict__ absmax_q,        // [num_blocks]
    const half*     __restrict__ absmax2,          // [num_groups]
    const half*     __restrict__ code2,            // [256]
    float           offset,
    int             log2_blocksize,
    int             log2_s2_blocksize,
    int64_t         n_elements,
    OutT*           __restrict__ output)
{
    // Load NF4 table into shared memory to avoid constant memory warp serialization
    __shared__ float s_nf4_table[16];
    if (threadIdx.x < 16) {
        s_nf4_table[threadIdx.x] = TVLLM_NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    // Each thread processes 4 packed bytes = 8 output elements
    int tid_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int n_packed = (int)((n_elements + 1) / 2);

    if (tid_vec >= (n_packed + 3) / 4) return;

    // Vectorized read of 4 bytes, fallback for tail
    int byte_offset = tid_vec * 4;
    uint32_t packed4;
    if (byte_offset + 4 <= n_packed) {
        packed4 = reinterpret_cast<const uint32_t*>(packed_weights)[tid_vec];
    } else {
        packed4 = 0;
        for (int b = 0; b < 4 && byte_offset + b < n_packed; b++) {
            packed4 |= ((uint32_t)packed_weights[byte_offset + b]) << (b << 3);
        }
    }

    int elem_base = tid_vec * 8;

    uint32_t out_packed[4];

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int elem0 = elem_base + b * 2;
        int elem1 = elem0 + 1;

        // Unpack high/low 4-bit indices, lookup NF4 table
        uint8_t packed_byte = (packed4 >> (b * 8)) & 0xFF;
        uint8_t idx_hi = (packed_byte >> 4) & 0x0F;
        uint8_t idx_lo = packed_byte & 0x0F;

        float val_hi = s_nf4_table[idx_hi];
        float val_lo = s_nf4_table[idx_lo];

        // Double quantization decode
        int block_idx0 = elem0 >> log2_blocksize;
        int group_idx0 = block_idx0 >> log2_s2_blocksize;

        uint8_t aq0 = absmax_q[block_idx0];
        float absmax_real0 = __half2float(code2[aq0])
                           * __half2float(absmax2[group_idx0])
                           + offset;

        OutT out0, out1;

        if (elem0 < n_elements) {
            float dq0 = val_hi * absmax_real0;
            out0 = nf4_cast_from_float<OutT>(dq0);
        } else {
            out0 = nf4_cast_from_float<OutT>(0.0f);
        }

        if (elem1 < n_elements) {
            // Adjacent elements likely in same block; recompute only on boundary
            int block_idx1 = elem1 >> log2_blocksize;
            float absmax_real1;
            if (block_idx1 == block_idx0) {
                absmax_real1 = absmax_real0;
            } else {
                uint8_t aq1 = absmax_q[block_idx1];
                int group_idx1 = block_idx1 >> log2_s2_blocksize;
                absmax_real1 = __half2float(code2[aq1])
                             * __half2float(absmax2[group_idx1])
                             + offset;
            }
            float dq1 = val_lo * absmax_real1;
            out1 = nf4_cast_from_float<OutT>(dq1);
        } else {
            out1 = nf4_cast_from_float<OutT>(0.0f);
        }

        // Pack two fp16/bf16 values into one uint32_t
        uint16_t bits0 = nf4_raw_bits(out0);
        uint16_t bits1 = nf4_raw_bits(out1);
        out_packed[b] = (uint32_t)bits0 | ((uint32_t)bits1 << 16);
    }

    // Vectorized write: full 4-pack uses uint4 (128-bit), tail writes individually
    int out_base = tid_vec * 4;
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);

    int valid_packs = 0;
    for (int b = 0; b < 4; b++) {
        if (byte_offset + b < n_packed) valid_packs++;
    }

    if (valid_packs == 4) {
        reinterpret_cast<uint4*>(out_u32)[tid_vec] =
            make_uint4(out_packed[0], out_packed[1], out_packed[2], out_packed[3]);
    } else {
        for (int b = 0; b < valid_packs; b++) {
            out_u32[out_base + b] = out_packed[b];
        }
    }
}

// ============================================================================
//  Host entry point
// ============================================================================

torch::Tensor nf4_dequantize(
    const torch::Tensor& packed_weights,
    const torch::Tensor& absmax_q,
    const torch::Tensor& absmax2,
    const torch::Tensor& code2,
    float offset,
    int blocksize,
    int s2_blocksize,
    int64_t num_rows,
    int64_t num_cols) {

  TORCH_CHECK(packed_weights.is_cuda(), "nf4_dequantize: packed_weights must be on CUDA");
  TORCH_CHECK(packed_weights.dtype() == torch::kByte, "nf4_dequantize: packed_weights must be uint8");
  TORCH_CHECK(absmax_q.dtype() == torch::kByte, "nf4_dequantize: absmax_q must be uint8");
  TORCH_CHECK(absmax2.dtype() == torch::kHalf, "nf4_dequantize: absmax2 must be fp16");
  TORCH_CHECK(code2.dtype() == torch::kHalf, "nf4_dequantize: code2 must be fp16");

  int64_t n_elements = num_rows * num_cols;
  auto output = torch::empty({num_rows, num_cols}, absmax2.options());

  int n_packed = (int)((n_elements + 1) / 2);
  int n_packed_vec = (n_packed + 3) / 4;
  constexpr int threads_per_block = 256;
  int num_blocks_kernel = (n_packed_vec + threads_per_block - 1) / threads_per_block;
  int log2_bs = log2_pow2(blocksize);
  int log2_s2 = log2_pow2(s2_blocksize);

  // absmax2 is always stored as fp16, output dtype follows model dtype
  // For bf16 models, we need to handle the output dtype separately
  auto out_dtype = output.scalar_type();

  if (out_dtype == at::ScalarType::Half) {
    nf4_dequantize_kernel<half><<<num_blocks_kernel, threads_per_block>>>(
        packed_weights.data_ptr<uint8_t>(),
        absmax_q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(absmax2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(code2.data_ptr<at::Half>()),
        offset,
        log2_bs,
        log2_s2,
        n_elements,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else if (out_dtype == at::ScalarType::BFloat16) {
    // For bf16: output is bf16, but absmax2/code2 are still fp16
    auto output_bf16 = torch::empty({num_rows, num_cols},
        packed_weights.options().dtype(torch::kBFloat16));
    nf4_dequantize_kernel<__nv_bfloat16><<<num_blocks_kernel, threads_per_block>>>(
        packed_weights.data_ptr<uint8_t>(),
        absmax_q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(absmax2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(code2.data_ptr<at::Half>()),
        offset,
        log2_bs,
        log2_s2,
        n_elements,
        reinterpret_cast<__nv_bfloat16*>(output_bf16.data_ptr<at::BFloat16>()));
    return output_bf16;
  } else {
    TORCH_CHECK(false, "nf4_dequantize: unsupported output dtype");
  }

  return output;
}

// ============================================================================
//  Tile dequantize kernel — dequantizes a row range of the weight matrix
//  into a pre-allocated buffer, using global element offsets for correct
//  absmax indexing.
// ============================================================================

template <typename OutT>
__global__ void nf4_dequantize_tile_kernel(
    const uint8_t*  __restrict__ packed_weights,  // pointer ALREADY offset to tile start
    const uint8_t*  __restrict__ absmax_q,        // full array (global indexing)
    const half*     __restrict__ absmax2,
    const half*     __restrict__ code2,
    float           offset,
    int             log2_blocksize,
    int             log2_s2_blocksize,
    int64_t         n_tile_elements,              // tile_rows * num_cols
    int64_t         global_elem_offset,           // row_start * num_cols
    OutT*           __restrict__ output)
{
    __shared__ float s_nf4_table[16];
    if (threadIdx.x < 16) {
        s_nf4_table[threadIdx.x] = TVLLM_NF4_TABLE[threadIdx.x];
    }
    __syncthreads();

    int tid_vec = blockIdx.x * blockDim.x + threadIdx.x;
    int n_packed = (int)((n_tile_elements + 1) / 2);

    if (tid_vec >= (n_packed + 3) / 4) return;

    int byte_offset = tid_vec * 4;
    uint32_t packed4;
    if (byte_offset + 4 <= n_packed) {
        packed4 = reinterpret_cast<const uint32_t*>(packed_weights)[tid_vec];
    } else {
        packed4 = 0;
        for (int b = 0; b < 4 && byte_offset + b < n_packed; b++) {
            packed4 |= ((uint32_t)packed_weights[byte_offset + b]) << (b << 3);
        }
    }

    int elem_base = tid_vec * 8;  // local element index in tile

    uint32_t out_packed[4];

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        int local0 = elem_base + b * 2;
        int local1 = local0 + 1;
        int64_t global0 = global_elem_offset + local0;

        uint8_t packed_byte = (packed4 >> (b * 8)) & 0xFF;
        uint8_t idx_hi = (packed_byte >> 4) & 0x0F;
        uint8_t idx_lo = packed_byte & 0x0F;

        float val_hi = s_nf4_table[idx_hi];
        float val_lo = s_nf4_table[idx_lo];

        int block_idx0 = (int)(global0 >> log2_blocksize);
        int group_idx0 = block_idx0 >> log2_s2_blocksize;
        uint8_t aq0 = absmax_q[block_idx0];
        float absmax_real0 = __half2float(code2[aq0])
                           * __half2float(absmax2[group_idx0])
                           + offset;

        OutT out0, out1;

        if (local0 < n_tile_elements) {
            out0 = nf4_cast_from_float<OutT>(val_hi * absmax_real0);
        } else {
            out0 = nf4_cast_from_float<OutT>(0.0f);
        }

        if (local1 < n_tile_elements) {
            int64_t global1 = global_elem_offset + local1;
            int block_idx1 = (int)(global1 >> log2_blocksize);
            float absmax_real1;
            if (block_idx1 == block_idx0) {
                absmax_real1 = absmax_real0;
            } else {
                int group_idx1 = block_idx1 >> log2_s2_blocksize;
                absmax_real1 = __half2float(code2[absmax_q[block_idx1]])
                             * __half2float(absmax2[group_idx1])
                             + offset;
            }
            out1 = nf4_cast_from_float<OutT>(val_lo * absmax_real1);
        } else {
            out1 = nf4_cast_from_float<OutT>(0.0f);
        }

        out_packed[b] = (uint32_t)nf4_raw_bits(out0) | ((uint32_t)nf4_raw_bits(out1) << 16);
    }

    int out_base = tid_vec * 4;
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);

    int valid_packs = 0;
    for (int b = 0; b < 4; b++) {
        if (byte_offset + b < n_packed) valid_packs++;
    }

    if (valid_packs == 4) {
        reinterpret_cast<uint4*>(out_u32)[tid_vec] =
            make_uint4(out_packed[0], out_packed[1], out_packed[2], out_packed[3]);
    } else {
        for (int b = 0; b < valid_packs; b++) {
            out_u32[out_base + b] = out_packed[b];
        }
    }
}

// ============================================================================
//  Host: dequantize a row tile into a pre-allocated buffer
// ============================================================================

static void nf4_dequantize_tile_launch(
    const torch::Tensor& packed_weights,
    const torch::Tensor& absmax_q,
    const torch::Tensor& absmax2,
    const torch::Tensor& code2,
    float offset,
    int blocksize,
    int s2_blocksize,
    int64_t num_cols,
    int64_t row_start,
    int64_t tile_rows,
    torch::Tensor& output)    // pre-allocated [tile_rows, num_cols]
{
  int64_t n_tile_elements = tile_rows * num_cols;
  int64_t global_offset = row_start * num_cols;
  int64_t packed_byte_offset = global_offset / 2;

  int n_packed = (int)((n_tile_elements + 1) / 2);
  int n_packed_vec = (n_packed + 3) / 4;
  constexpr int threads_per_block = 256;
  int num_blocks_kernel = (n_packed_vec + threads_per_block - 1) / threads_per_block;
  int log2_bs = log2_pow2(blocksize);
  int log2_s2 = log2_pow2(s2_blocksize);

  const uint8_t* pw_ptr = packed_weights.data_ptr<uint8_t>() + packed_byte_offset;

  auto out_dtype = output.scalar_type();
  if (out_dtype == at::ScalarType::Half) {
    nf4_dequantize_tile_kernel<half><<<num_blocks_kernel, threads_per_block>>>(
        pw_ptr,
        absmax_q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(absmax2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(code2.data_ptr<at::Half>()),
        offset, log2_bs, log2_s2,
        n_tile_elements, global_offset,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else if (out_dtype == at::ScalarType::BFloat16) {
    nf4_dequantize_tile_kernel<__nv_bfloat16><<<num_blocks_kernel, threads_per_block>>>(
        pw_ptr,
        absmax_q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(absmax2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(code2.data_ptr<at::Half>()),
        offset, log2_bs, log2_s2,
        n_tile_elements, global_offset,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
  } else {
    TORCH_CHECK(false, "nf4_dequantize_tile: unsupported output dtype");
  }
}

// ============================================================================
//  Host: fused tiled NF4 linear  (dequant tile → cuBLAS GEMM, repeat)
// ============================================================================

torch::Tensor nf4_linear_tiled(
    const torch::Tensor& input,
    const torch::Tensor& packed_weights,
    const torch::Tensor& absmax_q,
    const torch::Tensor& absmax2,
    const torch::Tensor& code2,
    float offset,
    int blocksize,
    int s2_blocksize,
    int64_t num_rows,     // N (out_features)
    int64_t num_cols)     // K (in_features)
{
  // Check env var for fallback to original full-dequantize path.
  static const bool use_tiled = []() {
    const char* env = std::getenv("TVLLM_NF4_TILED");
    return env == nullptr || std::string(env) != "0";
  }();

  if (!use_tiled) {
    // Fallback: full dequantize then matmul
    auto w = nf4_dequantize(packed_weights, absmax_q, absmax2, code2,
                            offset, blocksize, s2_blocksize, num_rows, num_cols);
    if (w.scalar_type() != input.scalar_type()) {
      w = w.to(input.scalar_type());
    }
    return torch::matmul(input, w.t());
  }

  constexpr int64_t TILE_N = 128;

  int64_t K = num_cols;
  int64_t N = num_rows;

  // Flatten input to 2D: [M, K]
  auto x_2d = input.reshape({-1, K});
  int64_t M = x_2d.size(0);

  // Allocate output [M, N] and tile dequant buffer [TILE_N, K]
  auto y = torch::empty({M, N}, x_2d.options());
  auto tile_buf = torch::empty({TILE_N, K}, x_2d.options());

  for (int64_t row = 0; row < N; row += TILE_N) {
    int64_t tile_rows = std::min(TILE_N, N - row);

    // Get a view of the buffer sized to the current tile
    auto buf = tile_buf.narrow(0, 0, tile_rows);   // [tile_rows, K]

    // Dequantize weight rows [row, row+tile_rows) into buf
    nf4_dequantize_tile_launch(packed_weights, absmax_q, absmax2, code2,
                               offset, blocksize, s2_blocksize,
                               K, row, tile_rows, buf);

    // GEMM: y[:, row:row+tile_rows] = x_2d @ buf^T
    //   x_2d      : [M, K]          contiguous
    //   buf.t()   : [K, tile_rows]   transposed view
    //   out_slice  : [M, tile_rows]  strides (N, 1) — cuBLAS handles via ldc
    auto out_slice = y.narrow(1, row, tile_rows);
    torch::mm_out(out_slice, x_2d, buf.t());
  }

  // Reshape back to match input batch dims: replace last dim K → N
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  return y.reshape(out_sizes);
}

}  // namespace cuda
}  // namespace tvllm

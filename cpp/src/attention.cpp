// ---------------------------------------------------------------------------
// attention.cpp
//
// Prefill: FlashInfer BatchPrefillWithRaggedKVCacheWrapper (supports any GQA)
// Decode:  FlashInfer BatchDecodeWithPagedKVCacheWrapper (GQA padded to pow2)
// ---------------------------------------------------------------------------

#include "tvllm/attention.h"

#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace tvllm {
namespace cuda {

// ============================================================================
//  Helpers
// ============================================================================

namespace {

// Default softmax scale: 1/sqrt(head_dim).
inline float default_scale(int64_t head_dim, float softmax_scale) {
    return (softmax_scale > 0.0f)
        ? softmax_scale
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// Ensure tensor is contiguous int32 on CPU.
inline torch::Tensor to_cpu_int32(const torch::Tensor& t) {
    auto out = t.is_cpu() ? t : t.to(torch::kCPU);
    if (out.dtype() != torch::kInt32) out = out.to(torch::kInt32);
    if (!out.is_contiguous()) out = out.contiguous();
    return out;
}

// Round up to the next power of 2 (for values >= 1).
inline int64_t next_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p *= 2;
    return p;
}

// Convert block_tables [batch, max_blocks] + context_lens [batch] (both CPU
// int32) into FlashInfer's paged format: indptr, indices, last_page_len.
// Returned tensors are on `device`.
struct PagedMeta {
    torch::Tensor indptr;        // [batch_size + 1]
    torch::Tensor indices;       // [total_pages]
    torch::Tensor last_page_len; // [batch_size]
};

PagedMeta build_paged_meta(
    const torch::Tensor& block_tables_cpu,
    const torch::Tensor& context_lens_cpu,
    int32_t block_size,
    torch::Device device)
{
    int64_t batch_size = context_lens_cpu.size(0);
    int64_t max_blocks = block_tables_cpu.size(1);
    auto ctx_ptr = context_lens_cpu.data_ptr<int32_t>();
    auto bt_ptr  = block_tables_cpu.data_ptr<int32_t>();

    std::vector<int32_t> indptr_vec(batch_size + 1);
    std::vector<int32_t> last_page_vec(batch_size);
    std::vector<int32_t> indices_vec;
    indices_vec.reserve(batch_size * max_blocks);

    indptr_vec[0] = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        int32_t ctx    = ctx_ptr[i];
        int32_t npages = (ctx + block_size - 1) / block_size;
        indptr_vec[i + 1] = indptr_vec[i] + npages;
        int32_t rem = ctx % block_size;
        last_page_vec[i] = (rem == 0) ? block_size : rem;
        for (int32_t j = 0; j < npages; ++j)
            indices_vec.push_back(bt_ptr[i * max_blocks + j]);
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(device);
    return {
        torch::tensor(indptr_vec, opts),
        torch::tensor(indices_vec, opts),
        torch::tensor(last_page_vec, opts),
    };
}

// ============================================================================
//  Lazy-initialised FlashInfer state (all wrappers)
// ============================================================================

struct FlashInferState {
    // Ragged prefill
    py::object prefill_wrapper;
    torch::Tensor prefill_workspace;

    // Paged decode
    py::object decode_wrapper;
    torch::Tensor decode_workspace;

    // Paged prefill (prefix caching)
    py::object prefill_paged_wrapper;
    torch::Tensor prefill_paged_workspace;

    bool initialized = false;
};

// Heap-allocated singleton: destructor never runs, avoiding use-after-finalize
// crashes when static C++ destructors fire after the Python interpreter exits.
static FlashInferState& state() {
    static FlashInferState* s = new FlashInferState();
    return *s;
}

static void ensure_init() {
    auto& s = state();
    if (s.initialized) return;

    py::gil_scoped_acquire gil;
    auto fi = py::module_::import("flashinfer");
    auto ws_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    constexpr int64_t kWorkspaceBytes = 128LL * 1024 * 1024;

    // Ragged prefill
    s.prefill_workspace = torch::empty({kWorkspaceBytes}, ws_opts);
    s.prefill_wrapper =
        fi.attr("prefill").attr("BatchPrefillWithRaggedKVCacheWrapper")(
            py::cast(s.prefill_workspace));

    // Paged decode
    s.decode_workspace = torch::empty({kWorkspaceBytes}, ws_opts);
    s.decode_wrapper =
        fi.attr("decode").attr("BatchDecodeWithPagedKVCacheWrapper")(
            py::cast(s.decode_workspace),
            py::arg("kv_layout") = "NHD");

    // Paged prefill
    s.prefill_paged_workspace = torch::empty({kWorkspaceBytes}, ws_opts);
    s.prefill_paged_wrapper =
        fi.attr("prefill").attr("BatchPrefillWithPagedKVCacheWrapper")(
            py::cast(s.prefill_paged_workspace),
            py::arg("kv_layout") = "NHD");

    s.initialized = true;
}

// Cached GQA padding parameters (set once per decode plan, reused across runs).
struct DecodePadInfo {
    int64_t num_heads        = 0;
    int64_t num_kv_heads     = 0;
    int64_t head_dim         = 0;
    int64_t group_size       = 0;
    int64_t padded_group     = 0;
    int64_t padded_num_heads = 0;
    bool    needs_pad        = false;
};

static DecodePadInfo& decode_pad_info() {
    static DecodePadInfo info;
    return info;
}

}  // anonymous namespace

// ============================================================================
//  flash_attn_varlen  (prefill, ragged KV)
// ============================================================================

torch::Tensor flash_attn_varlen(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& cu_seqlens_k,
    int64_t /*max_seqlen_q*/,
    int64_t /*max_seqlen_k*/,
    float softmax_scale,
    bool causal)
{
    int64_t num_heads    = q.size(1);
    int64_t num_kv_heads = k.size(1);
    int64_t head_dim     = q.size(2);
    softmax_scale = default_scale(head_dim, softmax_scale);

    ensure_init();
    py::gil_scoped_acquire gil;
    auto& s = state();

    py::object py_dtype = py::cast(q).attr("dtype");
    s.prefill_wrapper.attr("plan")(
        cu_seqlens_q, cu_seqlens_k,
        static_cast<int>(num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        py::arg("causal")       = py::bool_(causal),
        py::arg("sm_scale")     = py::float_(softmax_scale),
        py::arg("q_data_type")  = py_dtype,
        py::arg("kv_data_type") = py_dtype);

    return s.prefill_wrapper.attr("run")(q, k, v).cast<torch::Tensor>();
}

// ============================================================================
//  flash_attn_decode  (paged GQA decode)
//
//  FlashInfer requires power-of-2 GQA group sizes.  When group_size is not
//  a power of 2 (e.g. 12/2 = 6), we pad Q heads within each KV group to
//  the next power of 2 and slice back after the kernel.
// ============================================================================

void flash_attn_decode_plan(
    const torch::Tensor& q_any,
    const torch::Tensor& k_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    float softmax_scale,
    int64_t block_size)
{
    int64_t num_heads    = q_any.size(2);
    int64_t head_dim     = q_any.size(3);
    int64_t num_kv_heads = k_cache.size(2);
    softmax_scale = default_scale(head_dim, softmax_scale);

    // Cache GQA padding info for run().
    auto& pad = decode_pad_info();
    pad.num_heads        = num_heads;
    pad.num_kv_heads     = num_kv_heads;
    pad.head_dim         = head_dim;
    pad.group_size       = num_heads / num_kv_heads;
    pad.padded_group     = next_pow2(pad.group_size);
    pad.padded_num_heads = pad.padded_group * num_kv_heads;
    pad.needs_pad        = (pad.padded_num_heads != num_heads);

    // Convert block_tables + context_lens → FlashInfer paged format.
    auto meta = build_paged_meta(
        to_cpu_int32(block_tables), to_cpu_int32(context_lens),
        static_cast<int32_t>(block_size), q_any.device());

    ensure_init();
    py::gil_scoped_acquire gil;
    auto& s = state();

    py::object py_dtype = py::cast(q_any).attr("dtype");
    s.decode_wrapper.attr("plan")(
        meta.indptr, meta.indices, meta.last_page_len,
        static_cast<int>(pad.padded_num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        static_cast<int>(block_size),
        py::arg("sm_scale")     = py::float_(softmax_scale),
        py::arg("q_data_type")  = py_dtype,
        py::arg("kv_data_type") = py_dtype);
}

torch::Tensor flash_attn_decode_run(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache)
{
    const auto& pad = decode_pad_info();
    int64_t batch_size = q.size(0);

    // q: [batch, 1, num_heads, head_dim] → [batch, num_heads, head_dim]
    auto q_fi = q.squeeze(1);

    if (pad.needs_pad) {
        auto q_grouped = q_fi.reshape(
            {batch_size, pad.num_kv_heads, pad.group_size, pad.head_dim});
        auto q_padded = torch::zeros(
            {batch_size, pad.num_kv_heads, pad.padded_group, pad.head_dim},
            q_fi.options());
        q_padded.slice(2, 0, pad.group_size).copy_(q_grouped);
        q_fi = q_padded.reshape(
            {batch_size, pad.padded_num_heads, pad.head_dim});
    }

    py::gil_scoped_acquire gil;
    auto& s = state();
    auto out = s.decode_wrapper.attr("run")(
        q_fi, py::make_tuple(k_cache, v_cache)
    ).cast<torch::Tensor>();

    if (pad.needs_pad) {
        out = out.reshape(
                {batch_size, pad.num_kv_heads, pad.padded_group, pad.head_dim})
                 .slice(2, 0, pad.group_size)
                 .reshape({batch_size, pad.num_heads, pad.head_dim})
                 .contiguous();
    }

    return out.unsqueeze(1);
}

torch::Tensor flash_attn_decode(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    float softmax_scale,
    int64_t block_size)
{
    flash_attn_decode_plan(q, k_cache, block_tables, context_lens,
                           softmax_scale, block_size);
    return flash_attn_decode_run(q, k_cache, v_cache);
}

// ============================================================================
//  flash_attn_prefill_paged  (prefill with paged KV, for prefix caching)
// ============================================================================

void flash_attn_prefill_paged_plan(
    const torch::Tensor& q_any,
    const torch::Tensor& k_cache,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    float softmax_scale,
    int64_t block_size,
    bool causal)
{
    int64_t num_heads    = q_any.size(1);
    int64_t head_dim     = q_any.size(2);
    int64_t num_kv_heads = k_cache.size(2);
    softmax_scale = default_scale(head_dim, softmax_scale);

    // Convert block_tables + context_lens → FlashInfer paged format.
    auto meta = build_paged_meta(
        to_cpu_int32(block_tables), to_cpu_int32(context_lens),
        static_cast<int32_t>(block_size), q_any.device());

    // Ensure cu_seqlens_q is on device.
    auto cu_q = cu_seqlens_q;
    if (cu_q.device() != q_any.device()) cu_q = cu_q.to(q_any.device());
    if (cu_q.dtype() != torch::kInt32) cu_q = cu_q.to(torch::kInt32);

    ensure_init();
    py::gil_scoped_acquire gil;
    auto& s = state();

    py::object py_dtype = py::cast(q_any).attr("dtype");
    s.prefill_paged_wrapper.attr("plan")(
        cu_q,
        meta.indptr, meta.indices, meta.last_page_len,
        static_cast<int>(num_heads),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim),
        static_cast<int>(block_size),
        py::arg("causal")       = py::bool_(causal),
        py::arg("sm_scale")     = py::float_(softmax_scale),
        py::arg("q_data_type")  = py_dtype,
        py::arg("kv_data_type") = py_dtype);
}

torch::Tensor flash_attn_prefill_paged_run(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache)
{
    py::gil_scoped_acquire gil;
    auto& s = state();
    return s.prefill_paged_wrapper.attr("run")(
        q, py::make_tuple(k_cache, v_cache)
    ).cast<torch::Tensor>();
}

}  // namespace cuda
}  // namespace tvllm

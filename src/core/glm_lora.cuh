// src/core/glm_lora.cuh - STRATA_GLM_LORA=<adapter.gguf>: a rank-1 LoRA adapter applied while the GLM path runs (off
// unless the variable is set; the pack is not changed).  Written for Heretic's abliteration adapters for GLM-5.3-Flash
// (heretic-gguf, e.g. MorinoNushi/GLM-5.3-Flash-Heretic-Abliterated-LoRA-V3-GGUF), llama.cpp's LoRA GGUF layout:
//
//   blk.N.attn_output.weight.lora_a     [n_in, 1]            lora_b [1, n_embd]            (F16 or F32)
//   blk.N.ffn_down_shexp.weight.lora_a  [n_ff_sh, 1]         lora_b [1, n_embd]
//   blk.N.ffn_down_exps.weight.lora_a   [n_ff, 1, n_expert]  lora_b [1, n_embd, n_expert]
//
// Each adds delta = s B (A . x) to its matrix's output, x that matrix's own input, s = adapter.lora.alpha / rank *
// STRATA_GLM_LORA_SCALE (default 1).  A routed expert's delta is weighted by its routing weight like the expert's
// output.  The strengths differ per component and per routed expert, so the delta is taken from each matrix's input:
//   attention output - the q8_1 input of the output projection (DSA: F->attn_q, KDA: F->gated_q);
//   shared expert    - its q8_1 h (F->sh_hq);
//   routed experts   - each GPU-computed plan entry's q8_1 h (F->hq, entry i = plan entry i, plan_ptr[i] != 0);
//                      the CPU lane's in float (lane_experts' ff rows) on the host, added to its answer.
// Applied on the one-token decode only; the batched prompt path is turned off while it is on (prompts run token by
// token).  Not on batched rows, a drafter's verify windows, EXL3 packs or a helper GPU (logged at start).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace glm_lora {

// llama.cpp's block_q8_1: d and the block's sum as half2, then 32 int8
struct Q8 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q8) == 36, "q8_1 block");

struct Layer {
    // offsets into the half table, -1: none
    long long att_a = -1, att_b = -1, sh_a = -1, sh_b = -1, ex_a = -1, ex_b = -1;
    int att_in = 0;
};

namespace {
__device__ __forceinline__ float q8_val(const Q8* x, int j) {
    return __low2float(x[j >> 5].ds) * (float) x[j >> 5].qs[j & 31];
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// out[r] += scale * (A . deq(xq)) * B[r]; each block recomputes the dot (n_in <= 16K) and writes its share of rows
__global__ void attn_kernel(const Q8* __restrict__ xq, int n_in, const __half* __restrict__ A,
                            const __half* __restrict__ B, float* __restrict__ out, int n_out, float scale,
                            const int* skip) {
    if (skip != nullptr && *skip) return;
    __shared__ float red[8];
    float acc = 0.0f;
    for (int j = threadIdx.x; j < n_in; j += 256) acc += __half2float(A[j]) * q8_val(xq, j);
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) a += red[w];
    a *= scale;
    const int per = (n_out + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * per, r1 = min(n_out, r0 + per);
    for (int r = r0 + threadIdx.x; r < r1; r += 256) out[r] += a * __half2float(B[r]);
}

// the FFN's deltas: warps 0..7 the routed plan entries computed on the device, warp 8 the shared expert; then
// out[r] += scale * (c_sh B_sh[r] + sum_i w_i c_i B_e(i)[r])
__global__ void moe_kernel(const unsigned long long* __restrict__ plan_ptr, const float* __restrict__ plan_w,
                           const int* __restrict__ plan_id, const int* skip, int k, int n_expert,
                           const Q8* __restrict__ hq, int n_ff, const __half* __restrict__ eA,
                           const __half* __restrict__ eB, const Q8* __restrict__ sh_hq, int n_sh,
                           const __half* __restrict__ sA, const __half* __restrict__ sB, float* __restrict__ out,
                           int n_out, float scale) {
    if (skip != nullptr && *skip) return;
    __shared__ float c[9];
    __shared__ int id[8];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (w < 8) {
        int e = -1;
        if (eA != nullptr && w < k && plan_ptr[w] != 0ull) {
            e = plan_id[w];
            if (e < 0 || e >= n_expert) e = -1;
        }
        float acc = 0.0f;
        if (e >= 0) {
            const __half* a = eA + (size_t) e * n_ff;
            const Q8* x = hq + (size_t) w * (n_ff / 32);
            for (int j = lane; j < n_ff; j += 32) acc += __half2float(a[j]) * q8_val(x, j);
            acc = warp_sum(acc) * plan_w[w];
        }
        if (lane == 0) {
            c[w] = acc * scale;
            id[w] = e;
        }
    } else {
        float acc = 0.0f;
        if (sA != nullptr)
            for (int j = lane; j < n_sh; j += 32) acc += __half2float(sA[j]) * q8_val(sh_hq, j);
        acc = warp_sum(acc);
        if (lane == 0) c[8] = acc * scale;
    }
    __syncthreads();
    const int per = (n_out + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * per, r1 = min(n_out, r0 + per);
    for (int r = r0 + threadIdx.x; r < r1; r += blockDim.x) {
        float m = sB != nullptr ? c[8] * __half2float(sB[r]) : 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (id[i] >= 0) m += c[i] * __half2float(eB[(size_t) id[i] * n_out + r]);
        out[r] += m;
    }
}
}  // namespace

struct State {
    bool on = false;
    float scale = 1.0f;
    std::string path;
    std::vector<__half> host;      // every table, F16
    std::vector<float> ex_a_f, ex_b_f;   // the routed tables in F32 for the CPU lane (layer offsets in ex_f_off)
    std::vector<long long> ex_f_a_off, ex_f_b_off;
    std::vector<Layer> L;
    int n_embd = 0, n_ff = 0, n_sh = 0, n_expert = 0;
    int n_att = 0, n_shl = 0, n_exl = 0;
    std::mutex mu;
    __half* dev[64] = {};
};

inline State& state() {
    static State s;
    return s;
}

inline bool requested() {
    const char* p = getenv("STRATA_GLM_LORA");
    return p != nullptr && p[0] != '\0';
}

inline bool on() { return state().on; }

/// Reads STRATA_GLM_LORA and checks it against the model: att_in(il) is layer il's attention-output input width (0:
/// no attention output to adapt).  Logs and stays off on any mismatch.  Call once, before anything is applied.
inline void setup(int n_layers, const std::function<int(int)>& att_in, int n_embd, int n_ff, int n_sh, int n_expert) {
    State& s = state();
    static std::once_flag once;
    std::call_once(once, [&] {
        if (!requested()) return;
        s.path = getenv("STRATA_GLM_LORA");
        const auto refuse = [&](const std::string& why) {
            std::fprintf(stderr, "glm lora: %s: %s - running the stock model\n", s.path.c_str(), why.c_str());
            s.host.clear();
            s.ex_a_f.clear();
            s.ex_b_f.clear();
            s.L.clear();
        };
        try {
            strata::GgufFile f(s.path);
            const auto* at = f.get("adapter.type");
            if (at != nullptr && at->type == strata::MetaType::STRING && at->s != "lora")
                return refuse("adapter.type is " + at->s + ", not lora");
            float alpha = 1.0f;
            if (const auto* al = f.get("adapter.lora.alpha"); al != nullptr && al->is_num()) alpha = (float) al->num();
            s.n_embd = n_embd, s.n_ff = n_ff, s.n_sh = n_sh, s.n_expert = n_expert;
            s.L.assign((size_t) n_layers, Layer{});
            s.ex_f_a_off.assign((size_t) n_layers, -1);
            s.ex_f_b_off.assign((size_t) n_layers, -1);
            int rank = 0;
            // a tensor's values appended to the half table (F16 or F32 in the file), its offset
            const auto take = [&](const strata::TensorInfo& t) -> long long {
                const long long off = (long long) s.host.size();
                const size_t n = (size_t) t.elements();
                const uint8_t* d = f.tensor_data(t);
                s.host.resize(s.host.size() + n);
                __half* o = s.host.data() + off;
                if (t.type == 1) std::memcpy(o, d, n * 2);
                else
                    for (size_t i = 0; i < n; ++i) {
                        float v;
                        std::memcpy(&v, d + 4 * i, 4);
                        o[i] = __float2half(v);
                    }
                return off;
            };
            std::string why;
            for (const auto& t : f.tensors()) {
                const std::string& nm = t.name;
                if (nm.rfind("blk.", 0) != 0) continue;
                const size_t dot = nm.find('.', 4);
                if (dot == std::string::npos) continue;
                const int il = std::atoi(nm.substr(4, dot - 4).c_str());
                const std::string rest = nm.substr(dot + 1);
                const bool is_a = rest.size() > 7 && rest.compare(rest.size() - 7, 7, ".lora_a") == 0;
                const bool is_b = rest.size() > 7 && rest.compare(rest.size() - 7, 7, ".lora_b") == 0;
                if (!is_a && !is_b) continue;
                if (t.type != 0 && t.type != 1) { why = nm + " is " + t.type_name() + " (F16 or F32 only)"; break; }
                if (il < 0 || il >= n_layers) { why = nm + ": no layer " + std::to_string(il) + " in the model"; break; }
                const std::string base = rest.substr(0, rest.size() - 7);
                const auto& sh = t.shape;
                const uint64_t r = is_a ? (sh.size() > 1 ? sh[1] : 0) : sh[0];
                if (rank == 0) rank = (int) r;
                if ((int) r != rank || rank != 1) { why = nm + ": rank " + std::to_string(r) + " (rank 1 only)"; break; }
                Layer& Ly = s.L[(size_t) il];
                const auto want = [&](std::vector<uint64_t> w) {
                    if (sh != w) {
                        std::string g, e;
                        for (auto v : sh) g += (g.empty() ? "" : "x") + std::to_string(v);
                        for (auto v : w) e += (e.empty() ? "" : "x") + std::to_string(v);
                        why = nm + " is " + g + ", the model needs " + e;
                        return false;
                    }
                    return true;
                };
                if (base == "attn_output.weight") {
                    const int in = att_in(il);
                    if (in <= 0) { why = nm + ": layer " + std::to_string(il) + " has no attention output here"; break; }
                    if (!want(is_a ? std::vector<uint64_t>{(uint64_t) in, 1} : std::vector<uint64_t>{1, (uint64_t) n_embd})) break;
                    (is_a ? Ly.att_a : Ly.att_b) = take(t);
                    Ly.att_in = in;
                } else if (base == "ffn_down_shexp.weight") {
                    if (!want(is_a ? std::vector<uint64_t>{(uint64_t) n_sh, 1} : std::vector<uint64_t>{1, (uint64_t) n_embd})) break;
                    (is_a ? Ly.sh_a : Ly.sh_b) = take(t);
                } else if (base == "ffn_down_exps.weight") {
                    if (!want(is_a ? std::vector<uint64_t>{(uint64_t) n_ff, 1, (uint64_t) n_expert}
                                   : std::vector<uint64_t>{1, (uint64_t) n_embd, (uint64_t) n_expert})) break;
                    (is_a ? Ly.ex_a : Ly.ex_b) = take(t);
                } else {
                    why = nm + ": not an adapted matrix this engine applies (attn_output, ffn_down_shexp, ffn_down_exps)";
                    break;
                }
            }
            if (!why.empty()) return refuse(why);
            for (int il = 0; il < n_layers; ++il) {
                Layer& Ly = s.L[(size_t) il];
                if ((Ly.att_a < 0) != (Ly.att_b < 0) || (Ly.sh_a < 0) != (Ly.sh_b < 0) || (Ly.ex_a < 0) != (Ly.ex_b < 0))
                    return refuse("layer " + std::to_string(il) + " has a lora_a without its lora_b (or the reverse)");
                s.n_att += Ly.att_a >= 0, s.n_shl += Ly.sh_a >= 0, s.n_exl += Ly.ex_a >= 0;
                if (Ly.ex_a >= 0) {   // the CPU lane's F32 copies
                    s.ex_f_a_off[(size_t) il] = (long long) s.ex_a_f.size();
                    s.ex_f_b_off[(size_t) il] = (long long) s.ex_b_f.size();
                    const size_t na = (size_t) n_expert * n_ff, nb = (size_t) n_expert * n_embd;
                    for (size_t i = 0; i < na; ++i) s.ex_a_f.push_back(__half2float(s.host[(size_t) Ly.ex_a + i]));
                    for (size_t i = 0; i < nb; ++i) s.ex_b_f.push_back(__half2float(s.host[(size_t) Ly.ex_b + i]));
                }
            }
            if (s.n_att + s.n_shl + s.n_exl == 0) return refuse("no adapted matrix in it");
            s.scale = alpha / (float) rank;
            if (const char* sc = getenv("STRATA_GLM_LORA_SCALE")) s.scale *= (float) std::atof(sc);
            s.on = true;
            std::fprintf(stderr,
                         "glm lora: %s - rank %d, alpha %.2f, scale %.2f: attention output in %d layers, shared expert "
                         "in %d, routed experts in %d (%.1f MB F16 on the GPU, %.1f MB F32 for the CPU lane)\n",
                         s.path.c_str(), rank, alpha, s.scale, s.n_att, s.n_shl, s.n_exl, s.host.size() * 2 / 1e6,
                         (s.ex_a_f.size() + s.ex_b_f.size()) * 4 / 1e6);
        } catch (const std::exception& e) {
            refuse(e.what());
        }
    });
}

/// The half table on the CURRENT device (uploaded on first use), nullptr when off.
inline const __half* table() {
    State& a = state();
    if (!a.on) return nullptr;
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(a.mu);
    if (a.dev[d] == nullptr) {
        __half* p = nullptr;
        if (cudaMalloc((void**) &p, a.host.size() * sizeof(__half)) != cudaSuccess ||
            cudaMemcpy(p, a.host.data(), a.host.size() * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "glm lora: no memory for the adapter on CUDA%d - adapter OFF\n", d);
            a.on = false;
            return nullptr;
        }
        a.dev[d] = p;
    }
    return a.dev[d];
}

inline const Layer* layer(int il) {
    State& a = state();
    if (!a.on || il < 0 || il >= (int) a.L.size()) return nullptr;
    return &a.L[(size_t) il];
}

inline bool has_attn(int il) { const Layer* L = layer(il); return L != nullptr && L->att_a >= 0; }
inline bool has_ffn(int il) { const Layer* L = layer(il); return L != nullptr && (L->sh_a >= 0 || L->ex_a >= 0); }

/// Layer il's attention output: out (n_embd) += s B (A . xq), xq the projection's q8_1 input (n_in values).
inline void attn(int il, const void* xq, int n_in, float* out, cudaStream_t st, const int* skip = nullptr) {
    const Layer* L = layer(il);
    if (L == nullptr || L->att_a < 0 || L->att_in != n_in) return;
    const __half* T = table();
    if (T == nullptr) return;
    const State& a = state();
    attn_kernel<<<16, 256, 0, st>>>((const Q8*) xq, n_in, T + L->att_a, T + L->att_b, out, a.n_embd, a.scale, skip);
}

/// Layer il's FFN: the shared expert's delta from sh_hq and each device-computed plan entry's from its row of hq.
inline void moe(int il, const unsigned long long* plan_ptr, const float* plan_w, const int* plan_id, const int* skip,
                int k, const void* hq, const void* sh_hq, float* out, cudaStream_t st) {
    const Layer* L = layer(il);
    if (L == nullptr || (L->sh_a < 0 && L->ex_a < 0)) return;
    const __half* T = table();
    if (T == nullptr) return;
    const State& a = state();
    moe_kernel<<<16, 288, 0, st>>>(plan_ptr, plan_w, plan_id, skip, k, a.n_expert, (const Q8*) hq, a.n_ff,
                                   L->ex_a >= 0 ? T + L->ex_a : nullptr, L->ex_b >= 0 ? T + L->ex_b : nullptr,
                                   (const Q8*) sh_hq, a.n_sh, L->sh_a >= 0 ? T + L->sh_a : nullptr,
                                   L->sh_b >= 0 ? T + L->sh_b : nullptr, out, a.n_embd, a.scale);
}

/// The CPU lane (host): out (n_embd) += s sum_j w[j] (A_ids[j] . h_j) B_ids[j], h_j = ff + j n_ff (lane_experts' rows).
inline void cpu_add(int il, int ne, const int* ids, const float* w, const float* ff, float* out) {
    const State& a = state();
    if (!a.on || il < 0 || il >= (int) a.L.size() || a.ex_f_a_off[(size_t) il] < 0 || ne <= 0) return;
    const float* A = a.ex_a_f.data() + a.ex_f_a_off[(size_t) il];
    const float* B = a.ex_b_f.data() + a.ex_f_b_off[(size_t) il];
    for (int j = 0; j < ne; ++j) {
        if (ids[j] < 0 || ids[j] >= a.n_expert) continue;
        const float* Ae = A + (size_t) ids[j] * a.n_ff;
        const float* h = ff + (size_t) j * a.n_ff;
        float acc = 0.0f;
        for (int i = 0; i < a.n_ff; ++i) acc += Ae[i] * h[i];
        const float c = a.scale * w[j] * acc;
        const float* Be = B + (size_t) ids[j] * a.n_embd;
        for (int r = 0; r < a.n_embd; ++r) out[r] += c * Be[r];
    }
}

}  // namespace glm_lora

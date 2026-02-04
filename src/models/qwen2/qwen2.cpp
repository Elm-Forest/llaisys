#include "qwen2.hpp"
#include "../../llaisys/llaisys_tensor.hpp" // For LlaisysTensor struct wrapper

// 引入所有算子
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include "../../utils.hpp"
#include <cstring>

namespace llaisys::models::qwen2 {

// 辅助：封装内部 Tensor 到 C API 的 opaque handle
llaisysTensor_t wrap(tensor_t t) {
    return new LlaisysTensor{t};
}

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _current_pos(0) {
    
    core::context().setDevice(device_type, device_id);

    // 1. 分配权重张量
    _in_embed = create_weight({meta.voc, meta.hs});
    _out_embed = create_weight({meta.voc, meta.hs});
    _out_norm_w = create_weight({meta.hs});

    // 2. 初始化导出结构体的单体字段
    _weights_export.in_embed = wrap(_in_embed);
    _weights_export.out_embed = wrap(_out_embed);
    _weights_export.out_norm_w = wrap(_out_norm_w);

    // 3. 分配层级数组
    _weights_export.attn_norm_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_q_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_q_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_k_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_k_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_v_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_v_b = new llaisysTensor_t[meta.nlayer];
    _weights_export.attn_o_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_norm_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_gate_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_up_w = new llaisysTensor_t[meta.nlayer];
    _weights_export.mlp_down_w = new llaisysTensor_t[meta.nlayer];

    size_t head_dim = meta.dh; // meta.hs / meta.nh;
    // Qwen2.5/DeepSeek 实际上 hidden_size 1536, heads 12 -> head_dim 128.

    for (size_t i = 0; i < meta.nlayer; ++i) {
        // Norms
        auto attn_norm = create_weight({meta.hs});
        auto mlp_norm = create_weight({meta.hs});
        _layers_input_norm.push_back(attn_norm);
        _layers_post_norm.push_back(mlp_norm);
        _weights_export.attn_norm_w[i] = wrap(attn_norm);
        _weights_export.mlp_norm_w[i] = wrap(mlp_norm);

        // Attn Weights
        auto q_w = create_weight({meta.nh * meta.dh, meta.hs});
        auto q_b = create_weight({meta.nh * meta.dh});
        auto k_w = create_weight({meta.nkvh * meta.dh, meta.hs});
        auto k_b = create_weight({meta.nkvh * meta.dh});
        auto v_w = create_weight({meta.nkvh * meta.dh, meta.hs});
        auto v_b = create_weight({meta.nkvh * meta.dh});
        auto o_w = create_weight({meta.hs, meta.nh * meta.dh}); // Out proj input dim is hidden size? No, input is heads*head_dim
        
        _layers_q_w.push_back(q_w); _weights_export.attn_q_w[i] = wrap(q_w);
        _layers_q_b.push_back(q_b); _weights_export.attn_q_b[i] = wrap(q_b);
        _layers_k_w.push_back(k_w); _weights_export.attn_k_w[i] = wrap(k_w);
        _layers_k_b.push_back(k_b); _weights_export.attn_k_b[i] = wrap(k_b);
        _layers_v_w.push_back(v_w); _weights_export.attn_v_w[i] = wrap(v_w);
        _layers_v_b.push_back(v_b); _weights_export.attn_v_b[i] = wrap(v_b);
        _layers_o_w.push_back(o_w); _weights_export.attn_o_w[i] = wrap(o_w);

        // MLP Weights
        auto g_w = create_weight({meta.di, meta.hs});
        auto u_w = create_weight({meta.di, meta.hs});
        auto d_w = create_weight({meta.hs, meta.di});
        
        _layers_gate_w.push_back(g_w); _weights_export.mlp_gate_w[i] = wrap(g_w);
        _layers_up_w.push_back(u_w);   _weights_export.mlp_up_w[i] = wrap(u_w);
        _layers_down_w.push_back(d_w); _weights_export.mlp_down_w[i] = wrap(d_w);

        // KV Cache
        // Shape: [max_seq, n_kv_head, head_dim]
        // 初始化为零 (可选，但为了安全)
        auto k_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        auto v_c = Tensor::create({meta.maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        _k_cache.push_back(k_c);
        _v_cache.push_back(v_c);
    }
}

Qwen2Model::~Qwen2Model() {
    // 释放导出的 wrapper 结构 (内部的 shared_ptr 会自动释放 tensor 内存)
    auto free_wrappers = [](llaisysTensor_t *arr, size_t n) {
        for(size_t i=0; i<n; i++) delete arr[i];
        delete[] arr;
    };

    delete _weights_export.in_embed;
    delete _weights_export.out_embed;
    delete _weights_export.out_norm_w;

    free_wrappers(_weights_export.attn_norm_w, _meta.nlayer);
    free_wrappers(_weights_export.attn_q_w, _meta.nlayer);
    free_wrappers(_weights_export.attn_q_b, _meta.nlayer);
    free_wrappers(_weights_export.attn_k_w, _meta.nlayer);
    free_wrappers(_weights_export.attn_k_b, _meta.nlayer);
    free_wrappers(_weights_export.attn_v_w, _meta.nlayer);
    free_wrappers(_weights_export.attn_v_b, _meta.nlayer);
    free_wrappers(_weights_export.attn_o_w, _meta.nlayer);
    free_wrappers(_weights_export.mlp_norm_w, _meta.nlayer);
    free_wrappers(_weights_export.mlp_gate_w, _meta.nlayer);
    free_wrappers(_weights_export.mlp_up_w, _meta.nlayer);
    free_wrappers(_weights_export.mlp_down_w, _meta.nlayer);
}

tensor_t Qwen2Model::create_weight(const std::vector<size_t>& shape) {
    return Tensor::create(shape, _meta.dtype, _device_type, _device_id);
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights_export;
}

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    core::context().setDevice(_device_type, _device_id);
    auto &runtime = core::context().runtime();

    // 1. Prepare Inputs
    // Input Tokens [ntoken]
    auto input_tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    input_tokens->load(token_ids); // Host to Device

    // Position IDs
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    // Fill pos ids: if ntoken > 1, [0, 1, ...], else [_current_pos]
    std::vector<int64_t> pos_vec(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        if (ntoken > 1) pos_vec[i] = i; 
        else pos_vec[i] = _current_pos;
    }
    pos_ids->load(pos_vec.data());

    // 2. Embedding
    // x: [ntoken, hidden_size]
    auto x = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::embedding(x, input_tokens, _in_embed);

    // 3. Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = x; // Share pointer, logically x
        
        // --- Attention Block ---
        // Norm
        auto x_norm = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::rms_norm(x_norm, x, _layers_input_norm[i], _meta.epsilon);

        // QKV Proj
        // q: [ntoken, nh * dh]
        // k: [ntoken, nkvh * dh]
        // v: [ntoken, nkvh * dh]
        auto q_flat = Tensor::create({ntoken, _meta.nh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto k_flat = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);
        auto v_flat = Tensor::create({ntoken, _meta.nkvh * _meta.dh}, _meta.dtype, _device_type, _device_id);

        ops::linear(q_flat, x_norm, _layers_q_w[i], _layers_q_b[i]);
        ops::linear(k_flat, x_norm, _layers_k_w[i], _layers_k_b[i]);
        ops::linear(v_flat, x_norm, _layers_v_w[i], _layers_v_b[i]);

        // Reshape for RoPE and Attention
        // q: [ntoken, nh, dh]
        // k: [ntoken, nkvh, dh]
        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});

        // RoPE (In-place on Q and K usually, but ops usually take out!=in)
        // Here we use out=in or new tensors. Let's use new tensors to be safe or overwrite.
        // Ops definition: rope(out, in, ...).
        ops::rope(q, q, pos_ids, _meta.theta);
        ops::rope(k, k, pos_ids, _meta.theta);

        // Update KV Cache
        // Copy current k/v to cache at _current_pos
        // _k_cache[i] is [max_seq, nkvh, dh]
        // We slice the cache to get the destination window
        auto k_cache_dst = _k_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        auto v_cache_dst = _v_cache[i]->slice(0, _current_pos, _current_pos + ntoken);
        
        // Copy data. Since there is no `copy` op, we use memcpy via runtime.
        // Dst and Src are both on device and contiguous (slice of dim 0 is contiguous).
        runtime.api()->memcpy_sync(
            k_cache_dst->data(), k->data(), k->numel() * k->elementSize(), LLAISYS_MEMCPY_D2D
        );
        runtime.api()->memcpy_sync(
            v_cache_dst->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D
        );

        // Prepare inputs for Attention
        // Q: [ntoken, nh, dh]
        // K_total: [current_pos + ntoken, nkvh, dh] (View of cache)
        // V_total: [current_pos + ntoken, nkvh, dh]
        auto k_total = _k_cache[i]->slice(0, 0, _current_pos + ntoken);
        auto v_total = _v_cache[i]->slice(0, 0, _current_pos + ntoken);

        // Attn Output
        auto attn_out = Tensor::create({ntoken, _meta.nh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
        ops::self_attention(attn_out, q, k_total, v_total, scale);

        // Flatten Attn Output: [ntoken, nh*dh]
        auto attn_out_flat = attn_out->view({ntoken, _meta.nh * _meta.dh});

        // O Proj
        // out: [ntoken, hs]
        auto h_attn = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        // Note: o_proj usually has no bias in Qwen2, but our struct has no bias field for o_proj anyway.
        ops::linear(h_attn, attn_out_flat, _layers_o_w[i], nullptr);

        // Residual Add
        // x = residual + h_attn
        ops::add(x, residual, h_attn);
        residual = x;

        // --- MLP Block ---
        // Norm
        ops::rms_norm(x_norm, x, _layers_post_norm[i], _meta.epsilon);

        // Gate & Up
        auto gate = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        auto up = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::linear(gate, x_norm, _layers_gate_w[i], nullptr);
        ops::linear(up, x_norm, _layers_up_w[i], nullptr);

        // SwiGLU
        // act = swiglu(gate, up) -> stores result in gate usually? No, `out` arg.
        // We reuse `up` memory for output or create new? Swiglu out has same shape.
        auto act = Tensor::create({ntoken, _meta.di}, _meta.dtype, _device_type, _device_id);
        ops::swiglu(act, gate, up);

        // Down
        auto h_mlp = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
        ops::linear(h_mlp, act, _layers_down_w[i], nullptr);

        // Residual Add
        ops::add(x, residual, h_mlp);
    }

    // 4. Final Norm
    auto x_final = Tensor::create({ntoken, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(x_final, x, _out_norm_w, _meta.epsilon);

    // 5. LM Head & Argmax
    // We only need the last token's logits for generation
    // slice input x_final to take the last row: [1, hs]
    auto x_last = x_final->slice(0, ntoken - 1, ntoken);

    // logits: [1, vocab]
    auto logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, x_last, _out_embed, nullptr); // Shared weights with in_embed usually? Struct has out_embed.

    // Argmax
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    ops::argmax(max_idx, max_val, logits->view({_meta.voc})); // View as 1D

    // Copy result to host
    int64_t next_token = 0;
    // Runtime API memcpy D2H
    runtime.api()->memcpy_sync(&next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);

    // Update global position
    _current_pos += ntoken;

    return next_token;
}

} // namespace llaisys::models::qwen2
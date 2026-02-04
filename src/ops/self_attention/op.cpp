#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    

    ASSERT(q->ndim() == 3, "SelfAttention: q must be 3D.");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be 3D.");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be 3D.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be 3D.");

    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    ASSERT(k->shape()[2] == d, "SelfAttention: k dim 2 must match q dim 2 (d).");
    ASSERT(v->shape()[0] == total_len, "SelfAttention: v dim 0 must match k dim 0 (total_len).");
    ASSERT(v->shape()[1] == nkvhead, "SelfAttention: v dim 1 must match k dim 1 (nkvhead).");

    ASSERT(attn_val->shape()[0] == seqlen, "SelfAttention: output seqlen mismatch.");
    ASSERT(attn_val->shape()[1] == nhead, "SelfAttention: output nhead mismatch.");
    ASSERT(attn_val->shape()[2] == dv, "SelfAttention: output dv mismatch.");

    ASSERT(nhead % nkvhead == 0, "SelfAttention: nhead must be divisible by nkvhead (GQA).");
    ASSERT(total_len >= seqlen, "SelfAttention: total_len (history) cannot be smaller than current seqlen.");


    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());


    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), 
           "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                   attn_val->dtype(), 
                                   seqlen, total_len, nhead, nkvhead, d, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), 
                                   attn_val->dtype(), 
                                   seqlen, total_len, nhead, nkvhead, d, dv, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
}
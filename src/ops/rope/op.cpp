#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    ASSERT(in->ndim() == 3, "RoPE: input must be 3D [seqlen, nhead, d].");
    ASSERT(out->ndim() == 3, "RoPE: output must be 3D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D [seqlen].");

    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(out->shape()[0] == seqlen && out->shape()[1] == nhead && out->shape()[2] == head_dim, 
           "RoPE: output shape must match input shape.");
    ASSERT(pos_ids->shape()[0] == seqlen, "RoPE: pos_ids dimension must match sequence length.");
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    // Dtype Checks
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be INT64.");

    // Contiguity
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), 
           "RoPE: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                         out->dtype(), seqlen, nhead, head_dim, theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), 
                         out->dtype(), seqlen, nhead, head_dim, theta);
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
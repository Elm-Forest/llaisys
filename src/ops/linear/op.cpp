#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    ASSERT(in->ndim() == 2, "Linear: input must be 2D.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Linear: output must be 2D.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "Linear: weight dim 1 must match input dim 1 (K).");
    ASSERT(out->shape()[0] == M, "Linear: output dim 0 must match input dim 0 (M).");
    ASSERT(out->shape()[1] == N, "Linear: output dim 1 must match weight dim 0 (N).");

    if (bias && bias->numel() > 0) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be 1D.");
        ASSERT(bias->shape()[0] == N, "Linear: bias dim must match output dim 1 (N).");
    }

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");

    const std::byte* bias_data = (bias && bias->numel() > 0) ? bias->data() : nullptr;

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), M, N, K);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data,
                           out->dtype(), M, N, K);
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
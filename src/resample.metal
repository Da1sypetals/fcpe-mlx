#include <metal_stdlib>
using namespace metal;

kernel void resample_sinc_hann(
    device const float *padded_input [[buffer(0)]],
    device float *output             [[buffer(1)]],
    device const float *kernels      [[buffer(2)]],
    constant int &kernel_len         [[buffer(3)]],
    constant int &orig_freq          [[buffer(4)]],
    constant int &new_freq_i         [[buffer(5)]],
    constant int &target_len         [[buffer(6)]],
    uint gid          [[thread_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tg_size      [[threads_per_threadgroup]],
    threadgroup float *shared_input  [[threadgroup(0)]]
)
{
    if ((int)gid >= target_len) return;

    int i = (int)gid / new_freq_i;
    int j = (int)gid % new_freq_i;

    uint base_gid = gid - tid;
    int base_i = (int)base_gid / new_freq_i;
    int local_i = i - base_i;

    int shared_len = ((int)tg_size - 1) * orig_freq + kernel_len;

    for (int k = (int)tid; k < shared_len; k += (int)tg_size) {
        shared_input[k] = padded_input[base_i * orig_freq + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int start = local_i * orig_freq;
    device const float *kernel_j = kernels + j * kernel_len;

    float sum = 0.0f;
    int k = 0;
    int kernel_len4 = kernel_len & ~3;

    for (; k < kernel_len4; k += 4) {
        float4 s = float4(
            shared_input[start + k],
            shared_input[start + k + 1],
            shared_input[start + k + 2],
            shared_input[start + k + 3]
        );
        float4 w = float4(
            kernel_j[k],
            kernel_j[k + 1],
            kernel_j[k + 2],
            kernel_j[k + 3]
        );
        sum += dot(s, w);
    }
    for (; k < kernel_len; k++) {
        sum += shared_input[start + k] * kernel_j[k];
    }

    output[i * new_freq_i + j] = sum;
}

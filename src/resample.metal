#include <metal_stdlib>
using namespace metal;

kernel void resample_sinc_hann(
    device const float *padded_input [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const float *kernels [[buffer(2)]],
    constant int &kernel_len [[buffer(3)]],
    constant int &orig_freq [[buffer(4)]],
    constant int &new_freq_i [[buffer(5)]],
    constant int &target_len [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
)
{
    if ((int)gid >= target_len) return;

    int i = (int)gid / new_freq_i;
    int j = (int)gid % new_freq_i;
    int out_idx = i * new_freq_i + j;
    if (out_idx >= target_len) return;

    int start = i * orig_freq;
    device const float *kernel_j = kernels + j * kernel_len;

    float sum = 0.0;
    for (int k = 0; k < kernel_len; k++) {
        float sample = padded_input[start + k];
        sum += sample * kernel_j[k];
    }
    output[out_idx] = sum;
}

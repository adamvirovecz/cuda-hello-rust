#include <cstddef>

#include <cuda_runtime.h>

extern "C" __global__ void vector_add_kernel(const float* a, const float* b, float* c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

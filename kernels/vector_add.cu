#include <cstddef>

#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

static inline __host__ __device__ uint32_t idiv_up(uint32_t a, uint32_t b) {
    uint32_t d = a / b;
    return ((a % b) == 0) ? d : (d + 1);
}

extern "C" cudaError_t launch_vector_add(
    const float* d_a,
    const float* d_b,
    float* d_c,
    uint32_t n
) {
    uint32_t block = 256;
    uint32_t grid = idiv_up(n, 256);
    vector_add_kernel<<<block, grid>>>(d_a, d_b, d_c, n);
    return cudaDeviceSynchronize();
}
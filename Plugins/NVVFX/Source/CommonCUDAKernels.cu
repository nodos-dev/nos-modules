#include "CommonCUDAKernels.h"

__global__ void NormalizeKernel(int* data, int max, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        data[idx] = (data[idx]) - max;
    }
}
    void NormalizeKernelWrapper(dim3 blocks, dim3 thread, int* data, int max, int dataSize)
    {
        NormalizeKernel<<<blocks, thread >>> (data, max, dataSize);
    }
